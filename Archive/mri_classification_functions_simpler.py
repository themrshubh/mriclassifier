import numpy as np
import pydicom as dicom
import os
from matplotlib import pyplot
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import math
import scipy.ndimage
import cv2

# PyTorch libraries and modules
import torch
from torch.utils.data import Dataset
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Flatten, Softmax, BatchNorm2d, Dropout
from hyperopt import STATUS_OK
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchmetrics.functional import f1


# Purpose: Walk the directory of data and read in the image data for each patient identification number found.
# Note: 3 patient ID's are flagged as erroneous in Kaggle: '00109', '00123', '00709'
# image_type can be: FLAIR, T1w, T1wCE, T2w
def retrieve_patient_data(image_type):
    file_path = './train'
    samples_with_issues = ['00109', '00123', '00709']
    num_samples = 582

    list_of_patient_ids = []
    for directory, sub_directory, file_list in os.walk(file_path):
        if directory == file_path:
            list_of_patient_ids = sub_directory
    dimensions = (num_samples, 32, 128, 128)
    all_data = np.zeros(dimensions)
    counter = 0
    for identification in list_of_patient_ids:
        if identification not in samples_with_issues and counter < num_samples + 1:
            patient_data = read_patient(identification, image_type)
            all_data[counter, :, :] = patient_data
            counter += 1

    return all_data


# Purpose: Given a patient identification, retrieve the specified image data for the individual patient.
# image_type can be: FLAIR, T1w, T1wCE, T2w
def read_patient(identification, image_type):
    file_path = './train/' + identification + '/' + image_type
    patient_files_list = []
    for directory, sub_directory, file_list in os.walk(file_path):
        for filename in file_list:
            if '.dcm' in filename.lower():
                patient_files_list.append(os.path.join(directory, filename))

    # get ref file
    ref_ds = dicom.read_file(patient_files_list[0])

    # load dimensions based on the number of rows, columns, and slices (along the Z axis)
    dimensions = (len(patient_files_list), int(ref_ds.Rows), int(ref_ds.Columns))

    # Load spacing values (in mm)
    spacing = (float(ref_ds.SliceThickness), float(ref_ds.PixelSpacing[0]), float(ref_ds.PixelSpacing[1]))

    dicom_array = np.zeros(dimensions, dtype=ref_ds.pixel_array.dtype)

    # loop through all the DICOM files
    for filename in patient_files_list:
        # read the file
        ds = dicom.read_file(filename)
        # normalize and store the raw image data
        dicom_array[patient_files_list.index(filename), :, :] = ds.pixel_array / 255.0
    # new_array = resample_images(dicom_array, spacing)  # (use this line to include resampling of data)
    final_array = resize_images(dicom_array)
    return np.array(final_array)


# Purpose: The MRI images are taken at varying pixel spacing and slice depth. This standardizes the spacing across all
# images.
def resample_images(images, spacing, new_spacing=[1, 1, 1]):
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = images.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / images.shape
    images = scipy.ndimage.interpolation.zoom(images, real_resize_factor)
    return images


# Purpose: Both before and after resampling, each patient can have a varying number of images and the images may come
# in varying pixel resolution. This resizes the samples so each the data for each patient is of a consistent size prior
# to input into the neural network.
def resize_images(images):
    dimensions = (32, 128, 128)
    new_images = np.zeros(dimensions)
    length = images.shape[0]
    start = math.ceil((length - 32)/2)
    images = images[start:start+32, :, :]
    for index in range(0, images.shape[0]):
        image = images[index, :, :]
        image = cv2.resize(image, dsize=(128, 128))
        new_images[index, :, :] = image
    return new_images


# Purpose: Read in the categorical labels for the training data
def retrieve_train_labels():
    labels_df = pd.read_csv('train_labels.csv')
    samples_with_issues = [109, 123, 709]
    for sample in samples_with_issues:
        labels_df = labels_df[labels_df.BraTS21ID != sample]
    # labels_df = labels_df[labels_df.BraTS21ID.isin(samples_with_issues)]
    labels = labels_df['MGMT_value'].values
    return labels


# Purpose: Given an array of images, plot 25 of the images in a 5x5 arrangement.
def plot_images_together_range(dicom_array, rows=5, cols=5, start_with=4, show_every=1):
    fig, ax = pyplot.subplots(rows, cols, figsize=[10, 10])
    for i in range(rows * cols):
        ind = start_with + i * show_every
        ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
        ax[int(i / rows), int(i % rows)].imshow(dicom_array[ind, :, :], cmap='gray')
        ax[int(i / rows), int(i % rows)].axis('off')
    pyplot.show()


# Purpose: Create a dataloader for Pytorch
class mriData(Dataset):

    def __init__(self, x, y):
        self.data = x
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Purpose: Define the training and evaluation steps for the architecture; track and output performance metrics.
class mriNetBase(Module):

    def training_step(self, batch):
        data, labels = batch
        out = self(data)
        loss = F.cross_entropy(out, labels)
        return loss

    def evaluation_step(self, batch):
        data, labels = batch
        out = self(data)
        loss = F.cross_entropy(out, labels)
        step_accuracy = accuracy(out, labels)
        step_f1 = f_1_score(out, labels)
        return {'loss': loss.detach(), 'acc': step_accuracy, 'f1': step_f1}

    def evaluation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        batch_f1s = [x['f1'] for x in outputs]
        epoch_f1 = torch.stack(batch_f1s).mean()  # Combine f1 scores
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item(), 'f1': epoch_f1.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, train_f1: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, "
              "val_f1: {:.4f}".format(epoch, result['train']['loss'], result['train']['acc'], result['train']['f1'],
                                      result['val']['loss'], result['val']['acc'], result['val']['f1']))


# Purpose: Define the model architecture
class mriNet(mriNetBase):

    def __init__(self, con_out, con_kernel, con_padding, linear_1_in, linear_1_out):
        super(mriNet, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(32, con_out, kernel_size=con_kernel, stride=(1, 1), padding=con_padding),
            # BatchNorm2d(con_out),
            # Dropout(p=0.1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Conv2d(con_out, con_out, kernel_size=con_kernel, stride=(1, 1), padding=con_padding),
            # BatchNorm2d(32),
            # ReLU(inplace=True),
            # MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Flatten(),
            Linear(linear_1_in, linear_1_out),
            ReLU(inplace=True),
            Linear(linear_1_out, 2),
            Softmax(dim=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.linear_layers(x)
        return x


# Purpose: Calculate accuracy given predictions and truth data
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# Purpose: Calculate F1 score given predictions and labels
def f_1_score(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return f1(labels, preds, num_classes=2, average='macro')


# Purpose: Train the model and store data for learning curves
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr, weight_decay=0.001)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = {}
        result['train'] = evaluate(model, train_loader)
        result['val'] = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history


# Purpose: Evaluate the model on a dataset
def evaluate(model, loader):
    model.eval()
    outputs = [model.evaluation_step(batch) for batch in loader]
    return model.evaluation_epoch_end(outputs)


# Purpose: Plot the learning curves for a single evaluation against a test dataset
def plot_learning_curve(history, current_fold = None):
    # plot the training and validation loss per fold
    if current_fold is None:
        pyplot.plot([x['train']['loss'] for x in history], '-bx', label='Training Loss')
        pyplot.plot([x['val']['loss'] for x in history], '-rx', label='Test Loss')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Loss')
        pyplot.title('Training and Test Loss')
        pyplot.show()
        pyplot.close()

        pyplot.plot([x['train']['acc'] for x in history], '-bx', label='Training Accuracy')
        pyplot.plot([x['val']['acc'] for x in history], '-rx', label='Test Accuracy')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Accuracy')
        pyplot.title('Training and Test Accuracy')
        pyplot.show()
        pyplot.close()

    else:
        pyplot.plot([x['train']['loss'] for x in history], '-bx', label='Training Loss')
        pyplot.plot([x['val']['loss'] for x in history], '-rx', label='Validation Loss')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Loss')
        pyplot.title('Training and Cross-Validation Loss (Fold ' + str(current_fold) + ')')
        pyplot.show()
        pyplot.close()

        pyplot.plot([x['train']['acc'] for x in history], '-bx', label='Training Accuracy')
        pyplot.plot([x['val']['acc'] for x in history], '-rx', label='Validation Accuracy')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Accuracy')
        pyplot.title('Training and Cross-Validation Accuracy (Fold ' + str(current_fold) + ')')
        pyplot.show()
        pyplot.close()


# Purpose: Plot the learning curve for each fold of a k-fold cross-validation
def plot_learning_curve_cv(history_list):
    # plot the training and validation loss per fold
    fold = 1
    for history in history_list:
        plot_learning_curve(history, current_fold=fold)
        fold += 1


# Purpose: Run a k-fold cross-validation of the model and return the results.
def k_fold_cross_validation(params):
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['con_out', 'con_kernel', 'con_padding', 'linear_1_out', 'batch_size']:
        params[parameter_name] = int(params[parameter_name])

    # K-fold Cross Validation
    inputs = params['x_train']
    targets = params['y_train']

    # n_splits = 3 so 2/3 of the training/validation data will be used for training and 1/3 for validation.
    kfold = KFold(n_splits=3, shuffle=True, random_state=43)

    # Build, train, and evaluate the model using the params
    train_accuracy_list = []
    val_accuracy_list = []
    loss_list = []
    val_loss_list = []
    history_list = []

    for train, valid in kfold.split(inputs, targets):
        # convert data to tensor format
        x_train = inputs[train].astype(float)
        x_train = torch.from_numpy(x_train).float()
        y_train = targets[train].astype(int)
        y_train = torch.from_numpy(y_train)

        x_val = inputs[valid].astype(float)
        x_val = torch.from_numpy(x_val).float()
        y_val = targets[valid].astype(int)
        y_val = torch.from_numpy(y_val)

        # defining the model
        con_1_shape = int(math.floor((128 - (params['con_kernel'] - 1) + 2*params['con_padding']) / 2))
        # con_2_shape = int(math.floor((con_1_shape - (params['con_kernel'] - 1) + 2*params['con_padding']) / 2))

        linear_1_in = params['con_out'] * con_1_shape * con_1_shape

        model = mriNet(params["con_out"], params["con_kernel"], params["con_padding"], linear_1_in,
                         params["linear_1_out"])
        # print(summary(model, (64, 128, 128)))

        # if torch.cuda.is_available():
            # model.cuda()
            # x_train = x_train.cuda()
            # y_train = y_train.cuda()
            # x_val = x_val.cuda()
            # y_val = y_val.cuda()

        # defining the optimizer, learning rate, and number of epochs
        # optimizer = Adam
        optimizer = SGD
        lr = 0.001
        n_epochs = 200

        training_data = mriData(x_train, y_train)
        validation_data = mriData(x_val, y_val)

        train_dataloader = DataLoader(training_data, batch_size=params['batch_size'], shuffle=True)
        validation_dataloader = DataLoader(validation_data, batch_size=params['batch_size'], shuffle=False)

        history = fit(n_epochs, lr, model, train_dataloader, validation_dataloader, optimizer)

        train_acc = [x['train']['acc'] for x in history]
        val_acc = [x['val']['acc'] for x in history]
        loss = [x['train']['loss'] for x in history]
        val_loss = [x['val']['loss'] for x in history]

        train_accuracy_list.append(train_acc)
        val_accuracy_list.append(val_acc)
        loss_list.append(loss)
        val_loss_list.append(val_loss)
        history_list.append(history)

        del model
        del history

    overall_accuracy = sum(val_accuracy_list) / len(val_accuracy_list)
    # Return a dictionary with information for the evaluation
    cv_results = {
        'params': params,
        'loss': loss_list,
        'acc': train_accuracy_list,
        'val_loss': val_loss_list,
        'val_acc': val_accuracy_list,
        'history_list': history_list,
        'cv_accuracy': overall_accuracy,
    }

    return cv_results


# Code for Bayesian Optimization derived from the following references:
#
# “An Introductory Example of Bayesian Optimization in Python with Hyperopt | by Will Koehrsen | Towards Data Science.”
# https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0
#
# D. D. Labs, “Parameter Tuning with Hyperopt,” Medium, Dec. 23, 2017.
# https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce

# Code for K-fold Cross Validation derived from the following reference:
#
# “K-fold Cross Validation with TensorFlow and Keras,” MachineCurve, Feb. 18, 2020.
# https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/


# Define the objective function
def objective(params):

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['con_out', 'con_kernel', 'con_padding', 'linear_1_out', 'batch_size']:
        params[parameter_name] = int(params[parameter_name])

    # K-fold Cross Validation
    inputs = params['x_train']
    targets = params['y_train']

    # n_splits = 3 so 2/3 of the training/validation data will be used for training and 1/3 for validation.
    kfold = KFold(n_splits=3, shuffle=True, random_state=43)

    # Build, train, and evaluate the model using the params
    train_accuracy_list = []
    val_accuracy_list = []
    loss_list = []
    val_loss_list = []
    history_list = []
    tpe_loss_list = []

    for train, valid in kfold.split(inputs, targets):
        # convert data to tensor format
        x_train = inputs[train].astype(float)
        x_train = torch.from_numpy(x_train).float()
        y_train = targets[train].astype(int)
        y_train = torch.from_numpy(y_train)

        x_val = inputs[valid].astype(float)
        x_val = torch.from_numpy(x_val).float()
        y_val = targets[valid].astype(int)
        y_val = torch.from_numpy(y_val)

        # defining the model
        con_1_shape = int(math.floor((128 - (params['con_kernel'] - 1) + 2*params['con_padding']) / 2))
        # con_2_shape = int(math.floor((con_1_shape - (params['con_kernel'] - 1) + 2*params['con_padding']) / 2))

        linear_1_in = params['con_out'] * con_1_shape * con_1_shape

        model = mriNet(params["con_out"], params["con_kernel"], params["con_padding"], linear_1_in,
                         params["linear_1_out"])
        # print(summary(model, (64, 128, 128)))

        if torch.cuda.is_available():
            print('GPU In Use')
            model.cuda()
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            x_val = x_val.cuda()
            y_val = y_val.cuda()
        else:
            print('No GPU')

        # defining the optimizer, learning rate, and number of epochs
        # optimizer = Adam
        optimizer = SGD
        lr = 0.001
        n_epochs = 200

        training_data = mriData(x_train, y_train)
        validation_data = mriData(x_val, y_val)

        train_dataloader = DataLoader(training_data, batch_size=params['batch_size'], shuffle=True)
        validation_dataloader = DataLoader(validation_data, batch_size=params['batch_size'], shuffle=False)

        history = fit(n_epochs, lr, model, train_dataloader, validation_dataloader, optimizer)

        train_acc = [x['train']['acc'] for x in history]
        val_acc = [x['val']['acc'] for x in history]
        loss = [x['train']['loss'] for x in history]
        val_loss = [x['val']['loss'] for x in history]
        tpe_loss = 1 - max(val_acc)

        train_accuracy_list.append(train_acc)
        val_accuracy_list.append(val_acc)
        loss_list.append(loss)
        val_loss_list.append(val_loss)
        tpe_loss_list.append(tpe_loss)
        history_list.append(history)

        del model
        del x_train
        del y_train
        del x_val
        del y_val
        del history

    tpe_loss = sum(tpe_loss_list) / len(tpe_loss_list)
    print('CNN Filter Size: ' + str(params['con_out']) + ' CNN Kernel Size: ' + str(params["con_kernel"]) +
          ' CNN Padding: ' + str(params['con_padding']) + ' Linear #1 Units: ' + str(params['linear_1_out']) +
          ' TPE Loss: ' + str(tpe_loss))
    # Return a dictionary with information for the evaluation
    return {
        'loss': tpe_loss,
        'params': params,
        'val_accuracy_list': val_accuracy_list,
        'loss_list': loss_list,
        'val_loss_list': val_loss_list,
        'history_list': history_list,
        'status': STATUS_OK
    }
