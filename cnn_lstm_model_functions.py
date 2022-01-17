from matplotlib import pyplot
from sklearn.model_selection import KFold
import math
import numpy as np

# PyTorch libraries and modules
import torch
from torch.utils.data import Dataset
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Flatten, Softmax, Dropout, LSTM
from hyperopt import STATUS_OK
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchmetrics.functional import f1


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
class mriCNN_LSTMBase(Module):

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
class mriCNN_LSTM(mriCNN_LSTMBase):

    def __init__(self, con_out, con_kernel, con_padding, lstm_in, lstm_hidden, linear_1_in, linear_1_out):
        super(mriCNN_LSTM, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(50, con_out, kernel_size=con_kernel, stride=(1, 1), padding=con_padding),
            Dropout(p=0.5),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Flatten(start_dim=2),
            # Conv2d(con_out, con_out, kernel_size=con_kernel, stride=(1, 1), padding=con_padding),
            # ReLU(inplace=True),
            # MaxPool2d(kernel_size=2, stride=2),
        )

        self.lstm = LSTM(lstm_in, lstm_hidden, num_layers=1, bidirectional=True)

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
        x, (h_n, c_n) = self.lstm(x)
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
def plot_learning_curve(history, dataset_name, current_fold=None):
    # plot the training and validation loss per fold
    if current_fold is None:
        pyplot.plot([x['train']['loss'] for x in history], '-bx', label='Training Loss')
        pyplot.plot([x['val']['loss'] for x in history], '-rx', label='Test Loss')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Loss')
        pyplot.title('Training and Test Loss')
        plot_filepath = 'cnn_lstm_model_evaluation/Training_and_Test_Loss_' + dataset_name + '.pdf'
        pyplot.savefig(plot_filepath)
        pyplot.close()

        pyplot.plot([x['train']['acc'] for x in history], '-bx', label='Training Accuracy')
        pyplot.plot([x['val']['acc'] for x in history], '-rx', label='Test Accuracy')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Accuracy')
        pyplot.title('Training and Test Accuracy')
        plot_filepath = 'cnn_lstm_model_evaluation/Training_and_Test_Accuracy_' + dataset_name + '.pdf'
        pyplot.savefig(plot_filepath)
        pyplot.close()

    else:
        pyplot.plot([x['train']['loss'] for x in history], '-bx', label='Training Loss')
        pyplot.plot([x['val']['loss'] for x in history], '-rx', label='Validation Loss')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Loss')
        pyplot.title('Training and Cross-Validation Loss (Fold ' + str(current_fold) + ')')
        plot_filepath = 'cnn_lstm_model_tuning/Training_and_CV_Loss_Fold_' + str(current_fold) + '_' + dataset_name + '.pdf'
        pyplot.savefig(plot_filepath)
        pyplot.close()

        pyplot.plot([x['train']['acc'] for x in history], '-bx', label='Training Accuracy')
        pyplot.plot([x['val']['acc'] for x in history], '-rx', label='Validation Accuracy')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Accuracy')
        plot_filepath = 'cnn_lstm_model_tuning/Training_and_CV_Accuracy_Fold_' + str(current_fold) + '_' + dataset_name + '.pdf'
        pyplot.savefig(plot_filepath)
        pyplot.close()


# Purpose: Plot the learning curve for each fold of a k-fold cross-validation
def plot_learning_curve_cv(history_list, dataset_name):
    # plot the training and validation loss per fold
    fold = 1
    for history in history_list:
        plot_learning_curve(history, dataset_name, current_fold=fold)
        fold += 1


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
    for parameter_name in ['con_out', 'con_kernel', 'con_padding', 'lstm_hidden', 'linear_1_out', 'batch_size']:
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
        con_1_shape = int(math.floor((params['image_width'] - (params['con_kernel'] - 1) + 2 * params['con_padding']) / 2))
        lstm_in = con_1_shape * con_1_shape

        linear_1_in = params['con_out'] * params['lstm_hidden'] * 2

        model = mriCNN_LSTM(params["con_out"], params["con_kernel"], params["con_padding"], lstm_in,
                            params["lstm_hidden"], linear_1_in, params["linear_1_out"])

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
        lr = 0.0005
        n_epochs = params['epochs']

        training_data = mriData(x_train, y_train)
        validation_data = mriData(x_val, y_val)

        train_dataloader = DataLoader(training_data, batch_size=params['batch_size'], shuffle=True)
        validation_dataloader = DataLoader(validation_data, batch_size=params['batch_size'], shuffle=False)

        history = fit(n_epochs, lr, model, train_dataloader, validation_dataloader, optimizer)

        train_acc = [x['train']['acc'] for x in history]
        val_acc = [x['val']['acc'] for x in history]
        loss = [x['train']['loss'] for x in history]
        val_loss = [x['val']['loss'] for x in history]
        tpe_loss = [1.0 - x for x in val_acc]

        train_accuracy_list.append(train_acc)
        val_accuracy_list.append(val_acc)
        loss_list.append(loss)
        val_loss_list.append(val_loss)
        tpe_loss_list.append(tpe_loss)
        history_list.append(history)

        del model
        del history

    tpe_losses = np.mean(np.array([tpe_loss_list[0], tpe_loss_list[1], tpe_loss_list[2]]), axis=0)

    tpe_loss = min(tpe_losses)

    print('CNN Filter Size: ' + str(params['con_out']) + ' CNN Kernel Size: ' + str(params["con_kernel"]) +
          ' CNN Padding: ' + str(params['con_padding']) + 'LSTM Units: ' + str(params['lstm_hidden']) +
          'Linear #1 Units: ' + str(params['linear_1_out']) + ' Epochs: ' + str(params['epochs']) +
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
        con_1_shape = int(math.floor((params['image_width'] - (params['con_kernel'] - 1) + 2 * params['con_padding']) / 2))
        lstm_in = con_1_shape * con_1_shape

        linear_1_in = params['con_out'] * params['lstm_hidden'] * 2

        model = mriCNN_LSTM(params["con_out"], params["con_kernel"], params["con_padding"], lstm_in,
                            params["lstm_hidden"], linear_1_in, params["linear_1_out"])
        # print(summary(model, (64, 192, 192)))
        if torch.cuda.is_available():
            print('GPU In Use')
            model.cuda()
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            x_val = x_val.cuda()
            y_val = y_val.cuda()

        # defining the optimizer, learning rate, and number of epochs
        # optimizer = Adam
        optimizer = SGD
        lr = 0.0005
        n_epochs = params['n_epochs']

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

    train_losses = np.mean(np.array([loss_list[0], loss_list[1], loss_list[2]]), axis=0)
    train_accuracy = np.mean(np.array([train_accuracy_list[0], train_accuracy_list[1], train_accuracy_list[2]]), axis=0)

    cv_losses = np.mean(np.array([val_loss_list[0], val_loss_list[1], val_loss_list[2]]), axis=0)
    cv_accuracy = np.mean(np.array([val_accuracy_list[0], val_accuracy_list[1], val_accuracy_list[2]]), axis=0)

    pyplot.plot(train_losses, '-bx', label='Training Loss')
    pyplot.plot(cv_losses, '-rx', label='Validation Loss')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Cross-Entropy Loss')
    pyplot.title('Training and 3-fold Cross-Validation Loss (Aggregate)')
    plot_filepath = 'cnn_lstm_model_tuning/Training_and_CV_Loss_Aggregate_' + params['dataset_name'] + '.pdf'
    pyplot.savefig(plot_filepath)
    pyplot.close()

    pyplot.plot(train_accuracy, '-bx', label='Training Accuracy')
    pyplot.plot(cv_accuracy, '-rx', label='Validation Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.title('Training and 3-fold Cross-Validation Accuracy (Aggregate)')
    plot_filepath = 'cnn_lstm_model_tuning/Training_and_CV_Accuracy_Aggregate_' + params['dataset_name'] + '.pdf'
    pyplot.savefig(plot_filepath)
    pyplot.close()

    best_index = 0
    best_cv_accuracy = cv_accuracy[best_index]
    for index in range(1, len(cv_accuracy)):
        if cv_accuracy[index] > best_cv_accuracy:
            best_cv_accuracy = cv_accuracy[index]
            best_index = index

    overall_accuracy = cv_accuracy[best_index]
    # Return a dictionary with information for the evaluation
    cv_results = {
        'params': params,
        'loss': loss_list,
        'acc': train_accuracy_list,
        'val_loss': val_loss_list,
        'val_acc': val_accuracy_list,
        'history_list': history_list,
        'cv_accuracy': overall_accuracy,
        'best_n_epochs': best_index + 1
    }

    return cv_results
