from mri_classification_functions_simpler import *
from sklearn.model_selection import train_test_split
import math
import torch

torch.manual_seed(18)


def main():
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())
    print('CUDA availability ', torch.cuda.is_available())

    t2w_data = retrieve_patient_data('T2w')  # image_type can be: FLAIR, T1w, T1wCE, T2w
    t2w_data_labels = retrieve_train_labels()

    plot_images_together_range(t2w_data[1])   # example has biomarker
    plot_images_together_range(t2w_data[2])   # example does not have biomarker

    # create our own test dataset so we do not have to make submissions to Kaggle to get results
    x_train_cv, x_test, y_train_cv, y_test = train_test_split(t2w_data, t2w_data_labels, test_size=0.15,
                                                              random_state=42)

    print('Number of Negative Entries (Train and CV):', y_train_cv.tolist().count(0))
    print('Number of Positive Entries (Train and CV):', y_train_cv.tolist().count(1))

    print('Number of Negative Entries (Test):', y_test.tolist().count(0))
    print('Number of Positive Entries (Test):', y_test.tolist().count(1))

    print('Processing complete. Running Classifier...')

    # Define the model parameters
    params = {
        'con_out': 1,
        'con_kernel': 16,
        'con_padding': 0,
        'linear_1_out': 100,
        'batch_size': 100,
    }

    # evaluate the model on the test dataset

    # convert data to tensor format
    x_train_cv = x_train_cv.astype(float)
    x_train_cv = torch.from_numpy(x_train_cv).float()
    y_train_cv = y_train_cv.astype(int)
    y_train_cv = torch.from_numpy(y_train_cv)

    x_test = x_test.astype(float)
    x_test = torch.from_numpy(x_test).float()
    y_test = y_test.astype(int)
    y_test = torch.from_numpy(y_test)

    # defining the model
    con_1_shape = int(math.floor((128 - (params['con_kernel'] - 1) + 2 * params['con_padding']) / 2))
    # con_2_shape = int(math.floor((con_1_shape - (params['con_kernel'] - 1) + 2 * params['con_padding']) / 2))

    linear_1_in = params['con_out'] * con_1_shape * con_1_shape

    model = mriNet(params["con_out"], params["con_kernel"], params["con_padding"], linear_1_in,
                     params["linear_1_out"])
    # print(summary(model, (64, 128, 128)))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    if torch.cuda.is_available():
        print('GPU In Use')
        model.cuda()
        x_train_cv = x_train_cv.cuda()
        y_train_cv = y_train_cv.cuda()
        x_test = x_test.cuda()
        y_test = y_test.cuda()
    else:
        print('No GPU')

    # defining the optimizer, learning rate, and number of epochs
    # optimizer = Adam
    optimizer = SGD
    lr = 0.00075
    n_epochs = 25

    training_data = mriData(x_train_cv, y_train_cv)
    test_data = mriData(x_test, y_test)

    train_dataloader = DataLoader(training_data, batch_size=params['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=params['batch_size'], shuffle=False)

    history = fit(n_epochs, lr, model, train_dataloader, test_dataloader, optimizer)

    plot_learning_curve(history)

    print("Test Accuracy: {:.4f}".format(history[len(history)-1]['val']['acc']))
    print("Test F1 Score: {:.4f}".format(history[len(history) - 1]['val']['f1']))


if __name__ == '__main__':
    main()

'''Active CUDA Device: GPU 0
Available devices  1
CUDA availability  True
Number of Negative Entries (Train and CV): 234
Number of Positive Entries (Train and CV): 260
Number of Negative Entries (Test): 42
Number of Positive Entries (Test): 46
Processing complete. Running Classifier...
GPU In Use
Epoch [0], train_loss: 0.6923, train_acc: 0.5006, train_f1: 0.4622, val_loss: 0.6932, val_acc: 0.5455, val_f1: 0.5152
Epoch [1], train_loss: 0.6921, train_acc: 0.5084, train_f1: 0.4678, val_loss: 0.6932, val_acc: 0.5455, val_f1: 0.5152
Epoch [2], train_loss: 0.6920, train_acc: 0.5103, train_f1: 0.4666, val_loss: 0.6931, val_acc: 0.5455, val_f1: 0.5152
Epoch [3], train_loss: 0.6919, train_acc: 0.5161, train_f1: 0.4731, val_loss: 0.6931, val_acc: 0.5568, val_f1: 0.5301
Epoch [4], train_loss: 0.6918, train_acc: 0.5123, train_f1: 0.4644, val_loss: 0.6931, val_acc: 0.5455, val_f1: 0.5152
Epoch [5], train_loss: 0.6916, train_acc: 0.5163, train_f1: 0.4687, val_loss: 0.6931, val_acc: 0.5682, val_f1: 0.5394
Epoch [6], train_loss: 0.6915, train_acc: 0.5148, train_f1: 0.4648, val_loss: 0.6930, val_acc: 0.5341, val_f1: 0.4999
Epoch [7], train_loss: 0.6913, train_acc: 0.5104, train_f1: 0.4555, val_loss: 0.6929, val_acc: 0.5341, val_f1: 0.4999
Epoch [8], train_loss: 0.6912, train_acc: 0.5171, train_f1: 0.4592, val_loss: 0.6929, val_acc: 0.5682, val_f1: 0.5335
Epoch [9], train_loss: 0.6910, train_acc: 0.5203, train_f1: 0.4581, val_loss: 0.6929, val_acc: 0.5455, val_f1: 0.5089
Epoch [10], train_loss: 0.6909, train_acc: 0.5218, train_f1: 0.4588, val_loss: 0.6928, val_acc: 0.5455, val_f1: 0.5089
Epoch [11], train_loss: 0.6907, train_acc: 0.5220, train_f1: 0.4595, val_loss: 0.6928, val_acc: 0.5568, val_f1: 0.5179
Epoch [12], train_loss: 0.6906, train_acc: 0.5265, train_f1: 0.4642, val_loss: 0.6927, val_acc: 0.5455, val_f1: 0.5089
Epoch [13], train_loss: 0.6904, train_acc: 0.5309, train_f1: 0.4676, val_loss: 0.6927, val_acc: 0.5568, val_f1: 0.5179
Epoch [14], train_loss: 0.6903, train_acc: 0.5336, train_f1: 0.4719, val_loss: 0.6928, val_acc: 0.5455, val_f1: 0.5020
Epoch [15], train_loss: 0.6902, train_acc: 0.5345, train_f1: 0.4704, val_loss: 0.6927, val_acc: 0.5455, val_f1: 0.5020
Epoch [16], train_loss: 0.6900, train_acc: 0.5357, train_f1: 0.4681, val_loss: 0.6927, val_acc: 0.5455, val_f1: 0.5020
Epoch [17], train_loss: 0.6899, train_acc: 0.5377, train_f1: 0.4696, val_loss: 0.6926, val_acc: 0.5455, val_f1: 0.5020
Epoch [18], train_loss: 0.6898, train_acc: 0.5364, train_f1: 0.4679, val_loss: 0.6927, val_acc: 0.5455, val_f1: 0.5020
Epoch [19], train_loss: 0.6896, train_acc: 0.5405, train_f1: 0.4668, val_loss: 0.6926, val_acc: 0.5455, val_f1: 0.5020
Epoch [20], train_loss: 0.6895, train_acc: 0.5419, train_f1: 0.4709, val_loss: 0.6926, val_acc: 0.5455, val_f1: 0.5020
Epoch [21], train_loss: 0.6894, train_acc: 0.5424, train_f1: 0.4667, val_loss: 0.6925, val_acc: 0.5795, val_f1: 0.5426
Epoch [22], train_loss: 0.6892, train_acc: 0.5399, train_f1: 0.4606, val_loss: 0.6924, val_acc: 0.5909, val_f1: 0.5518
Epoch [23], train_loss: 0.6890, train_acc: 0.5463, train_f1: 0.4667, val_loss: 0.6924, val_acc: 0.5909, val_f1: 0.5518
Epoch [24], train_loss: 0.6888, train_acc: 0.5508, train_f1: 0.4740, val_loss: 0.6923, val_acc: 0.5909, val_f1: 0.5518
Test Accuracy: 0.5909
Test F1 Score: 0.5518'''
