from mri_classification_functions_simpler import *
from sklearn.model_selection import train_test_split
import math
import torch

torch.manual_seed(18)


def main():
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())
    print('CUDA availability ', torch.cuda.is_available())

    t1w_data = retrieve_patient_data('T1w')  # image_type can be: FLAIR, T1w, T1wCE, T2w
    t1w_data_labels = retrieve_train_labels()

    plot_images_together_range(t1w_data[1])   # example has biomarker
    plot_images_together_range(t1w_data[2])   # example does not have biomarker

    # create our own test dataset so we do not have to make submissions to Kaggle to get results
    x_train_cv, x_test, y_train_cv, y_test = train_test_split(t1w_data, t1w_data_labels, test_size=0.15,
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
Epoch [0], train_loss: 0.6924, train_acc: 0.5105, train_f1: 0.4721, val_loss: 0.6926, val_acc: 0.5682, val_f1: 0.5625
Epoch [1], train_loss: 0.6923, train_acc: 0.5077, train_f1: 0.4684, val_loss: 0.6924, val_acc: 0.5682, val_f1: 0.5625
Epoch [2], train_loss: 0.6921, train_acc: 0.5088, train_f1: 0.4672, val_loss: 0.6923, val_acc: 0.5568, val_f1: 0.5498
Epoch [3], train_loss: 0.6919, train_acc: 0.5070, train_f1: 0.4630, val_loss: 0.6922, val_acc: 0.5455, val_f1: 0.5337
Epoch [4], train_loss: 0.6917, train_acc: 0.5119, train_f1: 0.4672, val_loss: 0.6921, val_acc: 0.5682, val_f1: 0.5493
Epoch [5], train_loss: 0.6915, train_acc: 0.5161, train_f1: 0.4672, val_loss: 0.6921, val_acc: 0.5568, val_f1: 0.5396
Epoch [6], train_loss: 0.6913, train_acc: 0.5188, train_f1: 0.4661, val_loss: 0.6920, val_acc: 0.5568, val_f1: 0.5396
Epoch [7], train_loss: 0.6911, train_acc: 0.5186, train_f1: 0.4667, val_loss: 0.6919, val_acc: 0.5455, val_f1: 0.5256
Epoch [8], train_loss: 0.6910, train_acc: 0.5186, train_f1: 0.4641, val_loss: 0.6917, val_acc: 0.5227, val_f1: 0.4967
Epoch [9], train_loss: 0.6908, train_acc: 0.5145, train_f1: 0.4558, val_loss: 0.6916, val_acc: 0.5568, val_f1: 0.5243
Epoch [10], train_loss: 0.6906, train_acc: 0.5170, train_f1: 0.4601, val_loss: 0.6915, val_acc: 0.5568, val_f1: 0.5243
Epoch [11], train_loss: 0.6904, train_acc: 0.5180, train_f1: 0.4562, val_loss: 0.6914, val_acc: 0.5455, val_f1: 0.5089
Epoch [12], train_loss: 0.6902, train_acc: 0.5197, train_f1: 0.4554, val_loss: 0.6913, val_acc: 0.5341, val_f1: 0.4857
Epoch [13], train_loss: 0.6900, train_acc: 0.5164, train_f1: 0.4512, val_loss: 0.6912, val_acc: 0.5341, val_f1: 0.4857
Epoch [14], train_loss: 0.6897, train_acc: 0.5201, train_f1: 0.4520, val_loss: 0.6912, val_acc: 0.5341, val_f1: 0.4857
Epoch [15], train_loss: 0.6895, train_acc: 0.5189, train_f1: 0.4440, val_loss: 0.6911, val_acc: 0.5568, val_f1: 0.5028
Epoch [16], train_loss: 0.6893, train_acc: 0.5223, train_f1: 0.4434, val_loss: 0.6909, val_acc: 0.5568, val_f1: 0.4940
Epoch [17], train_loss: 0.6891, train_acc: 0.5326, train_f1: 0.4493, val_loss: 0.6908, val_acc: 0.5455, val_f1: 0.4762
Epoch [18], train_loss: 0.6889, train_acc: 0.5257, train_f1: 0.4390, val_loss: 0.6907, val_acc: 0.5455, val_f1: 0.4762
Epoch [19], train_loss: 0.6886, train_acc: 0.5231, train_f1: 0.4312, val_loss: 0.6906, val_acc: 0.5341, val_f1: 0.4579
Epoch [20], train_loss: 0.6883, train_acc: 0.5291, train_f1: 0.4316, val_loss: 0.6906, val_acc: 0.5341, val_f1: 0.4579
Epoch [21], train_loss: 0.6881, train_acc: 0.5280, train_f1: 0.4272, val_loss: 0.6906, val_acc: 0.5341, val_f1: 0.4579
Epoch [22], train_loss: 0.6878, train_acc: 0.5235, train_f1: 0.4239, val_loss: 0.6905, val_acc: 0.5341, val_f1: 0.4579
Epoch [23], train_loss: 0.6876, train_acc: 0.5306, train_f1: 0.4231, val_loss: 0.6904, val_acc: 0.5341, val_f1: 0.4465
Epoch [24], train_loss: 0.6873, train_acc: 0.5349, train_f1: 0.4211, val_loss: 0.6904, val_acc: 0.5341, val_f1: 0.4465
Test Accuracy: 0.5341
Test F1 Score: 0.4465'''

