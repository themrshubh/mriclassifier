from mri_classification_functions_simpler import *
from sklearn.model_selection import train_test_split
import math
import torch

torch.manual_seed(18)


def main():
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())
    print('CUDA availability ', torch.cuda.is_available())

    t1wce_data = retrieve_patient_data('T1wCE')  # image_type can be: FLAIR, T1w, T1wCE, T2w
    t1wce_data_labels = retrieve_train_labels()

    # plot_images_together_range(t1wce_data[1])   # example has biomarker
    # plot_images_together_range(t1wce_data[2])   # example does not have biomarker

    # create our own test dataset so we do not have to make submissions to Kaggle to get results
    x_train_cv, x_test, y_train_cv, y_test = train_test_split(t1wce_data, t1wce_data_labels, test_size=0.15,
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
    test_dataloader = DataLoader(test_data, shuffle=False)

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
Epoch [0], train_loss: 0.6933, train_acc: 0.4899, train_f1: 0.4874, val_loss: 0.6951, val_acc: 0.5568, val_f1: 0.5568
Epoch [1], train_loss: 0.6927, train_acc: 0.4940, train_f1: 0.4927, val_loss: 0.6949, val_acc: 0.5455, val_f1: 0.5455
Epoch [2], train_loss: 0.6923, train_acc: 0.4930, train_f1: 0.4914, val_loss: 0.6948, val_acc: 0.5341, val_f1: 0.5341
Epoch [3], train_loss: 0.6918, train_acc: 0.4920, train_f1: 0.4899, val_loss: 0.6946, val_acc: 0.5227, val_f1: 0.5227
Epoch [4], train_loss: 0.6913, train_acc: 0.5040, train_f1: 0.5031, val_loss: 0.6943, val_acc: 0.5114, val_f1: 0.5114
Epoch [5], train_loss: 0.6908, train_acc: 0.5017, train_f1: 0.4990, val_loss: 0.6941, val_acc: 0.5114, val_f1: 0.5114
Epoch [6], train_loss: 0.6902, train_acc: 0.5074, train_f1: 0.5036, val_loss: 0.6938, val_acc: 0.5341, val_f1: 0.5341
Epoch [7], train_loss: 0.6895, train_acc: 0.5101, train_f1: 0.5049, val_loss: 0.6935, val_acc: 0.5341, val_f1: 0.5341
Epoch [8], train_loss: 0.6888, train_acc: 0.5171, train_f1: 0.5128, val_loss: 0.6933, val_acc: 0.5568, val_f1: 0.5568
Epoch [9], train_loss: 0.6881, train_acc: 0.5025, train_f1: 0.4952, val_loss: 0.6926, val_acc: 0.5682, val_f1: 0.5682
Epoch [10], train_loss: 0.6872, train_acc: 0.5097, train_f1: 0.5012, val_loss: 0.6923, val_acc: 0.5795, val_f1: 0.5795
Epoch [11], train_loss: 0.6857, train_acc: 0.4999, train_f1: 0.4866, val_loss: 0.6903, val_acc: 0.5909, val_f1: 0.5909
Epoch [12], train_loss: 0.6833, train_acc: 0.4987, train_f1: 0.4737, val_loss: 0.6873, val_acc: 0.5795, val_f1: 0.5795
Epoch [13], train_loss: 0.6812, train_acc: 0.5197, train_f1: 0.4925, val_loss: 0.6841, val_acc: 0.5795, val_f1: 0.5795
Epoch [14], train_loss: 0.6805, train_acc: 0.5192, train_f1: 0.4900, val_loss: 0.6824, val_acc: 0.5682, val_f1: 0.5682
Epoch [15], train_loss: 0.6797, train_acc: 0.5164, train_f1: 0.4864, val_loss: 0.6821, val_acc: 0.5682, val_f1: 0.5682
Epoch [16], train_loss: 0.6792, train_acc: 0.5163, train_f1: 0.4854, val_loss: 0.6821, val_acc: 0.5682, val_f1: 0.5682
Epoch [17], train_loss: 0.6783, train_acc: 0.5165, train_f1: 0.4872, val_loss: 0.6824, val_acc: 0.5682, val_f1: 0.5682
Epoch [18], train_loss: 0.6779, train_acc: 0.5214, train_f1: 0.4930, val_loss: 0.6828, val_acc: 0.5682, val_f1: 0.5682
Epoch [19], train_loss: 0.6773, train_acc: 0.5219, train_f1: 0.4915, val_loss: 0.6839, val_acc: 0.5795, val_f1: 0.5795
Epoch [20], train_loss: 0.6765, train_acc: 0.5270, train_f1: 0.5015, val_loss: 0.6839, val_acc: 0.5795, val_f1: 0.5795
Epoch [21], train_loss: 0.6761, train_acc: 0.5284, train_f1: 0.5005, val_loss: 0.6838, val_acc: 0.5682, val_f1: 0.5682
Epoch [22], train_loss: 0.6756, train_acc: 0.5301, train_f1: 0.5024, val_loss: 0.6841, val_acc: 0.5682, val_f1: 0.5682
Epoch [23], train_loss: 0.6751, train_acc: 0.5233, train_f1: 0.4943, val_loss: 0.6832, val_acc: 0.5455, val_f1: 0.5455
Epoch [24], train_loss: 0.6749, train_acc: 0.5253, train_f1: 0.4968, val_loss: 0.6833, val_acc: 0.5455, val_f1: 0.5455
Test Accuracy: 0.5455
Test F1 Score: 0.5455'''
