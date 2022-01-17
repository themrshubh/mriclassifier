from mri_classification_functions_simpler import *
from sklearn.model_selection import train_test_split
import math
import torch

torch.manual_seed(18)


def main():
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())
    print('CUDA availability ', torch.cuda.is_available())

    flair_data = retrieve_patient_data('FLAIR')  # image_type can be: FLAIR, T1w, T1wCE, T2w
    flair_data_labels = retrieve_train_labels()

    plot_images_together_range(flair_data[1])   # example has biomarker
    plot_images_together_range(flair_data[2])   # example does not have biomarker

    # create our own test dataset so we do not have to make submissions to Kaggle to get results
    x_train_cv, x_test, y_train_cv, y_test = train_test_split(flair_data, flair_data_labels, test_size=0.15,
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
Epoch [0], train_loss: 0.6931, train_acc: 0.4715, train_f1: 0.4680, val_loss: 0.6931, val_acc: 0.5568, val_f1: 0.5563
Epoch [1], train_loss: 0.6929, train_acc: 0.4757, train_f1: 0.4741, val_loss: 0.6931, val_acc: 0.5682, val_f1: 0.5673
Epoch [2], train_loss: 0.6928, train_acc: 0.4726, train_f1: 0.4679, val_loss: 0.6932, val_acc: 0.5682, val_f1: 0.5673
Epoch [3], train_loss: 0.6928, train_acc: 0.4777, train_f1: 0.4742, val_loss: 0.6932, val_acc: 0.5795, val_f1: 0.5782
Epoch [4], train_loss: 0.6927, train_acc: 0.4799, train_f1: 0.4779, val_loss: 0.6932, val_acc: 0.5795, val_f1: 0.5782
Epoch [5], train_loss: 0.6927, train_acc: 0.4785, train_f1: 0.4768, val_loss: 0.6932, val_acc: 0.5795, val_f1: 0.5782
Epoch [6], train_loss: 0.6927, train_acc: 0.4776, train_f1: 0.4754, val_loss: 0.6932, val_acc: 0.5682, val_f1: 0.5662
Epoch [7], train_loss: 0.6926, train_acc: 0.4772, train_f1: 0.4733, val_loss: 0.6932, val_acc: 0.5795, val_f1: 0.5769
Epoch [8], train_loss: 0.6926, train_acc: 0.4781, train_f1: 0.4754, val_loss: 0.6932, val_acc: 0.5795, val_f1: 0.5769
Epoch [9], train_loss: 0.6926, train_acc: 0.4841, train_f1: 0.4824, val_loss: 0.6932, val_acc: 0.5795, val_f1: 0.5769
Epoch [10], train_loss: 0.6925, train_acc: 0.4835, train_f1: 0.4814, val_loss: 0.6932, val_acc: 0.5795, val_f1: 0.5769
Epoch [11], train_loss: 0.6925, train_acc: 0.4837, train_f1: 0.4802, val_loss: 0.6932, val_acc: 0.5682, val_f1: 0.5646
Epoch [12], train_loss: 0.6925, train_acc: 0.4792, train_f1: 0.4787, val_loss: 0.6932, val_acc: 0.5568, val_f1: 0.5521
Epoch [13], train_loss: 0.6924, train_acc: 0.4799, train_f1: 0.4778, val_loss: 0.6932, val_acc: 0.5568, val_f1: 0.5521
Epoch [14], train_loss: 0.6924, train_acc: 0.4797, train_f1: 0.4755, val_loss: 0.6932, val_acc: 0.5455, val_f1: 0.5395
Epoch [15], train_loss: 0.6924, train_acc: 0.4852, train_f1: 0.4834, val_loss: 0.6932, val_acc: 0.5568, val_f1: 0.5521
Epoch [16], train_loss: 0.6923, train_acc: 0.4879, train_f1: 0.4856, val_loss: 0.6932, val_acc: 0.5682, val_f1: 0.5625
Epoch [17], train_loss: 0.6923, train_acc: 0.4919, train_f1: 0.4866, val_loss: 0.6932, val_acc: 0.5455, val_f1: 0.5368
Epoch [18], train_loss: 0.6923, train_acc: 0.4917, train_f1: 0.4903, val_loss: 0.6933, val_acc: 0.5455, val_f1: 0.5368
Epoch [19], train_loss: 0.6922, train_acc: 0.4851, train_f1: 0.4835, val_loss: 0.6933, val_acc: 0.5455, val_f1: 0.5368
Epoch [20], train_loss: 0.6922, train_acc: 0.4864, train_f1: 0.4813, val_loss: 0.6933, val_acc: 0.5455, val_f1: 0.5368
Epoch [21], train_loss: 0.6922, train_acc: 0.4816, train_f1: 0.4781, val_loss: 0.6933, val_acc: 0.5455, val_f1: 0.5368
Epoch [22], train_loss: 0.6921, train_acc: 0.4855, train_f1: 0.4811, val_loss: 0.6933, val_acc: 0.5455, val_f1: 0.5368
Epoch [23], train_loss: 0.6921, train_acc: 0.4810, train_f1: 0.4782, val_loss: 0.6933, val_acc: 0.5341, val_f1: 0.5237
Epoch [24], train_loss: 0.6921, train_acc: 0.4791, train_f1: 0.4744, val_loss: 0.6933, val_acc: 0.5341, val_f1: 0.5237
Test Accuracy: 0.5341
Test F1 Score: 0.5237'''

