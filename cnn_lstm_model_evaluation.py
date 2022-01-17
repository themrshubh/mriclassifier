from mri_data_functions import *
from cnn_lstm_model_functions import *
from sklearn.model_selection import train_test_split
import math
import torch
import argparse

torch.manual_seed(18)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-datasetName', default='FLAIR', type=str)
    args = parser.parse_args()

    dataset_name = args.datasetName
    print(dataset_name)

    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())
    print('CUDA availability ', torch.cuda.is_available())

    num_images = 50
    image_length = 192
    image_width = 192

    # image_type can be: FLAIR, T1w, T1wCE, T2w
    data = retrieve_patient_data(dataset_name, num_images, image_length, image_width)
    data_labels = retrieve_train_labels()

    # create our own test dataset so we do not have to make submissions to Kaggle to get results
    x_train_cv, x_test, y_train_cv, y_test = train_test_split(data, data_labels, test_size=0.15,
                                                              random_state=42)

    print('Number of Negative Entries (Train and CV):', y_train_cv.tolist().count(0))
    print('Number of Positive Entries (Train and CV):', y_train_cv.tolist().count(1))

    print('Number of Negative Entries (Test):', y_test.tolist().count(0))
    print('Number of Positive Entries (Test):', y_test.tolist().count(1))

    print('Processing complete. Running Classifier...')

    # Define the model parameters
    con_out = 1
    con_kernel = 4
    lstm_hidden = 600
    linear_1_out = 300

    if dataset_name == 'FLAIR':
        con_out = 50
        linear_1_out = 200
    elif dataset_name == 'T1wCE':
        con_out = 50
        con_kernel = 16
    elif dataset_name == 'T2w':
        con_kernel = 8
        lstm_hidden = 400
        linear_1_out = 400

    params = {
        'con_out': con_out,
        'con_kernel': con_kernel,
        'con_padding': 0,
        'lstm_hidden': lstm_hidden,
        'linear_1_out': linear_1_out,
        'batch_size': 50,
        'n_epochs': 50,
        'image_width': image_width,
        'x_train': x_train_cv,
        'y_train': y_train_cv,
        'dataset_name': dataset_name
    }

    # run 3-fold cross validation using the selected parameters from model tuning to identify the best number of epochs
    cv_results = k_fold_cross_validation(params)
    n_epochs = cv_results['best_n_epochs']
    print('Number of Epochs: {}'.format(n_epochs))

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
    con_1_shape = int(math.floor((params['image_width'] - (params['con_kernel'] - 1) + 2 * params['con_padding']) / 2))
    lstm_in = con_1_shape * con_1_shape

    linear_1_in = params['con_out'] * params['lstm_hidden'] * 2

    model = mriCNN_LSTM(params["con_out"], params["con_kernel"], params["con_padding"], lstm_in,
                        params["lstm_hidden"], linear_1_in, params["linear_1_out"])
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
    lr = 0.0005

    training_data = mriData(x_train_cv, y_train_cv)
    test_data = mriData(x_test, y_test)

    train_dataloader = DataLoader(training_data, batch_size=params['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_data, shuffle=False)

    history = fit(n_epochs, lr, model, train_dataloader, test_dataloader, optimizer)

    plot_learning_curve(history, dataset_name)

    print("Test Accuracy {}: {:.4f}".format(dataset_name, history[len(history)-1]['val']['acc']))
    print("Test F1 Score {}: {:.4f}".format(dataset_name, history[len(history) - 1]['val']['f1']))


if __name__ == '__main__':
    main()

# -----With Re-Sampling-----
# Test Accuracy FLAIR: 0.5341
# Test F1 Score FLAIR: 0.5341

# Test Accuracy T1w: 0.5455
# Test F1 Score T1w: 0.5455

# Test Accuracy T1wCE: 0.5227
# Test F1 Score T1wCE: 0.5227

# Test Accuracy T2w: 0.5114
# Test F1 Score T2w: 0.5114
