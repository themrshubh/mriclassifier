from mri_data_functions import *
from cnn_model_functions import *
from hyperopt import hp, tpe, Trials, fmin
from sklearn.model_selection import train_test_split
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

    num_images = 60
    image_length = 192
    image_width = 192

    data = retrieve_patient_data(dataset_name, num_images, image_length, image_width)  #, resample=False)
    data_labels = retrieve_train_labels()

    # create our own test dataset so we do not have to make submissions to Kaggle to get results
    x_train_cv, x_test, y_train_cv, y_test = train_test_split(data, data_labels, test_size=0.15,
                                                              random_state=42)

    print('Number of Negative Entries (Train and CV):', y_train_cv.tolist().count(0))
    print('Number of Positive Entries (Train and CV):', y_train_cv.tolist().count(1))

    print('Number of Negative Entries (Test):', y_test.tolist().count(0))
    print('Number of Positive Entries (Test):', y_test.tolist().count(1))

    print('Processing complete. Running Classifier...')

    # Define the search space
    space = {
        'con_out': hp.choice('con_out', [1, 15, 30]),
        'con_kernel': hp.choice('con_kernel', [4, 8, 12, 16]),
        'con_padding': 0,
        'linear_1_out': hp.choice('linear_1_out', [100, 200, 300, 400]),
        'batch_size': 50,
        'epochs': 50,
        'image_width': image_width,
        'x_train': x_train_cv,
        'y_train': y_train_cv,
    }

    # Keep track of the results
    bayes_trials = Trials()

    # Run optimization
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=48, trials=bayes_trials)

    print('The best parameters for {} are {}'.format(dataset_name, best))

    # Sort the trials with lowest loss first.
    bayes_trials_results = sorted(bayes_trials.results, key=lambda x: x['loss'])

    cv_history_list = bayes_trials_results[0]['history_list']
    plot_learning_curve_cv(cv_history_list, dataset_name)


if __name__ == '__main__':
    main()

    # Define the search space
#    space = {
    #        'con_out': hp.choice('con_out', [1, 15, 30]),
    #        'con_kernel': hp.choice('con_kernel', [4, 8, 12, 16]),
    #       'con_padding': 0,
    #     'linear_1_out': hp.choice('linear_1_out', [100, 200, 300, 400]),
    #        'batch_size': 50,
    #       'epochs': 50,
    #       'image_width': image_width,
    #      'x_train': x_train_cv,
    #       'y_train': y_train_cv,
#   }

# Without Re-Sampling
# The best parameters for FLAIR are {'con_kernel': 1, 'con_out': 0, 'linear_1_out': 3}
# The best parameters for T1w are {'con_kernel': 0, 'con_out': 0, 'linear_1_out': 1}
# The best parameters for T1wCE are {'con_kernel': 0, 'con_out': 0, 'linear_1_out': 3}
# The best parameters for T2w are {'con_kernel': 3, 'con_out': 0, 'linear_1_out': 3}

# With Re-Sampling
# The best parameters for FLAIR are {'con_kernel': 2, 'con_out': 0, 'linear_1_out': 2}
# The best parameters for T1w are {'con_kernel': 3, 'con_out': 0, 'linear_1_out': 0}
# The best parameters for T1wCE are {'con_kernel': 2, 'con_out': 2, 'linear_1_out': 2}
# The best parameters for T2w are {'con_kernel': 2, 'con_out': 0, 'linear_1_out': 2}

