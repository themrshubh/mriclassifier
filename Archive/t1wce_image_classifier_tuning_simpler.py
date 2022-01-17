from mri_classification_functions_simpler import *
from hyperopt import hp, tpe, Trials, fmin
from sklearn.model_selection import train_test_split
import torch

torch.manual_seed(18)


def main():
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())
    print('CUDA availability ', torch.cuda.is_available())

    t1cwe_data = retrieve_patient_data('T1wCE')
    t1cwe_data_labels = retrieve_train_labels()

    # create our own test dataset so we do not have to make submissions to Kaggle to get results
    x_train_cv, x_test, y_train_cv, y_test = train_test_split(t1cwe_data, t1cwe_data_labels, test_size=0.15,
                                                              random_state=42)

    print('Number of Negative Entries (Train and CV):', y_train_cv.tolist().count(0))
    print('Number of Positive Entries (Train and CV):', y_train_cv.tolist().count(1))

    print('Number of Negative Entries (Test):', y_test.tolist().count(0))
    print('Number of Positive Entries (Test):', y_test.tolist().count(1))

    print('Processing complete. Running Classifier...')

    # Define the search space
    space = {
        'con_out': hp.choice('con_out', [16, 32]),
        'con_kernel': hp.choice('con_kernel', [4, 8, 12]),
        'con_padding': 0,
        'linear_1_out': hp.choice('linear_1_out', [200, 300, 400]),
        'batch_size': 10,
        'x_train': x_train_cv,
        'y_train': y_train_cv,
    }

    # Keep track of the results
    bayes_trials = Trials()

    # Run optimization
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=18, trials=bayes_trials)

    # best = {'batch_size': 1, 'con_kernel': 0, 'con_out': 2, 'linear_1_out': 2}

    # Sort the trials with lowest loss first.
    bayes_trials_results = sorted(bayes_trials.results, key=lambda x: x['loss'])

    cv_history_list = bayes_trials_results[0]['history_list']
    plot_learning_curve_cv(cv_history_list)


if __name__ == '__main__':
    main()

# space = {
    #     'con_out': hp.choice('con_out', [16, 32]),
    #     'con_kernel': hp.choice('con_kernel', [4, 8, 12]),
    #     'con_padding': 0,
    #     'linear_1_out': hp.choice('linear_1_out', [200, 300, 400]),
    #    'batch_size': 10,
    #    'x_train': x_train_cv,
    #   'y_train': y_train_cv,
# }
# [3:26:48<00:00, 689.37s/trial, best loss: 0.4098038872083028]
# CNN Filter Size: 16 CNN Kernel Size: 8 CNN Padding: 0 Linear #1 Units: 300 TPE Loss: 0.4098038872083028

'''Number of Negative Entries (Train and CV): 234
Number of Positive Entries (Train and CV): 260
Number of Negative Entries (Test): 42
Number of Positive Entries (Test): 46'''
