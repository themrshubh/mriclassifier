from mri_data_functions import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import svm
import time
import argparse


def tune_svm_four_kernels(reg_param_list, degree_list, coeff_list, gamma_list, x_train, y_train, x_test, y_test):
    st = time.time()
    # tune with linear kernel
    param_grid_linear = {'C': reg_param_list,
                         'kernel': ['linear']}
    linear_grid = GridSearchCV(svm.SVC(), param_grid_linear, verbose=0)
    linear_grid.fit(x_train, y_train)
    best_linear_params = linear_grid.best_params_
    linear_grid_pred = linear_grid.predict(x_test)
    linear_accuracy = accuracy_score(y_test, linear_grid_pred)
    linear_precision = precision_score(y_test, linear_grid_pred)
    linear_recall = recall_score(y_test, linear_grid_pred)
    linear_f1 = f1_score(y_test, linear_grid_pred)

    en = time.time()
    print(f'Linear Kernel = {en - st}s')
    st = en

    # tune with polynomial kernel
    param_grid_poly = {'C': reg_param_list,
                       'degree': degree_list,
                       'coef0': coeff_list,
                       'kernel': ['poly']}
    poly_grid = GridSearchCV(svm.SVC(), param_grid_poly, verbose=0)
    poly_grid.fit(x_train, y_train)
    best_poly_params = poly_grid.best_params_
    poly_grid_pred = poly_grid.predict(x_test)
    poly_accuracy = accuracy_score(y_test, poly_grid_pred)
    poly_precision = precision_score(y_test, poly_grid_pred)
    poly_recall = recall_score(y_test, poly_grid_pred)
    poly_f1 = f1_score(y_test, poly_grid_pred)

    en = time.time()
    print(f'Poly Kernel = {en - st}s')
    st = en

    # tune with RBF kernel
    param_grid_rbf = {'C': reg_param_list,
                      'gamma': gamma_list,
                      'kernel': ['rbf']}
    rbf_grid = GridSearchCV(svm.SVC(), param_grid_rbf, verbose=0)
    rbf_grid.fit(x_train, y_train)
    best_rbf_params = rbf_grid.best_params_
    rbf_grid_pred = rbf_grid.predict(x_test)
    rbf_accuracy = accuracy_score(y_test, rbf_grid_pred)
    rbf_precision = precision_score(y_test, rbf_grid_pred)
    rbf_recall = recall_score(y_test, rbf_grid_pred)
    rbf_f1 = f1_score(y_test, rbf_grid_pred)

    en = time.time()
    print(f'RBF Kernel = {en - st}s')
    st = en

    # tune with Sigmoid kernel
    param_grid_sigmoid = {'C': reg_param_list,
                          'coef0': coeff_list,
                          'gamma': gamma_list,
                          'kernel': ['sigmoid']}
    sigmoid_grid = GridSearchCV(svm.SVC(), param_grid_sigmoid, verbose=0)
    sigmoid_grid.fit(x_train, y_train)
    best_sigmoid_params = sigmoid_grid.best_params_
    sigmoid_grid_pred = sigmoid_grid.predict(x_test)
    sigmoid_accuracy = accuracy_score(y_test, sigmoid_grid_pred)
    sigmoid_precision = precision_score(y_test, sigmoid_grid_pred)
    sigmoid_recall = recall_score(y_test, sigmoid_grid_pred)
    sigmoid_f1 = f1_score(y_test, sigmoid_grid_pred)

    en = time.time()
    print(f'Sigmoid Kernel = {en - st}s')
    st = en

    # output results
    print('\n')
    print('\u0332'.join('Kernel Type') + '\t' + '\u0332'.join('Best Parameters'))
    print('Linear\t\tC: ' + str(best_linear_params['C']))
    print('Polynomial\tC: ' + str(best_poly_params['C']) + ', degree: ' + str(best_poly_params['degree']) +
          ', coef0: ' + str(best_poly_params['coef0']))
    print('RBF\t\tC: ' + str(best_rbf_params['C']) + ', \u03b3: ' + str(best_rbf_params['gamma']))
    print('Sigmoid\t\tC: ' + str(best_sigmoid_params['C']) + ', coef0: ' + str(best_sigmoid_params['coef0']) +
          ', \u03b3: ' + str(best_sigmoid_params['gamma']))

    print('\n')
    print('\u0332'.join('Kernel Type') + '\t' + '\u0332'.join('Accuracy') + '\t' + '\u0332'.join('Precision') + '\t' +
          '\u0332'.join('Recall') + '\t\t' + '\u0332'.join('F1 Score'))
    print('Linear\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f} '.format(linear_accuracy, linear_precision, linear_recall,
                                                                   linear_f1))
    print('Polynomial\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f} '.format(poly_accuracy, poly_precision, poly_recall,
                                                                     poly_f1))
    print('RBF\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f} '.format(rbf_accuracy, rbf_precision, rbf_recall, rbf_f1))
    print('Sigmoid\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f} '.format(sigmoid_accuracy, sigmoid_precision,
                                                                    sigmoid_recall, sigmoid_f1))

def combine_images(images):
    average = images.mean(axis=0)
    return average

def main():
    st = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-datasetName', default='FLAIR', type=str)
    args = parser.parse_args()
    dataset_name = args.datasetName
    print(dataset_name)
    
    num_images = 60
    image_length = 128
    image_width = 128

    raw_data = retrieve_patient_data(dataset_name, num_images, image_length, image_width, resample=False)  # image_type can be: FLAIR, T1w, T1wCE, T2w
    average_data = combine_images(raw_data)
    data = [[] for _ in range(len(average_data))]
    data_labels = retrieve_train_labels()

    for i in range(len(average_data)):
        data[i] = average_data[i].flatten()
        

    x_train_cv, x_test, y_train_cv, y_test = train_test_split(data, data_labels, test_size=0.15, random_state=42)

    en = time.time()
    print(f'Data preprocessing = {en - st}s')

    reg_param_list = [0.1, 0.2, 0.3, 1, 5, 10, 20, 100, 200, 1000]
    degree_list = [1, 2, 3, 4, 5]
    coeff_list = [0.0001, 0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 0.3, 1, 2, 5, 10]
    gamma_list = [0.0001, 0.001, 0.002, 0.01, 0.02, 0.03, 0.1, 0.2, 1, 2, 3]

    tune_svm_four_kernels(reg_param_list, degree_list, coeff_list, gamma_list, x_train_cv, y_train_cv, x_test, y_test)

main()