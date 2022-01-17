## Group Project

**Course**: CSC 522 (ALDA), North Carolina State University\
**Instructor**: Dr. Min Chi\
**Semester**: Fall 2021\
**Assignment**: Group Project\
**Author**: Project Group #8\
**Members**: Jonathan Wood, Oliver Fowler, Shubham Mankar, and Jesse Wood

### Files and Description

| File | Description                                                                                                       |
|----------|-----------------------------------------------------------------------------------------------------------------------|
|[mri_data_functions.py](https://github.ncsu.edu/jwood9/CSC_522_Project/blob/main/mri_data_functions.py)| Function definitions required for reading MRI data from file. |
|[cnn_model_functions.py](https://github.ncsu.edu/jwood9/CSC_522_Project/blob/main/cnn_model_functions.py)| Definition of the CNN model used for image classification. |
|[cnn_model_tuning.py](https://github.ncsu.edu/jwood9/CSC_522_Project/blob/main/cnn_model_tuning.py)| Code to perform bayesian optimization of the CNN model using 3-fold cross-validation. |
|[run_cnn_model_tuning.sh](https://github.ncsu.edu/jwood9/CSC_522_Project/blob/main/run_cnn_model_tuning.sh)| Runs the CNN model tuning with each type of MRI image as input. |
|[cnn_model_evaluation.py](https://github.ncsu.edu/jwood9/CSC_522_Project/blob/main/cnn_model_evaluation.py)| Code to evaluate the CNN model on the test dataset. |
|[run_cnn_model_evaluation.sh](https://github.ncsu.edu/jwood9/CSC_522_Project/blob/main/run_cnn_model_evaluation.sh)| Runs the CNN model evaluation with each type of MRI image as input. |
|[cnn_lstm_model_functions.py](https://github.ncsu.edu/jwood9/CSC_522_Project/blob/main/cnn_lstm_model_functions.py)| Definition of the CNN-LSTM model used for image classification. |
|[cnn_lstm_model_tuning.py](https://github.ncsu.edu/jwood9/CSC_522_Project/blob/main/cnn_model_tuning.py)| Code to perform bayesian optimization of the CNN-LSTM model using 3-fold cross-validation. |
|[run_cnn_model_tuning.sh](https://github.ncsu.edu/jwood9/CSC_522_Project/blob/main/run_cnn_model_tuning.sh)| Runs the CNN-LSTM model tuning with each type of MRI image as input. |
|[cnn_model_evaluation.py](https://github.ncsu.edu/jwood9/CSC_522_Project/blob/main/cnn_model_evaluation.py)| Code to evaluate the CNN-LSTM model on the test dataset. |
|[run_cnn_model_evaluation.sh](https://github.ncsu.edu/jwood9/CSC_522_Project/blob/main/run_cnn_model_evaluation.sh)| Runs the CNN-LSTM model evaluation with each type of MRI image as input. |
|[svm_model_tuning_evaluation.py](https://github.ncsu.edu/jwood9/CSC_522_Project/blob/main/svm_model_tuning_evaluation.py)| Code to tune and evaluate the SVM model on the test dataset. |
|[run_svm_model_tuning_evaluation.sh](https://github.ncsu.edu/jwood9/CSC_522_Project/blob/main/run_svm_model_tuning_evaluation.sh)| Runs the SVM model tuning and evaluation with each type of MRI image as input. |

### Programming Language and Libraries
The project was built using Python 3. The following Python Libraries were used in implementation:

hyperopt        0.2.5  
matplotlib      3.4.2  
numpy           1.19.5  
pandas          1.3.3  
pydicom         2.2.2  
scikit-learn    0.24.2  
scipy           1.6.2  
torch           1.9.1  
torchmetrics    0.5.1  
    
### Execution Details

Step 1: Clone the CSC_522_Project repository
<pre><code>git clone https://github.ncsu.edu/themrshubh/mriclassifier.git</code></pre>

Step 2: Change the working directory to 'mriclassifier'
<pre><code>cd mriclassifier</code></pre>

Step 3: Download the RSNA-MICCAI Brain Tumor Radiogenomic Classification dataset and extract it to the working directory. Note the dataset is very large and is available [here](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification).

Step 4: Run the script for tuning or evaluating one of the models.  

CNN Model: <pre><code>./run_cnn_model_tuning.sh  
./run_cnn_model_evaluation.sh</code></pre>

CNN-LSTM Model: <pre><code>./run_cnn_lstm_model_tuning.sh  
./run_cnn_lstm_model_evaluation.sh</code></pre>


SVM Model: <pre><code>./run_svm_model_tuning_evaluation.sh </code></pre>

Verbose results will be displayed in the console.  

Approximate Run Time for Models:  
SVM: 4 hours (tuning & evaluation using 64x64 image) on M1 Macbook Pro  
CNN: 16 hours (tuning), 2.5 hours (evaluation) on NVIDIA RTX 3060 GPU  
CNN-LSTM: 18 hours (tuning), 3 hours (evaluation) on NVIDIA RTX 3060 GPU  
ResNet:  
