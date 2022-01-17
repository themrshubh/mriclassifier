# CSC_522_Project

## The following files are included:

**File**: mri_classification_functions_simpler.py

**Description**: Function definitions for reading and pre-processing image data, defining the neural network architecture and operations, outputting evaluation metrics, and plotting learning curves.

**File**: flair_image_classifier_simpler.py

**Description**: Reads the FLAIR images, trains the model using the Training and CV Cohort, and evaluates the model on the Test Cohort.

**File**: t1w_image_classifier_simpler.py

**Description**: Reads the T1w images, trains the model using the Training and CV Cohort, and evaluates the model on the Test Cohort.

**File**: t1wce_image_classifier_simpler.py

**Description**: Reads the T1wCE images, trains the model using the Training and CV Cohort, and evaluates the model on the Test Cohort.

**File**: t2w_image_classifier_simpler.py

**Description**: Reads the T2 images, trains the model using the Training and CV Cohort, and evaluates the model on the Test Cohort.

**File**: t1wce_image_classifier_tuning_simpler.py

**Description**: Runs bayesian optimization over specified ranges for the hyperparameters of the model (with T1wCE images as input) using 3-fold cross-validation.
