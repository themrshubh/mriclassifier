import numpy as np
import pydicom as dicom
import os
from matplotlib import pyplot
import pandas as pd
import math
import scipy.ndimage
import cv2


# Purpose: Walk the directory of data and read in the image data for each patient identification number found.
# Note: 3 patient ID's are flagged as erroneous in Kaggle: '00109', '00123', '00709'
# image_type can be: FLAIR, T1w, T1wCE, T2w
def retrieve_patient_data(image_type, num_images, image_length, image_width, resample=True):
    file_path = './train'
    samples_with_issues = ['00109', '00123', '00709']
    num_samples = 582

    list_of_patient_ids = []
    for directory, sub_directory, file_list in os.walk(file_path):
        if directory == file_path:
            list_of_patient_ids = sub_directory
    dimensions = (num_samples, num_images, image_length, image_width)
    all_data = np.zeros(dimensions)
    counter = 0
    for identification in list_of_patient_ids:
        if identification not in samples_with_issues and counter < num_samples + 1:
            patient_data = read_patient(identification, image_type, num_images, image_length, image_width, resample)
            all_data[counter, :, :] = patient_data
            counter += 1

    return all_data


# Purpose: Given a patient identification, retrieve the specified image data for the individual patient.
# image_type can be: FLAIR, T1w, T1wCE, T2w
def read_patient(identification, image_type, num_images, image_length, image_width, resample=True):
    file_path = './train/' + identification + '/' + image_type
    patient_files_list = []
    for directory, sub_directory, file_list in os.walk(file_path):
        for filename in file_list:
            if '.dcm' in filename.lower():
                patient_files_list.append(os.path.join(directory, filename))

    # get ref file
    ref_ds = dicom.read_file(patient_files_list[0])

    # load dimensions based on the number of rows, columns, and slices (along the Z axis)
    dimensions = (len(patient_files_list), int(ref_ds.Rows), int(ref_ds.Columns))

    # Load spacing values (in mm)
    spacing = (float(ref_ds.SliceThickness), float(ref_ds.PixelSpacing[0]), float(ref_ds.PixelSpacing[1]))

    dicom_array = np.zeros(dimensions, dtype=ref_ds.pixel_array.dtype)

    # loop through all the DICOM files
    for filename in patient_files_list:
        # read the file
        ds = dicom.read_file(filename)
        # normalize and store the raw image data
        dicom_array[patient_files_list.index(filename), :, :] = ds.pixel_array / 255.0
    if resample:
        new_array = resample_images(dicom_array, spacing)  # (use this line to include resampling of data)
        final_array = resize_images(new_array, num_images, image_length, image_width)
    else:
        final_array = resize_images(dicom_array, num_images, image_length, image_width)
    return np.array(final_array)


# Purpose: The MRI images are taken at varying pixel spacing and slice depth. This standardizes the spacing across all
# images.
def resample_images(images, spacing, new_spacing=[1, 1, 1]):
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = images.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / images.shape
    images = scipy.ndimage.interpolation.zoom(images, real_resize_factor)
    return images


# Purpose: Both before and after resampling, each patient can have a varying number of images and the images may come
# in varying pixel resolution. This resizes the samples so each the data for each patient is of a consistent size prior
# to input into the neural network.
def resize_images(images, num_images, image_length, image_width):
    dimensions = (num_images, image_length, image_width)
    new_images = np.zeros(dimensions)
    length = images.shape[0]
    start = math.ceil((length - num_images)/2)
    images = images[start:start+num_images, :, :]
    for index in range(0, images.shape[0]):
        image = images[index, :, :]
        image = cv2.resize(image, dsize=(image_length, image_width))
        new_images[index, :, :] = image
    return new_images


# Purpose: Read in the categorical labels for the training data
def retrieve_train_labels():
    labels_df = pd.read_csv('train_labels.csv')
    samples_with_issues = [109, 123, 709]
    for sample in samples_with_issues:
        labels_df = labels_df[labels_df.BraTS21ID != sample]
    # labels_df = labels_df[labels_df.BraTS21ID.isin(samples_with_issues)]
    labels = labels_df['MGMT_value'].values
    return labels


# Purpose: Given an array of images, plot 25 of the images in a 5x5 arrangement.
def plot_images_together_range(dicom_array, rows=5, cols=5, start_with=4, show_every=1):
    fig, ax = pyplot.subplots(rows, cols, figsize=[10, 10])
    for i in range(rows * cols):
        ind = start_with + i * show_every
        ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
        ax[int(i / rows), int(i % rows)].imshow(dicom_array[ind, :, :], cmap='gray')
        ax[int(i / rows), int(i % rows)].axis('off')
    pyplot.show()
