# Databricks notebook source
import zipfile
import pandas as pd
import numpy as np
import os
import glob
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pyplot as plt

# COMMAND ----------

storage_account_name = "gpuworkspace0501694496"
storage_account_key = dbutils.secrets.get("Azure", "StorageKey")

# COMMAND ----------



dbutils.fs.mount(
  source = f"wasbs://data@{storage_account_name}.blob.core.windows.net",
  mount_point = "/mnt/data",
  extra_configs = {f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_key}
)

# COMMAND ----------

mounts = dbutils.fs.mounts()

mounts

# COMMAND ----------

with zipfile.ZipFile("/dbfs/mnt/data/rsna-miccai-brain-tumor-radiogenomic-classification.zip") as zip_file:
  zip_file.extractall("/dbfs/mnt/data/image-data")

# COMMAND ----------

# MAGIC %fs rm -r dbfs:/mnt/data/image-data

# COMMAND ----------

# MAGIC %fs ls "dbfs:/mnt/data/image-data"

# COMMAND ----------

labels = pd.read_csv("/dbfs/mnt/data/image-data/train_labels.csv")
labels.head()

# COMMAND ----------

labels.shape

# COMMAND ----------

# MAGIC %fs ls "dbfs:/mnt/data/image-data/train"

# COMMAND ----------

files = dbutils.fs.ls("dbfs:/mnt/data/image-data/train")

# COMMAND ----------

image_types = ["FLAIR", "T1w", "T1wCE", "T2w"]

# COMMAND ----------

dcm_files = dbutils.fs.ls(f"{files[0].path}/{image_types[0]}")

# COMMAND ----------

dcm_files

# COMMAND ----------

new_path = dcm_files[0].path.replace("dbfs:/", "/dbfs/")
ds = read_file(new_path)

# COMMAND ----------

# Thanks to help from this notebook on how to read in the files - https://www.kaggle.com/ihelon/brain-tumor-eda-with-animations-and-modeling

def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def visualize_sample(
    brats21id, 
    slice_i,
    mgmt_value,
    types=("FLAIR", "T1w", "T1wCE", "T2w")
):
    plt.figure(figsize=(16, 5))
    patient_path = os.path.join(
        "/dbfs/mnt/data/image-data/train", 
        str(brats21id).zfill(5),
    )
        
    for i, t in enumerate(types, 1):
        new_path = sorted(
            glob.glob(os.path.join(patient_path, t, "*")), 
            key=lambda x: int(x[:-4].split("-")[-1]),
        )
            
        data = load_dicom(new_path[int(len(new_path) * slice_i)])
        plt.subplot(1, 4, i)
        plt.imshow(data, cmap="gray")
        plt.title(f"{t}", fontsize=16)
        plt.axis("off")

    plt.suptitle(f"MGMT_value: {mgmt_value}", fontsize=16)
    plt.show()
