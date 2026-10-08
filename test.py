import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import kagglehub
import random


# Load dataset from Kaggle
print(tf.version.VERSION)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# image_dir = path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
# print(image_dir)

# # Display sample images
# def display_samples(num_samples=10):
#     fig, ax = plt.subplots(num_samples, 2, figsize=(10, 20))
#     image_dir = './lgg-mri-segmentation/kaggle_3m'
    
#     for i, patient_dir in enumerate(os.listdir(image_dir)[:num_samples]):
#         print(f"{i} ------- {patient_dir}")
#         if not patient_dir.startswith('TCGA'): continue
#         img_path = os.path.join(image_dir, patient_dir, f'{patient_dir}_1.tif')
#         # print(img_path)
#         mask_path = os.path.join(image_dir, patient_dir, f'{patient_dir}_1_mask.tif')
        
#         img = cv2.imread(img_path)
#         mask = cv2.imread(mask_path, 0)
        
#         ax[i,0].imshow(img)
#         ax[i,0].set_title('MRI Image')
#         ax[i,1].imshow(mask)
#         ax[i,1].set_title('Mask')
#     plt.show()

# display_samples()


import random
def show_predictions(num_samples=20):
    fig, ax = plt.subplots(num_samples, 3, figsize=(15, 20))
    
    for i in range(num_samples):
        ii = random.randrange(0, 200)
        ax[i,0].imshow(X_test[ii])
        ax[i,0].set_title('Input Image')
        ax[i,1].imshow(y_test[ii].squeeze(), cmap='gray')
        ax[i,1].set_title('Ground Truth')
        pred = model.predict(tf.expand_dims(X_test[ii], axis=0))[0]
        ax[i,2].imshow(pred.squeeze(), cmap='gray')
        ax[i,2].set_title('Prediction')
    plt.show()

show_predictions()