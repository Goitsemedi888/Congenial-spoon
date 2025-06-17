import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np

def load_dicom_images(folder_path, num_images=6, brightness_factor=0.7):
    dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm') and not ("_") in f]
    dicom_images = []

    
    for i in range(0, len(dicom_files), 40):
        dicom_data = pydicom.dcmread(dicom_files[i])
        image = dicom_data.pixel_array
                # Normalize to 0-255 if not already
        if image.max() > 255:
            image = (image / image.max()) * 255

        # Adjust brightness
        adjusted_image = np.clip(image * brightness_factor, 0, 255)

        dicom_images.append(adjusted_image)

    return dicom_images

def plot_dicom_images(dicom_images):
    fig, axs = plt.subplots(2, 3, figsize=(30, 25))

    plt.subplots_adjust(wspace=0.0, hspace=0.2)

    for i, ax in enumerate(axs.flat):
        if i < len(dicom_images):
            ax.imshow(dicom_images[i], cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')  # Turn off axis for empty subplots

    plt.show()

def showImage(file: str) -> None:
    ds = pydicom.dcmread(file)
    plt.imshow(ds.pixel_array, cmap = plt.cm.bone)
    plt.show()


# Replace 'your_dicom_folder_path' with the path to your DICOM images folder
#dicom_folder_path = '/Volumes/T7 Shield/Project/images/41050/21228'
#dicom_images = load_dicom_images(dicom_folder_path)
#plot_dicom_images(dicom_images)

#showImage("/Volumes/T7 Shield/Project/images/18811/25133/1.dcm")

from Feeder import Feeder
from CNN import CNN
from TransformerModel import Transformer
import os
import tensorflow as tf

#cnn = CNN()
#cnn.build(512, 512, 32)
transformer = Transformer()
transformer.build(32, 512, 512)
tf.keras.utils.plot_model(transformer.model, to_file='ViT_Arch.png', show_shapes=True)

