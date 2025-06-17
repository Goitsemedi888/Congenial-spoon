import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import os
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
import pydicom
import dicomsdl
import tensorflow as tf




class Feeder():
    def __glob_sorted(self, path: str):
        return sorted(glob(path), key=lambda x: int(x.split('/')[-1].split('.')[0]))


    def __get_windowed_image(self, dcm, WL=50, WW=400):
        resI, resS = dcm.RescaleIntercept, dcm.RescaleSlope
        
        img = dcm.to_numpy_image()
        img = resS * img + resI
        
        upper, lower = WL+WW//2, WL-WW//2
        X = np.clip(img.copy(), lower, upper)
        X = X - np.min(X)
        X = X / np.max(X)
        X = (X*255.0).astype('uint8')
        
        return X

    def __load_volume(self, dcms):
        volume = []
        for dcm_path in dcms:
            dcm = dicomsdl.open(dcm_path)
            image = self.__get_windowed_image(dcm)
            
            if np.min(image)<0:
                image = image + np.abs(np.min(image))
            
            image = image / image.max()
            
            volume.append(image)
            
        return np.stack(volume)



    def __slice_first_dimension_to_length(self, array3d, target_length) -> Optional[np.array]:
        # Calculate the interval for slicing the first dimension
        interval = np.floor(array3d.shape[0] / target_length).astype(int)
        
        if(interval == 0):
            return None
        
        # Use the calculated interval to slice the first dimension of the array
        # while keeping the other dimensions unchanged
        sliced_array = array3d[::interval, :, :]

        # Trim any excess elements if necessary to match the target length exactly
        return sliced_array[:target_length, :, :]


    def make_3D_data(self, trainCSV: str, getFolder: str, saveFolder: str):
        study_level = pd.read_csv(trainCSV)
        study_level.drop(["any_injury"], axis=1, inplace=True)

        SAVE_FOLDER = saveFolder
        os.makedirs(SAVE_FOLDER, exist_ok=1)

        for _, row in tqdm(study_level.iterrows()):
            patient = row.patient_id
            classification = row[1:].to_numpy()

            studies = os.listdir(f"{getFolder}/{patient}")
            studies = [item for item in studies if "_" not in item]
            
            if(len(studies) > 1):
                continue

            study = os.listdir(f"{getFolder}{patient}")[0]
            dcms = self.__glob_sorted(f"{getFolder}{patient}/{study}/*.dcm")
                
            volume = self.__load_volume(dcms)
            volume = self.__slice_first_dimension_to_length(volume, 32)

            if(volume.all() == None):
                continue
            
            np.savez(f"{SAVE_FOLDER}/{patient}_{study}.npz", volume=volume, classification=classification)
            print("Done")


    def fixShapes(self, npzFiles: list, folder: str):
        #folder = "3D_Volumes"
        #npzFiles = os.listdir(folder)
        for i in range(0, len(npzFiles)):
            print(f"On iteration: {i}")
            try:
                data = np.load(f"{folder}/{npzFiles[i]}")
                img = data["volume"]
                if img.shape != (32, 512, 512):
                    os.remove(f"{folder}/{npzFiles[i]}")
                    print(f"Deleted file: {npzFiles[i]}")
            except:
                    os.remove(f"{folder}/{npzFiles[i]}")
                    print("Could not read file")


    def bagData(self, start: int, size: int, folder: str) -> Tuple[tf.Tensor, Dict]:
        #folder = "3D_Volumes"
        #npzFiles = os.listdir(folder)
        x_train = []
        y_train = []
        y_names: list = ["bowel", "extra", "liver", "kidney", "spleen"]

        files = os.listdir(folder)

        for i in tqdm(range(start, size)):
            data = np.load(f"{folder}/{files[i]}")
            img = data["volume"]
            classification = data["classification"]

            y_train_dict = {
                y_names[0]: tf.convert_to_tensor(classification[0:2].reshape(1, -1), dtype=tf.float32),
                y_names[1]: tf.convert_to_tensor(classification[2:4].reshape(1, -1), dtype=tf.float32),
                y_names[2]: tf.convert_to_tensor(classification[4:7].reshape(1, -1), dtype=tf.float32),
                y_names[3]: tf.convert_to_tensor(classification[7:10].reshape(1, -1), dtype=tf.float32),
                y_names[4]: tf.convert_to_tensor(classification[10:13].reshape(1, -1), dtype=tf.float32)
                }

            x_train.append(tf.convert_to_tensor(img.reshape(1, 32, 512, 512)))
            y_train.append(y_train_dict)

        return (x_train, y_train)

    def iterData(self, start: int, batchSize: int, folder: str, files: list, y_names: list) -> Optional[Tuple[tf.Tensor, Dict]]:
        for i in (range(start, batchSize)):
            data = np.load(f"{folder}/{files[i]}")
            img = data["volume"]
            classification = data["classification"]

            y_train = {
                y_names[0]: tf.convert_to_tensor(classification[0:2].reshape(1, -1), dtype=tf.float32),
                y_names[1]: tf.convert_to_tensor(classification[2:4].reshape(1, -1), dtype=tf.float32),
                y_names[2]: tf.convert_to_tensor(classification[4:7].reshape(1, -1), dtype=tf.float32),
                y_names[3]: tf.convert_to_tensor(classification[7:10].reshape(1, -1), dtype=tf.float32),
                y_names[4]: tf.convert_to_tensor(classification[10:13].reshape(1, -1), dtype=tf.float32)
                }

            x_train = tf.convert_to_tensor(img.reshape(1, 32, 512, 512), dtype=tf.float32)
            yield x_train, y_train

    def showImage(self, file: str) -> None:
        ds = pydicom.dcmread(file)
        plt.imshow(ds.pixel_array, cmap = plt.cm.bone)
        plt.show()
