from Feeder import WalkFeeder
from CNN import CNN
import pandas as pd
import os
import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Optional
from tqdm import tqdm

folder = "KAGGLE_SUBMISSION/segmentation_data_v1"
npzFiles = os.listdir(folder)
y_names: list = ["bowel", "extra", "liver", "kidney", "spleen"]

def bagData(start: int, size: int, folder: str, files: list, y_names: list) -> Tuple[tf.Tensor, Dict]:
    #x_train = []
    #y_train = []

    for i in tqdm(range(start, size)):
        data = np.load(f"{folder}/{files[i]}")
        img = data["volume"]
        classification = data["classification"]
        print(classification)

        y_train = {
            y_names[0]: tf.convert_to_tensor(classification[0:2].reshape(1, -1), dtype=tf.float32),
            y_names[1]: tf.convert_to_tensor(classification[2:4].reshape(1, -1), dtype=tf.float32),
            y_names[2]: tf.convert_to_tensor(classification[4:7].reshape(1, -1), dtype=tf.float32),
            y_names[3]: tf.convert_to_tensor(classification[7:10].reshape(1, -1), dtype=tf.float32),
            y_names[4]: tf.convert_to_tensor(classification[10:13].reshape(1, -1), dtype=tf.float32)
            }

    x_train = tf.convert_to_tensor(img.reshape(1, 32, 512, 512), dtype=tf.float32)

    return (x_train, y_train)


x_test, y_test = bagData(1415, 1416, folder, npzFiles, y_names)

cnn = CNN(512, 512, 32, 9)
cnn.build()
cnn.compile()
cnn.loadWeights("testingWeights.ckpt")
y_pred = cnn.predictT(x_test)
print(y_pred)
print(y_test)