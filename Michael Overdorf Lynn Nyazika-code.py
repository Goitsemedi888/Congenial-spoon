import os
from pathlib import Path
import pandas as pd
from Feeder import Feeder
from CNN import CNN
from TransformerModel import Transformer
from Utils import Utils

# Setup generator function to grab train data from folder
feeder = Feeder()
gen = feeder.iterData

# Set to true if you want to retrain the models, false to load pre-trained weights
TRAIN = True

# Set folders where train and test sets are and get the number of samples in each folder
TRAINFOLDER = Path("../3D_Volumes/train")
TESTFOLDER = Path("../3D_Volumes/test")
TRAINSIZE = len(os.listdir(TRAINFOLDER))
TESTSIZE = len(os.listdir(TESTFOLDER))


# Instantiate CNN and ViT models.  Both take same input, and train on same data
cnn = CNN()
cnn.build(32, 512, 512)
cnn.compile()

vit = Transformer()
vit.build(32, 512, 512)
vit.compile()

if TRAIN:
    # If TRAIN start training both models
    print("###############################################")
    print("Training CNN Model")
    print("###############################################")

    lossC, accC = cnn.fitGenerator(0, TRAINSIZE, 1, TRAINFOLDER, gen)
    cnn.saveWeights("cnnWeights")

    print("###############################################")
    print("Training ViT Model")
    print("###############################################")

    lossV, accV = vit.fitGenerator(0, TRAINSIZE, 1, TRAINFOLDER, gen)
    vit.saveWeights("vitWeights")
else:
    # Else load pre-trained weights
    cnn.loadWeights("cnnWeights")
    vit.loadWeights("vitWeights")
    lossC, accC = Utils.UnPickle("cnnTrainingData")
    lossV, accV = Utils.UnPickle("vitTrainingData")

# Bag the test data into x_test and y_test sets
x_test, y_test = feeder.bagData(0, TESTSIZE, TESTFOLDER)

# Calculate Acc, F1, Precision and Recall scores for each model and plot training accuracy data
print("###############################################")
print("Evaluating CNN Model")
print("###############################################")
scoresC = cnn.evaluateT(x_test, y_test)
Utils.printScores(scoresC)

print("###############################################")
print("Evaluating ViT Model")
print("###############################################")
scoresV = vit.evaluateT(x_test, y_test)
Utils.printScores(scoresV)

Utils.plotAcc((accC, "CNN Accuracy"), (accV, "ViT Accuracy"))
