import pandas as pd
import pickle
import matplotlib.pyplot as plt

class Utils():
    @staticmethod
    def plotAcc(x1, x2):
        acc1, name1 = x1
        acc2, name2 = x2
        epochs = list(range(len(acc1["bowel"])))

        series_names = ["bowel", "extra", "liver", "kidney", "spleen"]
        _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        for _, series_name in enumerate(series_names):
            axs[0].plot(epochs, acc1[series_name], label=f"Acc1 - {series_name}")
            axs[1].plot(epochs, acc2[series_name], linestyle='--', label=f"Acc2 - {series_name}")

        axs[0].set_ylabel(f'{name1}')
        axs[1].set_ylabel(f'{name2}')
        axs[1].set_xlabel('Epoch')

        axs[0].legend(loc='upper left')
        axs[1].legend(loc='upper left')

        plt.suptitle('Training Accuracy')
        plt.show()


    @staticmethod
    def printScores(scores):
        for k in scores.keys():
            print(k)
            print(f"Accuracy: {scores[k][0]}")
            print(f"F1: {scores[k][1]}")
            print(f"Precision: {scores[k][2]}")
            print(f"Recall: {scores[k][3]}")

    @staticmethod
    def Pickle(fileName: str, object: any) -> None:
        with open(f"{fileName}.pkl", "wb") as file:
            pickle.dump(object, file)

    @staticmethod
    def UnPickle(fileName: str) -> any:
        with open(f"{fileName}.pkl", 'rb') as file:
            object = pickle.load(file)
            return object
