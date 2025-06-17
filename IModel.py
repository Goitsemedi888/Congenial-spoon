import abc
from typing import List, Any, Optional, Generator
import tensorflow as tf


class IModel(metaclass=abc.ABCMeta):

    def __init__(self):
        self.model: Optional[tf.keras.Model] = None


    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass), "build" and callable(subclass.buildModel)
            and
            hasattr(subclass), "compile" and callable(subclass.compile)
            and
            hasattr(subclass), "fit" and callable(subclass.fit)
            and
            hasattr(subclass), "evaluate" and callable(subclass.evaluate)
            and
            hasattr(subclass), "predict" and callable(subclass.predict)
            and
            hasattr(subclass), "saveWeight" and callable(subclass.weights)
            and
            hasattr(subclass), "loadWeights" and callable(subclass.loadWeights)
            and
            hasattr(subclass), "saveModel" and callable(subclass.saveModel)
            and
            hasattr(subclass), "loadModel" and callable(subclass.loadModel)
            and
            hasattr(subclass), "printSummary" and callable(subclass.printSummary)
            and
            hasattr(subclass), "exportSummary" and callable(subclass.exportSummary)
            and
            hasattr(subclass), "fitGenerator" and callable(subclass.fitGenerator)
            )

    @abc.abstractclassmethod
    def setCallbacks(self,  callBacks: List[tf.keras.callbacks.Callback]) -> None:
        raise NotImplementedError

    @abc.abstractclassmethod
    def build(self) -> None:
        raise NotImplementedError

    @abc.abstractclassmethod
    def compile(self, loss: str, optimizer: str, metrics: List[str]) -> None:
        raise NotImplementedError

    @abc.abstractclassmethod
    def fit(self, x: tf.Tensor, y: tf.Tensor, batchSize: int, epochs: int) -> Any:
        raise NotImplementedError

    @abc.abstractclassmethod
    def fitGenerator(self, batchSize: int, batches: int, epochs: int, generator: Generator) -> Any:
        raise NotImplementedError

    @abc.abstractclassmethod
    def fitGenerator(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def evaluateT(self) -> None:
        raise NotImplementedError

    @abc.abstractclassmethod
    def predictT(self) -> None:
        raise NotImplementedError

    @abc.abstractclassmethod
    def saveWeights(self, fileName: str) -> None:
        raise NotImplementedError

    @abc.abstractclassmethod
    def loadWeights(self, fileName: str) -> None:
        raise NotImplementedError

    @abc.abstractclassmethod
    def saveModel(self, fileName: str) -> None:
        raise NotImplementedError

    @abc.abstractclassmethod
    def loadModel(self, fileName: str) -> None:
        raise NotImplementedError

    @abc.abstractclassmethod
    def printSummary(self) -> None:
        raise NotImplementedError

    @abc.abstractclassmethod
    def exportSummary(self, fileName: str) -> None:
        raise NotImplementedError
