import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Generator, Tuple, Dict 
from IModel import IModel


class HyperParameters():
    def __init__(self):
        self.learning_rate: float = 0.001
        self.weight_decay: float = 0.0001
        self.batch_size: int = 1
        self.num_epochs: int = 100
        self.image_size: int = 72
        self.patch_size: int = 6
        self.num_patches: int = (self.image_size // self.patch_size) ** 2
        self.projection_dim: int = 64
        self.num_heads: int = 4
        self.transformer_units: List[int] = [self.projection_dim * 2, self.projection_dim]
        self.transformer_layers: int = 8
        self.mlp_head_units = [2048, 1024]


class Transformer(IModel):

    class __Patches(tf.keras.layers.Layer):
        def __init__(self, patch_size):
            super().__init__()
            self.patch_size = patch_size

        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches

    class __PatchEncoder(tf.keras.layers.Layer):
        def __init__(self, num_patches: int, projection_dim: int) -> None:
            super().__init__()
            self.num_patches = num_patches
            self.projection = tf.keras.layers.Dense(units=projection_dim)
            self.position_embedding = tf.keras.layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )

        def call(self, patch):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            encoded = self.projection(patch) + self.position_embedding(positions)
            return encoded

    def __init__(self, hyperParameters: HyperParameters = HyperParameters()) -> None:
        self.hyperParameters: HyperParameters = hyperParameters

    def __mlp(self, x: tf.keras.layers.Layer, hiddenUnits: List[int], dropoutRate: float) -> tf.keras.layers.Layer:
        for units in hiddenUnits:
            x = x
            x = tf.keras.layers.Dense(units, activation="gelu")(x)
            x = tf.keras.layers.Dropout(dropoutRate)(x)
        return x

    class __dataAugmentation():
        @staticmethod
        def getAugmentation(imgWidth: int, imgHeight: int, randomRotationFactor: float, randomHeightFactor: float, randomWidthFactor: float) -> None:
            augmentation = tf.keras.Sequential(
                [
                    tf.keras.layers.Resizing(imgWidth, imgHeight),
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomRotation(factor=randomRotationFactor),
                    tf.keras.layers.RandomZoom(randomHeightFactor, randomWidthFactor)
                ])
            return augmentation

    def setCallbacks(self, checkpoint_filepath = "/tmp/checkpoint") -> None:
        self.checkpoint_filepath = checkpoint_filepath
        self.callbacks = [keras.callbacks.ModelCheckpoint(
            self.checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True)]

    def loadFromCheckpoint(self, x_test: tf.Tensor, y_test: tf.Tensor) -> None:
        self.model.load_weights(self.checkpoint_filepath)
        _, accuracy, top_5_accuracy = self.model.evaluate(x_test, y_test)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    def build(self, imageDepth, imageWidth, imageHeight) -> None:
        inputs = tf.keras.Input(shape=(imageDepth, imageWidth, imageHeight, ))
        augmented = self.__dataAugmentation.getAugmentation(self.hyperParameters.image_size, self.hyperParameters.image_size, 0.2, 0.2, 0.2)(inputs)
        patches = self.__Patches(patch_size=self.hyperParameters.patch_size)(augmented)
        encodedPatches = self.__PatchEncoder(num_patches=self.hyperParameters.num_patches, projection_dim=self.hyperParameters.projection_dim)(patches)

        for _ in range(0, self.hyperParameters.transformer_layers):
            x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encodedPatches)
            attentionOutput = tf.keras.layers.MultiHeadAttention(num_heads=self.hyperParameters.num_heads, key_dim=self.hyperParameters.projection_dim, dropout=0.1)(x1, x1)
            x2 = tf.keras.layers.Add()([attentionOutput, encodedPatches])
            x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = self.__mlp(x3, hiddenUnits=self.hyperParameters.transformer_units, dropoutRate=0.1)
            encodedPatches = tf.keras.layers.Add()([x3, x2])

        representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encodedPatches)
        representation = tf.keras.layers.Flatten()(representation)
        representation = tf.keras.layers.Dropout(0.5)(representation)


        features = self.__mlp(representation, hiddenUnits=self.hyperParameters.mlp_head_units, dropoutRate=0.5)

        x_bowel = tf.keras.layers.Dense(32, activation='silu')(features)
        x_extra = tf.keras.layers.Dense(32, activation='silu')(features)
        x_liver = tf.keras.layers.Dense(32, activation='silu')(features)
        x_kidney = tf.keras.layers.Dense(32, activation='silu')(features)
        x_spleen = tf.keras.layers.Dense(32, activation='silu')(features)

        bowel = tf.keras.layers.Dense(2, name='bowel', activation='softmax')(x_bowel) # use sigmoid to convert predictions to [0-1]
        extra = tf.keras.layers.Dense(2, name='extra', activation='softmax')(x_extra) # use sigmoid to convert predictions to [0-1]
        liver = tf.keras.layers.Dense(3, name='liver', activation='softmax')(x_liver) # use softmax for the liver head
        kidney = tf.keras.layers.Dense(3, name='kidney', activation='softmax')(x_kidney) # use softmax for the kidney head
        spleen = tf.keras.layers.Dense(3, name='spleen', activation='softmax')(x_spleen) # use softmax for the spleen head

        outputs = [bowel, extra, liver, kidney, spleen]

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model = model

    def compile(self) -> None:
        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01),
                           loss={"bowel": "categorical_crossentropy", "extra": "categorical_crossentropy", "liver": "categorical_crossentropy", "kidney": "categorical_crossentropy", "spleen": "categorical_crossentropy"},
                           metrics={"bowel": "categorical_accuracy", "extra": "categorical_accuracy", "liver": "categorical_accuracy", "kidney": "categorical_accuracy", "spleen": "categorical_accuracy"})
            

    def fit(self, x: tf.Tensor, y: tf.Tensor) -> None:
        history = self.model.fit(x=x, y=y, batch_size=self.hyperParameters.batch_size, epochs=self.hyperParameters.num_epochs, verbose=True)
        return history

    def fitGenerator(self, start, batchSize: int, epochs: int, folder: str, generator: Generator) -> Tuple[Dict, Dict]:
        epochCount = 0

        headLossData = {"bowel": [], "extra": [], "liver": [], "kidney": [], "spleen": []}
        headAccData = {"bowel": [], "extra": [], "liver": [], "kidney": [], "spleen": []}

        binaryAccBowel = tf.keras.metrics.CategoricalAccuracy()
        binaryAccExtra = tf.keras.metrics.CategoricalAccuracy()
        categoricalAccLiver = tf.keras.metrics.CategoricalAccuracy()
        categoricalAccKidney = tf.keras.metrics.CategoricalAccuracy()
        categoricalAccSpleen = tf.keras.metrics.CategoricalAccuracy()

        lossBowelFunc = tf.keras.losses.CategoricalCrossentropy()
        lossExtraFunc = tf.keras.losses.CategoricalCrossentropy()
        lossLiverFunc = tf.keras.losses.CategoricalCrossentropy()
        lossKidneyFunc = tf.keras.losses.CategoricalCrossentropy()
        lossSpleenFunc = tf.keras.losses.CategoricalCrossentropy()

        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)

        npzFiles = os.listdir(folder)
        y_names: list = ["bowel", "extra", "liver", "kidney", "spleen"]

        for epoch in range(epochs):
            
            print(f"epoch: {epochCount}/{epoch}")
            binaryAccBowel.reset_states()
            binaryAccExtra.reset_states()
            categoricalAccLiver.reset_states()
            categoricalAccKidney.reset_states()
            categoricalAccSpleen.reset_states()
            lossBowelFunc

            iterator = generator(start, batchSize, folder, npzFiles, y_names)
            for step in range(start, batchSize):
                x_train, y_train = next(iterator)
                with tf.GradientTape() as tape:
                    logits = self.model(x_train, training=True)
                    lossBowel = lossBowelFunc(y_train['bowel'], logits[0])
                    lossExtra = lossExtraFunc(y_train['extra'], logits[1])
                    lossLiver = lossLiverFunc(y_train['liver'], logits[2])
                    lossKidney = lossKidneyFunc(y_train['kidney'], logits[3])
                    lossSpleen = lossSpleenFunc(y_train['spleen'], logits[4])
                    loss = (lossBowel + lossExtra + lossLiver + lossKidney + lossSpleen)/5
                    
                binaryAccBowel.update_state(y_train['bowel'], logits[0])
                binaryAccExtra.update_state(y_train['extra'], logits[1])
                categoricalAccLiver.update_state(y_train['liver'], logits[2])
                categoricalAccKidney.update_state(y_train['kidney'], logits[3])
                categoricalAccSpleen.update_state(y_train['spleen'], logits[4])

                grads = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                headLossData["bowel"].append(lossBowel)
                headLossData["extra"].append(lossExtra)
                headLossData["liver"].append(lossLiver)
                headLossData["kidney"].append(lossKidney)
                headLossData["spleen"].append(lossSpleen)

                headAccData["bowel"].append(binaryAccBowel.result())
                headAccData["extra"].append(binaryAccExtra.result())
                headAccData["liver"].append(categoricalAccLiver.result())
                headAccData["kidney"].append(categoricalAccKidney.result())
                headAccData["spleen"].append(categoricalAccSpleen.result())

                print(f"Step: {step}")
                print("Loss, Acc (Bowel) %.4f: %.4f" % (float(lossBowel), binaryAccBowel.result()))
                print("Loss, Acc (Extra) %.4f: %.4f" % (float(lossExtra), binaryAccExtra.result()))
                print("Loss, Acc (Liver) %.4f: %.4f" % (float(lossLiver), categoricalAccLiver.result()))
                print("Loss, Acc (Kidney) %.4f: %.4f" % (float(lossKidney), categoricalAccKidney.result()))
                print("Loss, Acc (Spleen) %.4f: %.4f" % (float(lossSpleen), categoricalAccSpleen.result()))

        return (headLossData, headAccData)

    def saveWeights(self, fileName: str) -> None:
        self.model.save_weights(fileName)
    
    def loadWeights(self, fileName: str) -> None:
        self.model.load_weights(fileName)

    def saveModel(self, fileName: str) -> None:
        self.model.save(fileName, save_format='tf')

    def loadModel(self, fileName: str) -> None:
        self.model = tf.keras.models.load_model(fileName)

    def evaluateT(self, x_test: List[tf.Tensor], y_test: List[tf.Tensor]) -> Dict:

        bowelAcc = tf.keras.metrics.CategoricalAccuracy()
        extraAcc = tf.keras.metrics.CategoricalAccuracy()
        liverAcc = tf.keras.metrics.CategoricalAccuracy()
        kidneyAcc = tf.keras.metrics.CategoricalAccuracy()
        spleenAcc = tf.keras.metrics.CategoricalAccuracy()

        bowelF1 = tf.keras.metrics.F1Score()
        extraF1 = tf.keras.metrics.F1Score()
        liverF1 = tf.keras.metrics.F1Score()
        kidneyF1 = tf.keras.metrics.F1Score()
        spleenF1 = tf.keras.metrics.F1Score()

        bowelPrec = tf.keras.metrics.Precision()
        extraPrec = tf.keras.metrics.Precision()
        liverPrec = tf.keras.metrics.Precision()
        kidneyPrec = tf.keras.metrics.Precision()
        spleenPrec = tf.keras.metrics.Precision()

        bowelRec = tf.keras.metrics.Recall()
        extraRec = tf.keras.metrics.Recall()
        liverRec = tf.keras.metrics.Recall()
        kidneyRec = tf.keras.metrics.Recall()
        spleenRec = tf.keras.metrics.Recall()


        for x, y in zip(x_test, y_test):
            logits = self.model(x, training=False)

            bowelAcc.update_state(y['bowel'], logits[0])
            extraAcc.update_state(y['extra'], logits[1])
            liverAcc.update_state(y['liver'], logits[2])
            kidneyAcc.update_state(y['kidney'], logits[3])
            spleenAcc.update_state(y['spleen'], logits[4])

            bowelF1.update_state(y['bowel'], logits[0])
            extraF1.update_state(y['extra'], logits[1])
            liverF1.update_state(y['liver'], logits[2])
            kidneyF1.update_state(y['kidney'], logits[3])
            spleenF1.update_state(y['spleen'], logits[4])

            bowelPrec.update_state(y['bowel'], logits[0])
            extraPrec.update_state(y['extra'], logits[1])
            liverPrec.update_state(y['liver'], logits[2])
            kidneyPrec.update_state(y['kidney'], logits[3])
            spleenPrec.update_state(y['spleen'], logits[4])

            bowelRec.update_state(y['bowel'], logits[0])
            extraRec.update_state(y['extra'], logits[1])
            liverRec.update_state(y['liver'], logits[2])
            kidneyRec.update_state(y['kidney'], logits[3])
            spleenRec.update_state(y['spleen'], logits[4])

        headAccData = {
            "bowel": [bowelAcc.result(), bowelF1.result(), bowelPrec.result(), bowelRec.result()],
            "extra": [extraAcc.result(), extraF1.result(), extraPrec.result(), extraRec.result()], 
            "liver": [liverAcc.result(), liverF1.result(), liverPrec.result(), liverRec.result()],
            "kidney": [kidneyAcc.result(), kidneyF1.result(), kidneyPrec.result(), kidneyRec.result()],
            "spleen": [spleenAcc.result(), spleenF1.result(), spleenPrec.result(), spleenRec.result()]
            }

        return headAccData

    def predictT(self, x_train: tf.Tensor) -> tf.Tensor:
        return self.model.predict(x_train)

    def printSummary(self) -> None:
        print(self.model.summary())

    def __getSummary(self, fileName: str) -> None:
        with open(fileName,'a') as f:
            print(fileName, file=f)

    def exportSummary(self, fileName: str) -> None:
            self.model.summary(print_fn=self.__getSummary(fileName))







        


