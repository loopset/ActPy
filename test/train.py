from pathlib import Path
import sys

import ROOT
from keras import layers, models, callbacks, optimizers
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Append ActPy src to path
path_src = Path(__file__).parents[1]
sys.path.append(str(path_src) + "/src")

from actpy import rinterface


# Read the data
reactions = ROOT.RDataFrame(
    "tree", "/media/Data/E864/Simulations/Outputs/with_n.root"
).AsNumpy()["data"]
rdf = rinterface.ROOTtoPandas(reactions, True, 1)

noise = ROOT.RDataFrame(
    "tree", "/media/Data/E864/Simulations/Outputs/with_8Li.root"
).AsNumpy()["data"]
ndf = rinterface.ROOTtoPandas(noise, True, 0)

# Prepare to sklearn
X, y = rinterface.prepareToTrain([rdf, ndf])

# Split in train and test datasets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Shape of data
ishape = (128, 1)
model = models.Sequential()
model.add(layers.Input(shape=ishape))
model.add(layers.Conv1D(filters=20, kernel_size=2, activation="relu", padding="same"))
model.add(layers.MaxPooling1D())
model.add(layers.Conv1D(filters=16, kernel_size=3, activation="relu", padding="same"))
model.add(layers.MaxPooling1D())
model.add(layers.Conv1D(filters=12, kernel_size=4, activation="relu", padding="same"))
model.add(layers.MaxPooling1D())
model.add(layers.Flatten())
model.add(layers.Dense(units=128, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))
model.summary()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Use early stop
early_Stop = callbacks.EarlyStopping(monitor="val_loss", patience=4)

# Train
history = model.fit(
    Xtrain, ytrain, epochs=50, validation_split=0.2, callbacks=[early_Stop]
)

# Save model
model.save("cnn1d.keras")

# Plot confusion matrix
names = ["11B", "12C"]
ypred = model.predict(Xtest)
ypred = [np.argmax(preds) for preds in ypred]
cm = confusion_matrix(ytest, ypred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
disp.plot()
plt.gcf().suptitle("CNN1d Classifier")
plt.show()
