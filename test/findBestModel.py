from pathlib import Path
import sys

import ROOT
from keras import layers, models, callbacks, optimizers
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
import keras_tuner as kt

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

# Method to find best parameters of network
def build(hp):
    ishape = (128, 1)
    cnn = models.Sequential()
    cnn.add(layers.Input(shape=ishape))
    # hp_filters = hp.Choice('init_filter', values=[20, 30, 40, 50])
    cnn.add(layers.Conv1D(filters=20, kernel_size=2, activation="relu", padding="same"))
    cnn.add(layers.MaxPooling1D())
    cnn.add(layers.Conv1D(filters=16, kernel_size=3, activation="relu", padding="same"))
    cnn.add(layers.MaxPooling1D())
    cnn.add(layers.Conv1D(filters=12, kernel_size=4, activation="relu", padding="same"))
    cnn.add(layers.MaxPooling1D())
    cnn.add(layers.Flatten())
    # hp_final = hp.Int("dense_units", min_value=16, max_value=128, step=20)
    cnn.add(layers.Dense(units=128, activation="relu"))
    cnn.add(layers.Dense(2, activation="softmax"))
    cnn.summary()
    hp_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    cnn.compile(
        optimizer=optimizers.Adam(learning_rate=hp_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return cnn


tuner = kt.Hyperband(
    build,
    objective="val_accuracy",
    max_epochs=10,
    factor=3,
    hyperband_iterations=2,
    directory="my_dir",
    project_name="v1",
    overwrite=True,
)

early_Stop = callbacks.EarlyStopping(monitor='val_loss', patience=4)

tuner.search(Xtrain, ytrain, epochs = 50, validation_split = 0.2, callbacks=[early_Stop])

best_models = tuner.get_best_models(num_models=1)
best_model = best_models[0]
print('Best model has this architecture : ')
best_model.summary()

best_hps=tuner.get_best_hyperparameters()[0]
print('Best learning rate : ', best_hps.get("learning_rate"))

# Find the best set of epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(Xtrain, ytrain, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# And fit the model with the best set of hyperparameters
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(Xtrain, ytrain, epochs=best_epoch, validation_split=0.2)

