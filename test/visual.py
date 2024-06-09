# from pathlib import Path
import sys

sys.path.append('/media/Data/ActPy/src/')

from actpy import plotting, rinterface
import matplotlib.pyplot as plt
import numpy as np
import ROOT
from keras.models import load_model


# Read data
# root = ROOT.RDataFrame("tree", "/media/Data/E864/Simulations/Outputs/with_12C.root").AsNumpy()["data"]
# df = rinterface.ROOTtoPandas(root)
#
# X = rinterface.processDF(df)
#
# # Get the model
# filename = "./cnn1d.keras"
# model = load_model(filename)

# Plot
plotting.ProfilePlotter()
plt.show()
