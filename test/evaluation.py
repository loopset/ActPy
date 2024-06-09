from actpy import rinterface, plotting
import ROOT
from keras import models
import numpy as np
import pandas as pd


rdf = ROOT.RDataFrame("tree", "/media/Data/E864/Simulations/Outputs/with_12C.root").AsNumpy()["data"]

df = rinterface.ROOTtoPandas(rdf)

X = rinterface.processDF(df)

# Read model
model = models.load_model("./cnn1d.keras")

# eval the model
evals = model.predict(X)

labels = np.argmax(evals, axis=1)

final = ROOT.RDF.FromNumpy({"label" : labels})
final.Snapshot("Class_Tree", "classification.root")


