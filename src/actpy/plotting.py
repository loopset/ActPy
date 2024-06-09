from typing import cast

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import numpy as np


# Class that plots profile data
# And optionally executes a model
class ProfilePlotter:
    def __init__(self, data: np.ndarray | None = None, model=None) -> None:
        # Matplotlib things
        self.setStyle()
        self.fig = plt.figure(1, figsize=(12, 8))
        self.ax = cast(plt.Subplot, self.fig.gca())
        self.initButtons()
        # Data
        self.data = data if data is not None else np.random.rand(10, 128)
        # Iterator
        self.it = -1
        # Prediction model
        self.model = model if model else None
        # Start with first event
        self.next(None)

    def initButtons(self) -> None:
        # Next button
        self.anext = cast(plt.Axes, plt.axes([0, 0, 1, 1]))
        ln = InsetPosition(self.ax, [0.85, -0.1, 0.1, 0.05])
        self.anext.set_axes_locator(ln)
        self.bnext = Button(self.anext, "Next")
        self.bnext.on_clicked(self.next)
        # Previous button
        self.aprev = cast(plt.Axes, self.fig.add_axes([0, 0, 1, 1]))
        lb = InsetPosition(self.ax, [0.7, -0.1, 0.1, 0.05])
        self.aprev.set_axes_locator(lb)
        self.bprev = Button(self.aprev, "Previous")
        self.bprev.on_clicked(self.previous)

    def previous(self, event) -> None:
        self.it -= 1
        if self.it < 0:
            print("ProfilePlotter.previous():  reached begin of dataset")
            self.it = 0
        self.draw()

    def next(self, event) -> None:
        self.it += 1
        if self.it >= self.data.shape[0]:
            print("ProfilePlotter.next(): reached end of dataset")
            self.it = self.data.shape[0] - 1
        self.draw()

    def useModel(self) -> None:
        # Adds title to axis with prediction
        pred = "No model"
        proba = float()
        if self.model:
            # Keras model
            if 'keras' in type(self.model).__module__:
                ps = self.model.predict(self.data[self.it].reshape(1, -1), verbose=0)[0]
                pred = np.argmax(ps)
                proba = ps[pred]
        #     proba = max(self.model.predict_proba(self.data[self.it].reshape(1, -1))[0])
        self.ax.set_title("Label : {0} with p : {1:.1f} %".format(pred, proba * 100))

    def plotData(self) -> None:
        # Y axis
        y = self.data[self.it]
        # X axis
        x = [i + 0.5  for i in range(len(y))]
        # Bins (must contain last bin, [127, 128))
        bins = [i for i in range(len(y) + 1)]
        # Indeed plot
        self.ax.hist(x, bins= bins, weights=y, histtype='step')
        # self.ax.plot(x, y, color="royalblue")
        # Reset axis labels
        self.ax.set_xlabel("Pad")
        self.ax.set_ylabel("Norm. dE / dx [MeV / pad]")
        self.fig.suptitle("Entry : {0}".format(self.it))

    def draw(self) -> None:
        # Clear axes
        self.ax.clear()
        # Do stuff
        self.plotData()
        self.useModel()
        # And draw everything
        self.ax.figure.canvas.draw()

    def setStyle(self) -> None:
        plt.rcParams["axes.formatter.use_locale"] = True
        plt.rcParams["text.usetex"] = False
        plt.rcParams["axes.linewidth"] = 1.0
        plt.rcParams["xtick.major.width"] = 0.95
        plt.rcParams["ytick.major.width"] = 0.95
        plt.rcParams["xtick.minor.width"] = 0.75
        plt.rcParams["ytick.minor.width"] = 0.75
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.serif"] = "DejaVu Sans"
        plt.rcParams["font.size"] = 12
        plt.rcParams["mathtext.default"] = "regular"
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["ytick.minor.visible"] = True
        plt.rcParams["xtick.minor.visible"] = True
        plt.rcParams["xtick.major.size"] = 8
        plt.rcParams["ytick.major.size"] = 8
        plt.rcParams["xtick.minor.size"] = 5
        plt.rcParams["ytick.minor.size"] = 5
        plt.rcParams["xtick.top"] = True
        plt.rcParams["ytick.right"] = True
        plt.rcParams["legend.edgecolor"] = "0.5"
        plt.rcParams["legend.shadow"] = True
        plt.rcParams["legend.edgecolor"] = "0.5"
        plt.rcParams["legend.framealpha"] = None
        plt.rcParams["legend.handletextpad"] = 0.25  ##distancia do simbolo á letra
        plt.rcParams["axes.formatter.use_mathtext"] = True
        plt.rcParams["figure.titlesize"] = 22
        plt.rcParams["legend.fontsize"] = 18
        plt.rcParams["axes.labelsize"] = 20
        plt.rcParams["axes.titlesize"] = 20
        plt.rcParams["xtick.labelsize"] = 18
        plt.rcParams["ytick.labelsize"] = 18
