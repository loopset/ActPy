import numpy as np
import pandas as pd


# Function to convert RDataFrame to Pandas and add optional label
def ROOTtoPandas(data, uselabel: bool = False, label: int = 0) -> pd.DataFrame:
    df = pd.DataFrame({"x": data})
    if uselabel:
        df["y"] = [label for _ in range(df.size)]
    return df


# Converts std::vectors to np.array
def stdToNumpy(x, norm : bool = True) -> np.ndarray:
    # Create output array
    xret = np.zeros((x.shape[0], x[0].size()))
    # Populate it
    for i, vec in enumerate(x):
        ## Compute sum
        normalization = sum([val for val in vec]) if norm else 1
        for j, val in enumerate(vec):
            xret[i, j] = val / normalization
    return xret


# Merges a list of DFs, mixes them and returns X and y separately
def prepareToTrain(dfs: list, random: bool = True, norm: bool = True) -> tuple:
    df = pd.concat(dfs)
    if random:
        df = df.sample(frac=1)
    X, y = df.T.to_numpy()
    X = stdToNumpy(X, norm)
    y = y.astype(int)
    return X, y

# Applies stdToNumpy in pandas DataFrame
def processDF(df : pd.DataFrame, norm : bool = True) -> np.ndarray:
    return stdToNumpy(df.loc[:, 'x'].values, norm)

