import numpy as np
import pandas as pd


def get_x(df):
    ones = pd.DataFrame({'ones': np.ones(len(df))})
    data = pd.concat([ones, df], axis=1)
    return data.iloc[:, :-1].values


def get_y(df):
    return np.array(df.iloc[:, -1])


def normalize_feature(df):
    return df.apply(lambda colum: (colum - colum.mean()) / colum.std())
