import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


if __name__ == '__main__':
    dataframe = pd.read_csv('mirnas.csv')
    print(dataframe.columns)

