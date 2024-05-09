import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc


import tensorflow as tf

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

print(dftrain.head()) #head() shows the first 5 entries

y_train = dftrain.pop('survived') #stores the survived columns in y_train column
y_eval = dfeval.pop('survived')

#after removing survived column
print(dftrain.head())

print(y_train)

dftrain.describe()

dftrain.shape
dftrain.age.hist(bins=20)
dftrain.sex.value_counts().plot(kind='barh')
dftrain['class'].value_counts().plot(kind='barh')
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').sex_xlabel('%survive')

