#PyTorch Training Pipeline.
#Sample Example.

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("breast-cancer.csv")
print(df.head())