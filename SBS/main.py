import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame([
['green', 'M', 10.1, 'class1'],
['red', 'L', 13.5, 'class2'],
['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

size_mapping = { 'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)
print(df)

class_mapping = {label:idx for idx, label in
                 enumerate(np.unique(df['classlabel']))}
print(class_mapping)

X = df[['color','size','price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)

#one-hot encoding
ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()
array = ([[ 0. ,1. ,0. ,1. , 10.1],
       [ 0. ,0. ,1. ,2. , 13.5],
       [ 1. ,0. ,0. ,3. , 15.3]])