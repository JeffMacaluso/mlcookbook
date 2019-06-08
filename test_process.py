# Note: https://realpython.com/python-modules-packages/ 
import numpy as np
import pandas as pd

import mlcookbook

from sklearn.datasets import load_boston, load_breast_cancer

boston_data = load_boston()
data = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
data['Target'] = boston_data.target

breast_cancer_data = load_breast_cancer()
categorical_data = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)
categorical_data['Target'] = breast_cancer_data.target

print()
print('----------------------------------------------')
print('Testing Preprocessing')
print()

print('-----------')
print('PCA test Test')
mlcookbook.process.fit_PCA(data)
print()

print('-----------')
print('Oversampling Test')
mlcookbook.process.oversample_binary_label(categorical_data, 'Target')
print()

print('-----------')
print('Oversampling with SMOTE Test')
# TODO: Fix tihs
# mlcookbook.process.fit_PCA(categorical_data.drop('Target', axis=1), categorical_data['Target'])
# mlcookbook.process.fit_PCA(categorical_data, 'Target')
print()

print('-----------')
print('Target Mean Encoding Test')
# TODO: Fix this
mlcookbook.process.target_encode(data.drop('Target', axis=1), data.drop('Target', axis=1), data['Target'])
print()
