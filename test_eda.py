# Note: https://realpython.com/python-modules-packages/ 
import numpy as np
import pandas as pd

import mlcookbook

from sklearn.datasets import load_boston

boston_data = load_boston()
data = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
data['Target'] = boston_data.target

print()
print('----------------------------------------------')
print('Testing EDA')
print()

print('-----------')
print('Percent Missing Test')
mlcookbook.eda.percent_missing(data)
print()

print('-----------')
print('IQR outliers')
outliers = mlcookbook.eda.iqr_indices_of_outliers(data.iloc[:, 0])
print(outliers)
print()

print('-----------')
print('Z score outliers')
outliers = mlcookbook.eda.z_score_indices_of_outliers(data.iloc[:, 0])
print(outliers)
print()

print('-----------')
print('Percentile outliers')
outliers = mlcookbook.eda.percentile_indices_of_outliers(data.iloc[:, 0])
print(outliers)
print()

print('-----------')
print('Ellipses outliers')
outliers = mlcookbook.eda.ellipses_indices_of_outliers(data)
print(outliers)
print()

print('-----------')
print('Isolation Forest outliers')
outliers = mlcookbook.eda.isolation_forest_indices_of_outliers(data)
print(outliers)
print()

print('-----------')
print('One class SVM outliers')
outliers = mlcookbook.eda.one_class_svm_indices_of_outliers(data)
print(outliers)
print()

print('-----------')
print('Outlier Report')
mlcookbook.eda.outlier_report(data)
print()