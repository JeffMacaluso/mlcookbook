# Note: https://realpython.com/python-modules-packages/ 
import numpy as np
import pandas as pd

import mlcookbook

data = pd.DataFrame({'x1': [1, 5, 30000],
                     'x2': [3, 10, 9],
                     'y': [1, 9, 11]})
print(data)

mlcookbook.eda.percent_missing(data)

mlcookbook.eda.outlier_report(data)