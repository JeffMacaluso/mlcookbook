import mlcookbook.eda
import mlcookbook.ml
import mlcookbook.misc
import mlcookbook.nlp
import mlcookbook.plot
import mlcookbook.process

def diagnostics():
    '''
    Reports the current date/time, package versions, and machine hardware
    
    TODO: Make this work dynamically with imported libraries
    '''
    import sys
    import os
    import time
    import numpy as np
    import pandas as pd
    import sklearn
    print(time.strftime('%Y/%m/%d %H:%M'))
    print('OS:', sys.platform)
    print('CPU Cores:', os.cpu_count())
    print('Python:', sys.version)
    print('NumPy:', np.__version__)
    print('Pandas:', pd.__version__)
    print('Scikit-Learn:', sklearn.__version__)
