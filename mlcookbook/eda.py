import numpy as np
import pandas as pd

# TODO: Add function to dynamically determine categorical columns
# Check ideas from here https://datascience.stackexchange.com/questions/9892/how-can-i-dynamically-distinguish-between-categorical-data-and-numerical-data
# and here https://stackoverflow.com/questions/35826912/what-is-a-good-heuristic-to-detect-if-a-column-in-a-pandas-dataframe-is-categori 


# Printing the percentage of missing values per column
def percent_missing(dataframe):
    '''
    Prints the percentage of missing values for each column in a dataframe
    '''
    # Summing the number of missing values per column and then dividing by the total
    sumMissing = dataframe.isnull().values.sum(axis=0)
    pctMissing = sumMissing / dataframe.shape[0]
    
    if sumMissing.sum() == 0:
        print('No missing values')
    else:
        # Looping through and printing out each columns missing value percentage
        print('Percent Missing Values:', '\n')
        for idx, col in enumerate(dataframe.columns):
            if sumMissing[idx] > 0:
                print('{0}: {1:.2f}%'.format(col, pctMissing[idx] * 100))


def iqr_indices_of_outliers(X):
    '''
    Detects outliers using the interquartile range (IQR) method
    
    Input: An array of a variable to detect outliers for
    Output: An array with indices of detected outliers

    # Note: The function in its current form is taken from Chris Albon's Machine Learning with Python Cookbook
    # TODO: Update this to dynamically accept multiple features at once
    '''
    q1, q3 = np.percentile(X, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    outlier_indices = np.where((X > upper_bound) | (X < lower_bound))
    return outlier_indices


def z_score_indices_of_outliers(X, threshold=3):
    '''
    Detects outliers using the Z score method method
    
    Input: - X: An array of a variable to detect outliers for
           - threshold: The number of standard deviations from the mean
                        to be considered an outlier
                        
    Output: An array with indices of detected outliers
    # TODO: Update this to dynamically accept multiple features at once
    '''
    X_mean = np.mean(X)
    X_stdev = np.std(X)
    z_scores = [(y - X_mean) / X_stdev for y in X]
    outlier_indices = np.where(np.abs(z_scores) > threshold)
    return outlier_indices


def percentile_indices_of_outliers(X, percentile_threshold=0.1):
    '''
    Determines outliers based off of percentiles
    
    Input: An array of one variable to detect outliers for
    Output: An array with indices of detected outliers
    '''
    diff = (1 - percentile_threshold) / 2.0
    minval, maxval = np.percentile(X, [diff, 100 - diff])
    outlier_indices = np.where((X < minval) | (X > maxval))
    return outlier_indices


def ellipses_indices_of_outliers(X, contamination=0.1):
    '''
    Detects outliers using the elliptical envelope method
    
    Input: An array of all variables to detect outliers for
    - TODO: Put note for what contamination is
    Output: An array with indices of detected outliers
    '''
    from sklearn.covariance import EllipticEnvelope
    
    # Copying to prevent changes to the input array
    X = X.copy()
    
    # Dropping categorical columns
    non_categorical = []
    for feature in range(X.shape[1]):
        num_unique_values = len(np.unique(X.iloc[:, feature]))
        if num_unique_values > 30:
            non_categorical.append(feature)
    X = X.iloc[:, non_categorical]  # Subsetting to columns without categorical indexes

    # Testing if there are an adequate number of features
    if X.shape[0] < X.shape[1] ** 2.:
        print('Will not perform well. Reduce the dimensionality and try again.')
        return
    
    # Creating and fitting the detector
    outlier_detector = EllipticEnvelope(contamination=contamination)
    outlier_detector.fit(X)
    
    # Predicting outliers and outputting an array with 1 if it is an outlier
    outliers = outlier_detector.predict(X)
    outlier_indices = np.where(outliers == -1)
    return outlier_indices


def isolation_forest_indices_of_outliers(X, contamination=0.1, n_estimators=100):
    '''
    Detects outliers using the isolation forest method
    
    Input: An array of all variables to detect outliers for
    Output: An array with indices of detected outliers
    '''
    from sklearn.ensemble import IsolationForest
    
    # Copying to prevent changes to the input array
    X = X.copy()
    
    # Dropping categorical columns
    non_categorical = []
    for feature in range(X.shape[1]):
        num_unique_values = len(np.unique(X.iloc[:, feature]))
        if num_unique_values > 30:
            non_categorical.append(feature)
    X = X.iloc[:, non_categorical]  # Subsetting to columns without categorical indexes
    
    # Creating and fitting the detector
    outlier_detector = IsolationForest(contamination=contamination, n_estimators=n_estimators,
                                       behaviour='new')  # To prevent warnings
    outlier_detector.fit(X)
    
    # Predicting outliers and outputting an array with 1 if it is an outlier
    outliers = outlier_detector.predict(X)
    outlier_indices = np.where(outliers == -1)
    return outlier_indices


def one_class_svm_indices_of_outliers(X):
    '''
    Detects outliers using the one class SVM method
    
    Input: An array of all variables to detect outliers for
    Output: An array with indices of detected outliers
    '''
    from sklearn.svm import OneClassSVM
    
    # Copying to prevent changes to the input array
    X = X.copy()
    
    # Dropping categorical columns
    non_categorical = []
    for feature in range(X.shape[1]):
        num_unique_values = len(np.unique(X.iloc[:, feature]))
        if num_unique_values > 30:
            non_categorical.append(feature)
    X = X.iloc[:, non_categorical]  # Subsetting to columns without categorical indexes

    # Testing if there are an adequate number of features
    if X.shape[0] < X.shape[1] ** 2.:
        print('Will not perform well. Reduce the dimensionality and try again.')
        return
    
    # Creating and fitting the detector
    outlier_detector = OneClassSVM()
    outlier_detector.fit(X)
    
    # Predicting outliers and outputting an array with 1 if it is an outlier
    outliers = outlier_detector.predict(X)
    outlier_indices = np.where(outliers == -1)
    return outlier_indices


def outlier_report(dataframe, z_threshold=3, per_threshold=0.95, contamination=0.1, n_trees=100):
    '''
    TODO: - Write Docstring
          - Finish commenting function
          - Remove redundant functions
    '''
    
    # Converting to a pandas dataframe if it is an array
    if type(dataframe) != 'pandas.core.frame.DataFrame':
        try:
            dataframe = pd.DataFrame(dataframe)
        except:
            return 'Must be either a dataframe or a numpy array'
    
    # Creating a copy to avoid fidelity issues
    dataframe = dataframe.copy()
    
    # Dropping categorical columns
    dataframe = dataframe.select_dtypes(exclude=['bool_'])
    for column in dataframe.columns:
        num_unique_values = len(dataframe[column].unique())
        if num_unique_values < 30:
            dataframe = dataframe.drop(column, axis=1)
    
    # Dictionaries for individual features to be packaged into a master dictionary
    iqr_outlier_indices = {}
    z_score_outlier_indices = {}
    percentile_outlier_indices = {}
    multiple_outlier_indices = {}  # Indices with two or more detections
    
    print('Detecting outliers', '\n')
    
    # Creating an empty data frame to fill with results
    results = pd.DataFrame(columns=['IQR', 'Z Score', 'Percentile', 'Multiple'])
    
    # Single column outlier tests
    print('Single feature outlier tests')
    for feature in range(dataframe.shape[1]):
        
        # Gathering feature names for use in output dictionary and results dataframe
        feature_name = dataframe.columns[feature]
        
        # Finding outliers
        iqr_outliers = iqr_indices_of_outliers(dataframe.iloc[:, feature])[0]
        z_score_outliers = z_score_indices_of_outliers(dataframe.iloc[:, feature])[0]
        percentile_outliers = percentile_indices_of_outliers(dataframe.iloc[:, feature])[0]
        multiple_outliers = np.intersect1d(iqr_outliers, z_score_outliers)  # TODO: Fix this
        
        # Adding to the empty dictionaries
        iqr_outlier_indices[feature_name] = iqr_outliers
        z_score_outlier_indices[feature_name] = z_score_outliers
        percentile_outlier_indices[feature_name] = percentile_outliers
        multiple_outlier_indices[feature_name] = multiple_outliers
        
        # Adding to results dataframe
        outlier_counts = {'IQR': len(iqr_outliers),
                          'Z Score': len(z_score_outliers),
                          'Percentile': len(percentile_outliers),
                          'Multiple': len(multiple_outliers)}
        outlier_counts_series = pd.Series(outlier_counts, name=feature_name)
        results = results.append(outlier_counts_series)
    
    # Calculating the subtotal of outliers found
    results_subtotal = results.sum()
    results_subtotal.name = 'Total'
    results = results.append(results_subtotal)
    
    # Calculating the percent of total values in each column
    num_observations = dataframe.shape[0]
    results['IQR %'] = results['IQR'] / num_observations
    results['Z Score %'] = results['Z Score'] / num_observations
    results['Percentile %'] = results['Percentile'] / num_observations
    results['Multiple %'] = results['Multiple'] / num_observations
    
    # Printing the results dataframe as a table
    print(results, '\n')
    
    # All column outlier tests
    print('All feature outlier tests')
    ellipses_envelope_outlier_indices = ellipses_indices_of_outliers(dataframe)
    print('- Ellipses Envelope: {0}'.format(len(ellipses_envelope_outlier_indices[0])))
    
    isolation_forest_outlier_indices = isolation_forest_indices_of_outliers(dataframe)
    print('- Isolation Forest: {0}'.format(len(isolation_forest_outlier_indices[0])))

    one_class_svm_outlier_indices = one_class_svm_indices_of_outliers(dataframe)
    print('- One Class SVM: {0}'.format(len(one_class_svm_outlier_indices[0])))

    # Putting together the final dictionary for output
    all_outlier_indices = {}
    all_outlier_indices['Ellipses Envelope'] = ellipses_envelope_outlier_indices
    all_outlier_indices['Isolation Forest'] = isolation_forest_outlier_indices
    all_outlier_indices['One Class SVM'] = one_class_svm_outlier_indices
    all_outlier_indices['IQR'] = iqr_outlier_indices
    all_outlier_indices['Z Score'] = z_score_outlier_indices
    all_outlier_indices['Percentile'] = percentile_outlier_indices
    all_outlier_indices['Multiple'] = multiple_outlier_indices
    
    return all_outlier_indices