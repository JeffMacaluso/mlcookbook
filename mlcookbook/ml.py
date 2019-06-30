import numpy as np
import pandas as pd

# Random Search
def hyperparameter_random_search(X, y, model=None, parameters=None, num_folds=5, num_iterations=50):
    '''
    Performs a random search on hyperparameters and 

    TODO: Finish docstring
        - Add cross validation method
        - Add status bar
    '''
    # Randomized Search
    from sklearn.model_selection import RandomizedSearchCV
    import datetime

    # Making sure a model or parameters exists
    if model is None:
        print('Please provide a model')
        return
    if parameters is None:
        print('Please provide parameters for the model')
        return

    # Performing randomized search
    model = RandomizedSearchCV(model, param_distributions=parameters,
                               n_iter=num_iterations, n_jobs=-1, cv=num_folds,
                               verbose=0)
    print('Beginning random search at {0}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    model.fit(X, y)
    print('Completed random search at {0}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print()

    # Reporting the results
    print('Best Estimator:', model.best_estimator_)
    print('Best Parameters:', model.best_params_)
    print('Best Score:', model.best_score_)
    
    return model

# TODO: Add grid search

# TODO: Add xgboost probability threshold search

# Probability Threshold Search - scikit-learn
def optimal_probability_cutoff(model, test_dataset, test_labels, max_thresh=0.99, step_size=0.01):
    '''
    Finds the optimal probability cutoff to maximize the F1 score
    Returns the optimal probability cutoff, F1 score, and a plot of the results

    TODO: 
        - Add precision, recall, and accuracy
    '''
    from sklearn import metrics
    import matplotlib.pyplot as plt

    # Prediction probabilities of the test dataset
    predicted = model.predict_proba(test_dataset)[:, 1]

    # Creating an empty dataframe to fill with probability cutoff thresholds and f1 scores
    results = pd.DataFrame(columns=['Threshold', 'F1 Score'])

    # Setting f1 score average metric based on binary or multi-class classification
    if len(np.unique(test_labels)) == 2:
        avg = 'binary'
    else:
        avg = 'micro'

    # Looping trhough different probability thresholds
    for thresh in np.arange(0, (max_thresh+step_size), step_size):
        pred_bin = pd.Series(predicted).apply(lambda x: 1 if x > thresh else 0)
        f1 = metrics.f1_score(test_labels, pred_bin, average=avg)
        tempResults = {'Threshold': thresh, 'F1 Score': f1}
        results = results.append(tempResults, ignore_index=True)
        
    # Plotting the F1 score throughout different probability thresholds
    plt.figure(figsize=(7, 5))
    results.plot(x='Threshold', y='F1 Score')
    plt.title('F1 Score by Probability Cutoff Threshold')
    plt.ylabel('F1 Score')
    plt.show()
    
    best_index = list(results['F1 Score']).index(max(results['F1 Score']))
    print('Threshold for Optimal F1 Score:')
    return results.iloc[best_index]


# Prediction Intervals - Ensemble Scikit-Learn Models
def ensemble_prediction_intervals(model, X, X_train=None, y_train=None, percentile=0.95):
    '''
    Calculates the specified prediction intervals for each prediction
    from an ensemble scikit-learn model.
    
    Inputs:
        - model: The scikit-learn model to create prediction intervals for. This must be
                 either a RandomForestRegressor or GradientBoostingRegressor
        - X: The input array to create predictions & prediction intervals for
        - X_train: The training features for the gradient boosted trees
        - y_train: The training label for the gradient boosted trees
        - percentile: The prediction interval percentile. Default of 0.95 is 0.025 - 0.975
    
    Note: Use X_train and y_train when using a gradient boosted regressor because a copy of
          the model will be re-trained with quantile loss.
          These are not needed for a random forest regressor
    
    Output: A dataframe with the predictions and prediction intervals for X
    
    TO-DO: 
      - Try to optimize by removing loops where possible
      - Fix upper prediction intervals for gradient boosted regressors
      - Add xgboost
    '''
    # Checking if the model has the estimators_ attribute
    if 'estimators_' not in dir(model):
        print('Not an ensemble model - exiting function')
        return

    # Accumulating lower and upper prediction intervals
    lower_PI = []
    upper_PI = []
    
    # Generating predictions to be returned with prediction intervals
    print('Generating predictions with the model')
    predictions = model.predict(X)
    
    # Prediction intervals for a random forest regressor
    # Taken from https://blog.datadive.net/prediction-intervals-for-random-forests/
    if str(type(model)) == "<class 'sklearn.ensemble.forest.RandomForestRegressor'>":
        print('Generating upper and lower prediction intervals')
        
        # Looping through individual records for predictions
        for record in range(len(X)):
            estimator_predictions = []
        
            # Looping through estimators and gathering predictions
            for estimator in model.estimators_:
                individual_estimator_predictions = estimator.predict(X.iloc[record].values.reshape(1, -1))[0]
                estimator_predictions.append(individual_estimator_predictions)
            
            # Adding prediction intervals
            lower_PI.append(np.percentile(estimator_predictions, (1 - percentile) / 2.))
            upper_PI.append(np.percentile(estimator_predictions, 100 - (1 - percentile) / 2.))
    
    # Prediction intervals for gradient boosted trees
    # Taken from http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
    if str(type(model)) == "<class 'sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'>":
        # Cloning the model so the original version isn't overwritten
        from sklearn.base import clone
        quantile_model = clone(model)
        
        # Calculating buffer for upper/lower alpha to get the Xth percentile
        alpha_buffer = ((1 - x) / 2)
        alpha = percentile + alpha_buffer
        
        # Setting the loss function to quantile before re-fitting
        quantile_model.set_params(loss='quantile')
        
        # Upper prediction interval
        print('Generating upper prediction intervals')
        quantile_model.set_params(alpha=alpha)
        quantile_model.fit(X_train, y_train)
        upper_PI = quantile_model.predict(X)
        
        # Lower prediction interval
        print('Generating lower prediction intervals')
        quantile_model.set_params(alpha=(1 - alpha))
        quantile_model.fit(X_train, y_train)
        lower_PI = quantile_model.predict(X)
    
    # Compiling results of prediction intervals and the actual predictions
    results = pd.DataFrame({'lower_PI': lower_PI,
                            'prediction': predictions,
                            'upper_PI': upper_PI})
    
    return results