import warnings
from sklearn.linear_model import Lasso, Ridge, ElasticNet, MultiTaskLasso, MultiTaskElasticNet, SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")# 
params = {
    "randomForest": {
    'min_samples_split': [20, 30],  # Reduced to 2 values (instead of 3)
    'n_estimators': [50, 100],      # Reduced to 2 values, balancing speed and performance
    'max_features': ['sqrt', 0.33],  # Reduced feature subset options
    'max_depth': [5, 10],            # Reduced to 2 values (smaller depth to prevent overfitting)
    'min_samples_leaf': [2, 5],      # Reduced to 2 values (avoids overfitting)
    'n_jobs': [-1]                   # Continue using all available cores
},


	"extraTrees":{
        'n_estimators':[10,20,30,50,100,200],
        'max_features': ['sqrt', 0.25, 0.33],
        'max_depth':[None,10 ,25], 
        'n_jobs':[-1]
    },
	"rbfSVM":{'gamma': [1e-3, 1e-4],
	                     'C': [1, 10, 100, 1000]
			  },
	"linearSVM":{'C': [1, 10, 100, 1000]
				 },
    'polySVM': {
        'C': [0.1, 1, 10],  # Regularization parameter
        'degree': [2, 3, 4],  # Polynomial degree
        'epsilon': [0.1, 0.2],  # Epsilon-tube within which no penalty is given
    },
	"ridge":{'alpha':[0.001],'fit_intercept':[True,False],'solver':['svd'],'tol':[0.0001]
			 },
	"dummy":{'strategy':['mean','median']
			 },
	"LinearRegression":{'fit_intercept':[True,False],'normalize':[True,False]
						},	"NNGarroteRegression":{},"KernelRegression":{},
	"KernelRidge":{'alpha':[1,10]
				   },
	"AdaBoost":{'n_estimators':[50,100,200],'loss':['linear','square'],'learning_rate':[0.05,0.5,1.0,2.0]},
    "Bagging": {
    'estimator__min_samples_leaf': [1, 2, 5],  # Set parameters for the base estimator (e.g., DecisionTreeRegressor)
    'estimator__min_samples_split': [10, 20, 30],  # Base estimator parameter
    'n_estimators': [100, 200, 300],  # BaggingRegressor parameters
    'max_samples': [0.4, 0.6, 0.8],   # BaggingRegressor parameters
    'max_features': [0.3, 0.5, 0.75],  # BaggingRegressor parameters
    'bootstrap': [True],              # BaggingRegressor parameters
    'bootstrap_features': [False, True],
    'oob_score': [True],
    'n_jobs': [-1],
    'random_state': [42]
},

    
	'SGDRegression': {'penalty':['l1','l2','elasticnet',None],'l1_ratio':[0.01,0.10,0.20,0.80]
					  },
	"KNeighborsRegression":{'n_neighbors':[2,5,10],'weights':['uniform','distance'],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
							},
	"MultiLasso":{'alpha':[0.01,0.1,1.0,10.0]
				  },
	"lasso":{'alpha':[0.01,0.1,1.0,10.0]
			 },
	"DecisionTree": {
    'max_depth': [3, 5,10],  # Limit the depth to avoid overly complex trees
    'min_samples_split': [5,10, 20],  # Higher values ensure nodes have more samples before splitting
    # 'min_samples_leaf': [5, 10],  # Prevent nodes with very few samples
    'max_features': ['sqrt'],  # Limit the features considered for splitting to control variance
    'criterion': ['friedman_mse'],  # Keep the same criterion for simplicity
    'splitter': ['best'],  # Use 'best' to avoid randomness
    # 'max_leaf_nodes': [10, 20,30]  # Limit the number of leaf nodes to reduce complexity
},
	"MultiElasticNet":{'alpha':[0.5,1,2],'l1_ratio':[0,0.5,1.0],'normalize':[True,False],'warm_start':[True,False]
	},
	
     
    "xgboost": {
    'n_estimators': [100, 200],  # Number of boosting rounds (trees)
    'learning_rate': [0.01, 0.1],  # Step size for weight updates
    'max_depth': [3, 5],  # Maximum depth of trees (controls model complexity)
    'min_child_weight': [1, 3],  # Minimum sum of instance weight needed in a child node
    'subsample': [0.8],  # Fraction of samples used per tree
    'colsample_bytree': [0.8],  # Fraction of features used per tree
    'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required to make a split
    'reg_alpha': [0, 0.1, 1],  # L1 regularization (Lasso)
    'reg_lambda': [1, 1.5],  # L2 regularization (Ridge)
    'scale_pos_weight': [1, 2],  # Balance of positive and negative weights (if dealing with imbalanced data)
    'booster': ['gbtree'],  # Type of booster: trees only
    'objective': ['reg:squarederror'],  # Loss function for regression
    'early_stopping_rounds': [10],  # Early stopping rounds
},

   "gradientBoost": {
    'n_estimators': [100, 200],                   # Number of boosting stages
    'learning_rate': [0.01, 0.05],                # Lower learning rate with fewer trees
    'max_depth': [2, 3, 5],                       # Shallower trees to reduce overfitting
    'min_samples_split': [10, 20],                # Minimum samples to split a node
    'min_samples_leaf': [3, 5],                   # Minimum samples required in a leaf node
    'subsample': [0.6, 0.8],                      # Use a fraction of samples for each tree
    'max_features': ['sqrt', 'log2'],             # Limit features considered for splitting
},

	"ElasticNet": {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.5, 0.9],  # L1/L2 regularization mixing parameter
    'fit_intercept': [True, False],
},
    "catboost": {
    'iterations': [500, 1000],
    'learning_rate': [0.01, 0.1],
    'depth': [4, 6],
    'l2_leaf_reg': [3, 5],
    'subsample': [0.8, 1],  # You can use subsample with these bootstrap types
    'bootstrap_type': ['Bernoulli', 'MVS'],  # Supported bootstrap types with subsampling
    'random_seed': [42],
},
}


def getEstimator(regressor):
    if "lasso" in regressor or "Lasso" in regressor:
        estimator = Lasso(alpha=0.1)
    elif "MultiLasso" in regressor:
        estimator = MultiTaskLasso()
    elif "ridge" in regressor or "Ridge" in regressor:
        estimator = Ridge()
    elif "SGDRegression" in regressor:
        estimator = SGDRegressor()
    elif "KernelRegression" in regressor:
        estimator = KernelRidge(kernel='rbf')
    elif "LinearRegression" in regressor:
        estimator = LinearRegression()
    elif "KNeighborsRegression" in regressor:
        estimator = KNeighborsRegressor()
    elif "randomForest" in regressor or "RandomForest" in regressor:
        estimator = RandomForestRegressor()  # No need for BaggingRegressor
    elif "extraTrees" in regressor or "ExtraTrees" in regressor:
        estimator = ExtraTreesRegressor()  # No need for BaggingRegressor
    elif "rbfSVM" in regressor or "RBFSVM" in regressor:
        estimator = SVR(kernel="rbf")
    elif "linearSVM" in regressor or "LinearSVM" in regressor:
        estimator = SVR(kernel="linear")
    elif "polySVM" in regressor or "PolySVM" in regressor:
        estimator = SVR(kernel="poly")
    elif "ElasticNet" in regressor:
        estimator = ElasticNet()
    elif "MultiElasticNet" in regressor:
        estimator = MultiTaskElasticNet()
    elif "gradientBoost" in regressor:
        estimator = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    elif "AdaBoost" in regressor:
        estimator = AdaBoostRegressor()
    elif "Bagging" in regressor:
        # Use BaggingRegressor for custom base estimators
        base_estimator = DecisionTreeRegressor()  # Set a base model, if needed
        estimator = BaggingRegressor(estimator=base_estimator, max_samples=0.4, max_features=0.3)
    elif "DecisionTree" in regressor:
        estimator = DecisionTreeRegressor()
    elif "dummy" in regressor:
        estimator = DummyRegressor()
    elif "xgboost" in regressor:
        estimator = XGBRegressor()
    elif "catboost" in regressor:
        estimator = CatBoostRegressor()
    else:
        raise ValueError(f"Unknown regressor: {regressor}")

    return estimator



def crossVal_overfitting_check(x_features, y_target, estimator, CV=5):
    if isinstance(x_features, pd.Series):
        x_features = x_features.values.reshape(-1, 1)
    kf = KFold(n_splits=CV, shuffle=True, random_state=42)
    
    train_scores_r2 = []
    test_scores_r2 = []
    train_scores_rmse=[]
    test_scores_rmse=[]
    
    # Create a pipeline that scales only the features (x_features) and applies the model (estimator)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Step 1: Scale only features
        ('model', estimator)           # Step 2: Apply the model (e.g., RandomForest, XGBRegressor)
    ])
    
    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(x_features):
        # Split features (x_features) and target (y_target) for training and test sets
        x_train, x_test = x_features.iloc[train_index], x_features.iloc[test_index]
        y_train, y_test = y_target.iloc[train_index], y_target.iloc[test_index]

        # Train the model (with scaling applied only to x_train)
        pipeline.fit(x_train, y_train)
        
        # Predict on both training and test sets
        y_train_pred = pipeline.predict(x_train)
        y_test_pred = pipeline.predict(x_test)
        
        # Calculate R² scores
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        # Calculate RMSE scores
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Append scores
        train_scores_r2.append(train_r2)
        test_scores_r2.append(test_r2)
        train_scores_rmse.append(train_rmse)
        test_scores_rmse.append(test_rmse)
    # Compute the average R² scores across all folds
    avg_train_score_r2 = np.mean(train_scores_r2)
    avg_test_score_r2 = np.mean(test_scores_r2)
    avg_train_RMSE = np.mean(train_scores_rmse)
    avg_test_RMSE = np.mean(test_scores_rmse)
    return avg_test_score_r2, avg_train_score_r2,avg_test_RMSE,avg_train_RMSE


from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from catboost import CatBoostRegressor

def crossVal_xgboost_catboost_with_scaling(x_features, y_target, estimator, CV=5):
    if isinstance(x_features, pd.Series):
        x_features = x_features.values.reshape(-1, 1)
    kf = KFold(n_splits=CV, shuffle=True, random_state=42)
    
    train_scores_r2 = []
    test_scores_r2 = []
    train_scores_rmse = []
    test_scores_rmse = []

    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(x_features):
        # Split into train and test sets
        x_train, x_test = x_features.iloc[train_index], x_features.iloc[test_index]
        y_train, y_test = y_target.iloc[train_index], y_target.iloc[test_index]
        
        # Further split the training data into train and validation for early stopping
        if len(x_train) > 1:  # Check if there's enough data to split
            x_train_main, x_val, y_train_main, y_val = train_test_split(
                x_train, y_train, test_size=0.2, random_state=42
            )
        else:
            raise ValueError("Training data is too small for further splitting.")
        
        # Scale the features
        scaler = StandardScaler()
        
        x_train_main_scaled = scaler.fit_transform(x_train_main)
        x_val_scaled = scaler.transform(x_val)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        
        # Fit the estimator directly with early stopping
        if isinstance(estimator, (XGBRegressor, CatBoostRegressor)):
            eval_set = [(x_val_scaled, y_val)]
            if isinstance(estimator, XGBRegressor):
                estimator.fit(
                    x_train_main_scaled, y_train_main,
                    eval_set=eval_set,
                    verbose=False
                )
            elif isinstance(estimator, CatBoostRegressor):
                estimator.fit(
                    x_train_main_scaled, y_train_main,
                    eval_set=eval_set,
                    verbose=False
                )
        else:
            raise ValueError("Estimator must be XGBRegressor or CatBoostRegressor")
        
        # Predict on both training and test sets
        y_train_pred = estimator.predict(x_train_scaled)
        y_test_pred = estimator.predict(x_test_scaled)
        
        # Calculate R² scores
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Calculate RMSE scores
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Append scores
        train_scores_r2.append(train_r2)
        test_scores_r2.append(test_r2)
        train_scores_rmse.append(train_rmse)
        test_scores_rmse.append(test_rmse)
    
    # Compute the average R² and RMSE scores across all folds
    avg_train_score = np.mean(train_scores_r2)
    avg_test_score = np.mean(test_scores_r2)
    avg_train_RMSE = np.mean(train_scores_rmse)
    avg_test_RMSE = np.mean(test_scores_rmse)

    return avg_test_score, avg_train_score, avg_test_RMSE, avg_train_RMSE

from sklearn.model_selection import ParameterGrid
import numpy as np

def runGrid(algorithm, x_features, y_target, maximum=0.7, metric='r2'):
    best_params = None  # Track the best hyperparameter combination
    if algorithm not in params:
        raise KeyError(f"Algorithm '{algorithm}' not found in params")
    # Iterate over parameter grid
    for g in ParameterGrid(params[algorithm]):
        estimator = getEstimator(algorithm)
        # Set parameters
        estimator.set_params(**g)

        # Perform cross-validation and calculate the selected metric
        if metric == 'r2':
            if algorithm == 'catboost' or algorithm == 'xgboost':
                score, r2_train,_,_ = crossVal_xgboost_catboost_with_scaling(x_features, y_target, estimator)
            else:
                score, r2_train,_,_= crossVal_overfitting_check(x_features, y_target, estimator, 5)  # Perform cross-validation for R²            
        elif metric == 'rmse':
            if algorithm == 'catboost' or algorithm == 'xgboost':
                _,_,score, rmse_train = crossVal_xgboost_catboost_with_scaling(x_features, y_target, estimator)
            else:
                _,_,score, rmse_train= crossVal_overfitting_check(x_features, y_target, estimator, 5)
            
        # Compare the score based on the selected metric
        if metric == 'r2' and score > maximum:
            print("Best R² Score:", score)
            print("Overfitting parameters - Training R²:", r2_train)
            maximum = score
            best_train_r2=r2_train
            best_params = g  # Store the best hyperparameter combination
        elif metric == 'rmse' and score < maximum:  # For RMSE, lower is better
            print("Best RMSE Score:", score)
            maximum = score
            best_train_rmse = rmse_train
            best_params = g  # Store the best hyperparameter combination

    # After the loop, set the estimator to the best parameter combination
    best_estimator = None
    if best_params is not None:
        best_estimator = getEstimator(algorithm)
        best_estimator.set_params(**best_params)
        if metric == 'r2':
            print("Best R² Score Achieved:", maximum)
            print("Traning R2 is: ",best_train_r2)
            
        elif metric == 'rmse':
            print("Best RMSE Score Achieved:", maximum)
            print("Traning RMSE is: ",best_train_rmse)
        
    else:
        print(f"No better {metric} score was found.")
    return best_estimator