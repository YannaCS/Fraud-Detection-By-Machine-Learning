# Advanced hyperparameter optimization using continuous ranges
# Multiple methods: RandomizedSearch, Optuna, and Hyperopt

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import uniform, randint, loguniform
import optuna
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import time
from joblib import Parallel, delayed
import os
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Comprehensive LightGBM suppression
os.environ['LIGHTGBM_VERBOSITY'] = '-1'
os.environ['LIGHTGBM_SILENT'] = '1'
logging.getLogger('lightgbm').setLevel(logging.CRITICAL)

# Suppress all possible warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="lightgbm")

# Additional LightGBM specific suppression
try:
    import lightgbm as lgb
    lgb.set_verbosity(-1)
except:
    pass

# Method 1: RandomizedSearchCV with continuous distributions
# ========================================================
def randomized_search_optimize(model_name, X_train, y_train, n_iter=100):
    """Optimize using RandomizedSearchCV with continuous distributions"""
    # Define continuous parameter distributions
    param_dist = {
        'Random Forest': {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(5, 30),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': uniform(0.1, 0.9),  # Continuous between 0.1 and 1.0
            'class_weight': [{0: 1, 1: w} for w in range(2, 21)]  # Weight from 2 to 20
        },
        
        'Gradient Boosting': {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 10),
            'learning_rate': loguniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': uniform(0.1, 1.0),  # Continuous feature fraction
            'validation_fraction': uniform(0.1, 0.2),  # For early stopping
            'n_iter_no_change': randint(5, 20)  # Early stopping patience
        },
        
        'XGBoost': {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 15),
            'learning_rate': loguniform(0.01, 0.3),  # Log-uniform distribution
            'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
            'colsample_bytree': uniform(0.6, 0.4),  # 0.6 to 1.0
            'scale_pos_weight': uniform(2, 18),  # 2 to 20
            'gamma': loguniform(1e-8, 1.0),
            'reg_alpha': loguniform(1e-8, 1.0),
            'reg_lambda': loguniform(1e-8, 1.0)
        },
        
        'LightGBM': {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 15),
            'learning_rate': loguniform(0.01, 0.3),
            'num_leaves': randint(10, 200),
            'feature_fraction': uniform(0.6, 0.4),  # 0.6 to 1.0
            'bagging_fraction': uniform(0.6, 0.4),  # 0.6 to 1.0
            'scale_pos_weight': uniform(2, 18),  # 2 to 20
            'reg_alpha': loguniform(1e-8, 1.0),
            'reg_lambda': loguniform(1e-8, 1.0),
            'min_child_samples': randint(5, 100),
            'bagging_freq': [1, 5, 10]  # Enable bagging
        },
        
        'Neural Network': {
            'hidden_layer_sizes': [(randint(50, 300).rvs(),), 
                                  (randint(50, 300).rvs(), randint(25, 150).rvs()),
                                  (randint(100, 400).rvs(), randint(50, 200).rvs(), randint(25, 100).rvs())],
            'alpha': loguniform(1e-5, 1e-1),
            'learning_rate_init': loguniform(1e-4, 1e-1),
            'beta_1': uniform(0.8, 0.19),  # 0.8 to 0.99
            'beta_2': uniform(0.9, 0.099)  # 0.9 to 0.999
        },
        
        'Logistic Regression': {
            'C': loguniform(1e-4, 1e2),  # Regularization strength (inverse)
            'penalty': ['l1', 'l2', 'elasticnet'],
            'l1_ratio': uniform(0.0, 1.0),  # For elasticnet penalty
            'solver': ['liblinear', 'saga', 'lbfgs'],
            'max_iter': randint(100, 2000),
            'tol': loguniform(1e-6, 1e-2),
            'class_weight': [{0: 1, 1: w} for w in range(2, 21)]
        },
        
        'Decision Tree': {
            'max_depth': randint(3, 30),
            'min_samples_split': randint(2, 50),
            'min_samples_leaf': randint(1, 20),
            'max_features': uniform(0.1, 0.9),
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'min_impurity_decrease': loguniform(1e-8, 1e-2),
            'ccp_alpha': loguniform(1e-8, 1e-1),  # Pruning parameter
            'class_weight': [{0: 1, 1: w} for w in range(2, 21)]
        }
    }

    print(f"\nStarting RandomizedSearch optimization for {model_name}...")
    start_time = time.time()
    
    # Get base model
    if model_name == 'Random Forest':
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif model_name == 'Gradient Boosting':
        base_model = GradientBoostingClassifier(random_state=42)
    elif model_name == 'XGBoost':
        base_model = XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist')
    elif model_name == 'LightGBM':
        base_model = LGBMClassifier(random_state=42, verbosity=-1, objective='binary', 
                                   force_row_wise=True)  # Suppress warnings
    elif model_name == 'Neural Network':
        base_model = MLPClassifier(random_state=42, max_iter=2000, early_stopping=True)
    elif model_name == 'Logistic Regression':
        base_model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_name == 'Decision Tree':
        base_model = DecisionTreeClassifier(random_state=42)
    
    # Randomized search
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist[model_name],
        n_iter=n_iter,  # Number of parameter settings sampled
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True
    )
    
    random_search.fit(X_train, y_train)
    
    time_taken = time.time() - start_time
    print(f"Completed {model_name} RandomizedSearch in {time_taken:.2f} seconds")
    print(f"Best F1 Score: {random_search.best_score_:.4f}")
    print(f"Best Parameters: {random_search.best_params_}\n")
    
    return model_name, {
        'model': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'time_taken': time_taken
    }


# Method 2: Optuna (Bayesian Optimization)
# ========================================

def optuna_optimize(model_name, X_train, y_train, n_trials=100, verbose='medium'):
    """
    Optimize using Optuna with Bayesian optimization
    
    Parameters:
    -----------
    verbose : str, default='medium'
        - 'quiet': No progress output, only final results
        - 'medium': Progress bar with best score updates only
        - 'full': All trial details and progress bar
    """
    
    def objective(trial):
        # Define parameter ranges for each model
        if model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2, 20),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'random_state': 42,
                'eval_metric': 'logloss',
                'tree_method': 'hist'
            }
            model = XGBClassifier(**params)
            
        elif model_name == 'Gradient Boosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.3),
                'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 20),
                'random_state': 42
            }
            model = GradientBoostingClassifier(**params)
            
        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 200),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2, 20),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'random_state': 42,
                'verbosity': -1,
                'objective': 'binary',
                'force_row_wise': True,
                'bagging_freq': 1,
                'boost_from_average': False,
                'is_unbalance': False,
                'silent': True,  # Additional silence parameter
                'verbose': -1    # Another verbosity parameter
            }
            # Create model with output suppression
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                model = LGBMClassifier(**params)
            
        elif model_name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                'class_weight': {0: 1, 1: trial.suggest_float('pos_weight', 2, 20)},
                'random_state': 42,
                'n_jobs': -1
            }
            model = RandomForestClassifier(**params)
        
        elif model_name == 'Neural Network':
            # Suggest architecture
            n_layers = trial.suggest_int('n_layers', 1, 3)
            if n_layers == 1:
                hidden_layer_sizes = (trial.suggest_int('layer1_size', 50, 300),)
            elif n_layers == 2:
                hidden_layer_sizes = (
                    trial.suggest_int('layer1_size', 50, 300),
                    trial.suggest_int('layer2_size', 25, 150)
                )
            else:  # n_layers == 3
                hidden_layer_sizes = (
                    trial.suggest_int('layer1_size', 100, 400),
                    trial.suggest_int('layer2_size', 50, 200),
                    trial.suggest_int('layer3_size', 25, 100)
                )
            
            params = {
                'hidden_layer_sizes': hidden_layer_sizes,
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
                'beta_1': trial.suggest_float('beta_1', 0.8, 0.99),
                'beta_2': trial.suggest_float('beta_2', 0.9, 0.999),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),  # Removed 'logistic' for speed
                'solver': 'adam',  # Fixed to adam for better performance
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),  # Removed 'auto'
                'max_iter': trial.suggest_int('max_iter', 200, 1000),  # Reduced max iterations
                'early_stopping': True,
                'validation_fraction': 0.1,  # Fixed for consistency
                'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 15),  # Reduced patience
                'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
                'random_state': 42
            }
            model = MLPClassifier(**params)

        elif model_name == 'Logistic Regression':
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
            solver = trial.suggest_categorical('solver', ['liblinear', 'saga', 'lbfgs'])
            
            # Handle solver-penalty compatibility
            if penalty == 'elasticnet' and solver != 'saga':
                solver = 'saga'
            elif penalty == 'l1' and solver == 'lbfgs':
                solver = 'liblinear'
            
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                'penalty': penalty,
                'solver': solver,
                'max_iter': trial.suggest_int('max_iter', 100, 2000),
                'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
                'class_weight': {0: 1, 1: trial.suggest_float('pos_weight', 2, 20)},
                'random_state': 42
            }
            
            # Add l1_ratio only for elasticnet
            if penalty == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
            
            model = LogisticRegression(**params)

        elif model_name == 'Decision Tree':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 1e-8, 1e-2, log=True),
                'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-8, 1e-1, log=True),
                'class_weight': {0: 1, 1: trial.suggest_float('pos_weight', 2, 20)},
                'random_state': 42
            }
            model = DecisionTreeClassifier(**params)

        # Cross-validation with special handling for Neural Network
        if model_name == 'Neural Network':
            # Use fewer CV folds for Neural Network to speed up
            cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced from 5 to 3
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='f1',
                n_jobs=1  # Single job for Neural Network to avoid memory issues
            )
        elif model_name == 'LightGBM':
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='f1',
                    n_jobs=1  # Use single job for LightGBM to avoid output conflicts
                )
        else:
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='f1',
                n_jobs=-1
            )
        
        return cv_scores.mean()
    
    print(f"\nStarting Optuna optimization for {model_name}...")
    start_time = time.time()
    
    # Set logging level to suppress trial details for cleaner output
    optuna.logging.set_verbosity(optuna.logging.ERROR)  # Only show errors
    
    # Create study
    study = optuna.create_study(
        direction='maximize', 
        sampler=optuna.samplers.TPESampler(n_startup_trials=10)
    )
    
    # Custom progress tracking with model names - clean version
    if verbose == 'quiet':
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    else:  # Use progress bar format that updates only on improvement
        import sys
        
        best_value = -float('inf')
        best_trial = 0
        completed_trials = 0
        
        def progress_callback(study, trial):
            nonlocal best_value, best_trial, completed_trials
            completed_trials += 1
            
            # Only update display when best score improves
            if study.best_value > best_value:
                best_value = study.best_value
                best_trial = study.best_trial.number
                
                # Calculate progress
                progress_pct = (completed_trials / n_trials) * 100
                bar_length = 20
                filled_length = int(bar_length * completed_trials // n_trials)
                bar = '█' * filled_length + ' ' * (bar_length - filled_length)
                
                # Calculate timing
                elapsed = time.time() - start_time
                avg_time = elapsed / completed_trials
                remaining = avg_time * (n_trials - completed_trials)
                
                # Clear previous line and print new progress
                print(f"\r{' ' * 100}", end='', flush=True)  # Clear line
                print(f"\r{model_name}: {progress_pct:.0f}%|{bar}| {completed_trials}/{n_trials} "
                      f"[{elapsed:.0f}s<{remaining:.0f}s, {avg_time:.2f}s/it] "
                      f"Best Trial: {best_trial}, Best F1: {best_value:.4f}", 
                      end='', flush=True)
            
            # For Neural Network, show periodic progress even without improvement
            elif model_name == 'Neural Network' and completed_trials % 5 == 0:
                progress_pct = (completed_trials / n_trials) * 100
                bar_length = 20
                filled_length = int(bar_length * completed_trials // n_trials)
                bar = '█' * filled_length + ' ' * (bar_length - filled_length)
                
                elapsed = time.time() - start_time
                avg_time = elapsed / completed_trials
                remaining = avg_time * (n_trials - completed_trials)
                
                print(f"\r{' ' * 100}", end='', flush=True)  # Clear line
                print(f"\r{model_name}: {progress_pct:.0f}%|{bar}| {completed_trials}/{n_trials} "
                      f"[{elapsed:.0f}s<{remaining:.0f}s, {avg_time:.2f}s/it] "
                      f"Best Trial: {best_trial}, Best F1: {best_value:.4f}", 
                      end='', flush=True)
        
        study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])
        
        # Final update to show 100% completion
        print(f"\r{' ' * 100}", end='', flush=True)  # Clear line
        elapsed = time.time() - start_time
        avg_time = elapsed / n_trials
        bar = '█' * 20
        print(f"\r{model_name}: 100%|{bar}| {n_trials}/{n_trials} "
              f"[{elapsed:.0f}s<00:00, {avg_time:.2f}s/it] "
              f"Best Trial: {best_trial}, Best F1: {best_value:.4f}")
        print()  # Add newline for next model
    
    time_taken = time.time() - start_time
    
    print(f"Completed {model_name} Optuna optimization in {time_taken:.2f} seconds")
    print(f"Best F1 Score: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}\n")
    
    # Create final model with best parameters
    if model_name == 'XGBoost':
        best_model = XGBClassifier(**study.best_params)
    elif model_name == 'LightGBM':
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            best_model = LGBMClassifier(**study.best_params, force_row_wise=True)
    elif model_name == 'Random Forest':
        best_model = RandomForestClassifier(**study.best_params)
    elif model_name == 'Neural Network':
        # Handle Neural Network parameters separately due to custom architecture logic
        nn_params = study.best_params.copy()
        
        # Extract architecture parameters
        n_layers = nn_params.pop('n_layers')
        layer1_size = nn_params.pop('layer1_size', 100)
        layer2_size = nn_params.pop('layer2_size', 50)
        layer3_size = nn_params.pop('layer3_size', 25)
        
        # Build architecture based on n_layers
        if n_layers == 1:
            hidden_layer_sizes = (layer1_size,)
        elif n_layers == 2:
            hidden_layer_sizes = (layer1_size, layer2_size)
        else:  # n_layers == 3
            hidden_layer_sizes = (layer1_size, layer2_size, layer3_size)
        
        # Add architecture to parameters
        nn_params['hidden_layer_sizes'] = hidden_layer_sizes
        best_model = MLPClassifier(**nn_params)
    elif model_name == 'Gradient Boosting':
        best_model = GradientBoostingClassifier(**study.best_params)
    elif model_name == 'Logistic Regression':
        best_model = LogisticRegression(**study.best_params)
    elif model_name == 'Decision Tree':
        best_model = DecisionTreeClassifier(**study.best_params)
    
    return model_name, {
        'model': best_model,
        'best_params': study.best_params,
        'best_score': study.best_value,
        'time_taken': time_taken,
        'study': study
    }


# Method 3: Hyperopt (Tree-structured Parzen Estimator)
# =====================================================
def hyperopt_optimize(model_name, X_train, y_train, max_evals=100):
    """Optimize using Hyperopt TPE algorithm"""
    # Define search spaces
    if model_name == 'XGBoost':
        space = {
            'n_estimators': hp.randint('n_estimators', 100, 1000),
            'max_depth': hp.randint('max_depth', 3, 15),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': hp.uniform('scale_pos_weight', 2, 20),
            'gamma': hp.loguniform('gamma', np.log(1e-8), np.log(1.0)),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(1.0)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(1.0))
        }
    elif model_name == 'Gradient Boosting':
        space = {
            'n_estimators': hp.randint('n_estimators', 100, 1000),
            'max_depth': hp.randint('max_depth', 3, 10),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'min_samples_split': hp.randint('min_samples_split', 2, 20),
            'min_samples_leaf': hp.randint('min_samples_leaf', 1, 10),
            'max_features': hp.uniform('max_features', 0.1, 1.0),
            'validation_fraction': hp.uniform('validation_fraction', 0.1, 0.3),
            'n_iter_no_change': hp.randint('n_iter_no_change', 5, 20)
        }
    elif model_name == 'LightGBM':
        space = {
            'n_estimators': hp.randint('n_estimators', 100, 1000),
            'max_depth': hp.randint('max_depth', 3, 15),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'num_leaves': hp.randint('num_leaves', 10, 200),
            'feature_fraction': hp.uniform('feature_fraction', 0.6, 1.0),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 1.0),
            'scale_pos_weight': hp.uniform('scale_pos_weight', 2, 20),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(1.0)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(1.0)),
            'min_child_samples': hp.randint('min_child_samples', 5, 100)
        }
    elif model_name == 'Random Forest':
        space = {
            'n_estimators': hp.randint('n_estimators', 100, 1000),
            'max_depth': hp.randint('max_depth', 5, 30),
            'min_samples_split': hp.randint('min_samples_split', 2, 20),
            'min_samples_leaf': hp.randint('min_samples_leaf', 1, 10),
            'max_features': hp.uniform('max_features', 0.1, 1.0),
            'pos_weight': hp.uniform('pos_weight', 2, 20)  # For class_weight conversion
        }
    elif model_name == 'Neural Network':
        space = {
            'n_layers': hp.choice('n_layers', [1, 2, 3]),
            'layer1_size': hp.randint('layer1_size', 50, 300),
            'layer2_size': hp.randint('layer2_size', 25, 150),
            'layer3_size': hp.randint('layer3_size', 25, 100),
            'alpha': hp.loguniform('alpha', np.log(1e-5), np.log(1e-1)),
            'learning_rate_init': hp.loguniform('learning_rate_init', np.log(1e-4), np.log(1e-1)),
            'beta_1': hp.uniform('beta_1', 0.8, 0.99),
            'beta_2': hp.uniform('beta_2', 0.9, 0.999),
            'activation': hp.choice('activation', ['relu', 'tanh', 'logistic']),
            'solver': hp.choice('solver', ['adam', 'lbfgs']),
            'batch_size': hp.choice('batch_size', ['auto', 32, 64, 128, 256]),
            'max_iter': hp.randint('max_iter', 1000, 5000),
            'validation_fraction': hp.uniform('validation_fraction', 0.1, 0.2),
            'n_iter_no_change': hp.randint('n_iter_no_change', 10, 50)
        }
    elif model_name == 'Logistic Regression':
        space = {
            'C': hp.loguniform('C', np.log(1e-4), np.log(1e2)),
            'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet']),
            'solver': hp.choice('solver', ['liblinear', 'saga', 'lbfgs']),
            'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0),
            'max_iter': hp.randint('max_iter', 100, 2000),
            'tol': hp.loguniform('tol', np.log(1e-6), np.log(1e-2)),
            'pos_weight': hp.uniform('pos_weight', 2, 20)
        }
    elif model_name == 'Decision Tree':
        space = {
            'max_depth': hp.randint('max_depth', 3, 30),
            'min_samples_split': hp.randint('min_samples_split', 2, 50),
            'min_samples_leaf': hp.randint('min_samples_leaf', 1, 20),
            'max_features': hp.uniform('max_features', 0.1, 1.0),
            'criterion': hp.choice('criterion', ['gini', 'entropy']),
            'splitter': hp.choice('splitter', ['best', 'random']),
            'min_impurity_decrease': hp.loguniform('min_impurity_decrease', np.log(1e-8), np.log(1e-2)),
            'ccp_alpha': hp.loguniform('ccp_alpha', np.log(1e-8), np.log(1e-1)),
            'pos_weight': hp.uniform('pos_weight', 2, 20)
        }
    
    def objective(params):
        # Convert hyperopt params to integers where needed
        int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 
                     'num_leaves', 'min_child_samples', 'layer1_size', 'layer2_size', 'layer3_size',
                     'n_iter_no_change', 'max_iter']
        for param in int_params:
            if param in params:
                params[param] = int(params[param])
        
        if model_name == 'XGBoost':
            model = XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                tree_method='hist',
                **params
            )
        elif model_name == 'Gradient Boosting':
            model = GradientBoostingClassifier(
                random_state=42,
                **params
            )
        elif model_name == 'LightGBM':
            model = LGBMClassifier(
                random_state=42,
                verbosity=-1,
                objective='binary',
                force_row_wise=True,  # Remove auto-choosing overhead
                **params
            )
        elif model_name == 'Random Forest':
            # Convert pos_weight to class_weight
            pos_weight = params.pop('pos_weight', 10)
            params['class_weight'] = {0: 1, 1: pos_weight}
            model = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                **params
            )
        elif model_name == 'Neural Network':
            # Handle architecture based on n_layers
            n_layers = params.pop('n_layers')
            if n_layers == 0:  # hp.choice returns index, so 0=1 layer, 1=2 layers, 2=3 layers
                hidden_layer_sizes = (params.pop('layer1_size'),)
            elif n_layers == 1:
                hidden_layer_sizes = (params.pop('layer1_size'), params.pop('layer2_size'))
            else:  # n_layers == 2 (3 layers)
                hidden_layer_sizes = (params.pop('layer1_size'), params.pop('layer2_size'), params.pop('layer3_size'))
            
            # Remove unused layer parameters
            for layer_param in ['layer1_size', 'layer2_size', 'layer3_size']:
                params.pop(layer_param, None)
            
            params['hidden_layer_sizes'] = hidden_layer_sizes
            params['early_stopping'] = True
            
            model = MLPClassifier(
                random_state=42,
                **params
            )
        elif model_name == 'Logistic Regression':
            # Handle solver-penalty compatibility
            penalty_map = ['l1', 'l2', 'elasticnet']
            solver_map = ['liblinear', 'saga', 'lbfgs']
            
            penalty = penalty_map[params.pop('penalty')]
            solver = solver_map[params.pop('solver')]
            pos_weight = params.pop('pos_weight', 10)
            
            # Handle compatibility
            if penalty == 'elasticnet' and solver != 'saga':
                solver = 'saga'
            elif penalty == 'l1' and solver == 'lbfgs':
                solver = 'liblinear'
            
            # Only keep l1_ratio for elasticnet
            if penalty != 'elasticnet':
                params.pop('l1_ratio', None)
            
            params.update({
                'penalty': penalty,
                'solver': solver,
                'class_weight': {0: 1, 1: pos_weight}
            })
            
            model = LogisticRegression(
                random_state=42,
                **params
            )
        elif model_name == 'Decision Tree':
            # Convert categorical parameters
            criterion_map = ['gini', 'entropy']
            splitter_map = ['best', 'random']
            
            criterion = criterion_map[params.pop('criterion')]
            splitter = splitter_map[params.pop('splitter')]
            pos_weight = params.pop('pos_weight', 10)
            
            params.update({
                'criterion': criterion,
                'splitter': splitter,
                'class_weight': {0: 1, 1: pos_weight}
            })
            
            model = DecisionTreeClassifier(
                random_state=42,
                **params
            )
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1
        )
        
        # Hyperopt minimizes, so return negative score
        return {'loss': -cv_scores.mean(), 'status': STATUS_OK}
    
    print(f"\nStarting Hyperopt optimization for {model_name}...")
    start_time = time.time()
    
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=True
    )
    
    time_taken = time.time() - start_time
    
    # Convert best params
    int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 
                 'num_leaves', 'min_child_samples', 'layer1_size', 'layer2_size', 'layer3_size',
                 'n_iter_no_change', 'max_iter']
    for param in int_params:
        if param in best:
            best[param] = int(best[param])
    
    print(f"Completed {model_name} Hyperopt optimization in {time_taken:.2f} seconds")
    print(f"Best F1 Score: {-trials.best_trial['result']['loss']:.4f}")
    print(f"Best Parameters: {best}")
    
    # Create final model with reconstructed parameters
    if model_name == 'XGBoost':
        best_model = XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            tree_method='hist',
            **best
        )
    elif model_name == 'Gradient Boosting':
        best_model = GradientBoostingClassifier(
            random_state=42,
            **best
        )
    elif model_name == 'LightGBM':
        best_model = LGBMClassifier(
            random_state=42,
            verbosity=-1,
            objective='binary',
            **best
        )
    elif model_name == 'Random Forest':
        # Convert pos_weight back to class_weight
        pos_weight = best.pop('pos_weight', 10)
        best['class_weight'] = {0: 1, 1: pos_weight}
        best_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **best
        )
    elif model_name == 'Neural Network':
        # Handle architecture reconstruction
        n_layers = best.pop('n_layers')
        if n_layers == 0:  # 1 layer
            hidden_layer_sizes = (best.pop('layer1_size'),)
        elif n_layers == 1:  # 2 layers
            hidden_layer_sizes = (best.pop('layer1_size'), best.pop('layer2_size'))
        else:  # 3 layers
            hidden_layer_sizes = (best.pop('layer1_size'), best.pop('layer2_size'), best.pop('layer3_size'))
        
        # Remove unused layer parameters
        for layer_param in ['layer1_size', 'layer2_size', 'layer3_size']:
            best.pop(layer_param, None)
        
        best['hidden_layer_sizes'] = hidden_layer_sizes
        best['early_stopping'] = True
        
        best_model = MLPClassifier(
            random_state=42,
            **best
        )
    elif model_name == 'Logistic Regression':
        # Reconstruct categorical parameters
        penalty_map = ['l1', 'l2', 'elasticnet']
        solver_map = ['liblinear', 'saga', 'lbfgs']
        
        penalty = penalty_map[best.pop('penalty')]
        solver = solver_map[best.pop('solver')]
        pos_weight = best.pop('pos_weight', 10)
        
        # Handle compatibility
        if penalty == 'elasticnet' and solver != 'saga':
            solver = 'saga'
        elif penalty == 'l1' and solver == 'lbfgs':
            solver = 'liblinear'
        
        # Only keep l1_ratio for elasticnet
        if penalty != 'elasticnet':
            best.pop('l1_ratio', None)
        
        best.update({
            'penalty': penalty,
            'solver': solver,
            'class_weight': {0: 1, 1: pos_weight}
        })
        
        best_model = LogisticRegression(
            random_state=42,
            **best
        )
    elif model_name == 'Decision Tree':
        # Reconstruct categorical parameters
        criterion_map = ['gini', 'entropy']
        splitter_map = ['best', 'random']
        
        criterion = criterion_map[best.pop('criterion')]
        splitter = splitter_map[best.pop('splitter')]
        pos_weight = best.pop('pos_weight', 10)
        
        best.update({
            'criterion': criterion,
            'splitter': splitter,
            'class_weight': {0: 1, 1: pos_weight}
        })
        
        best_model = DecisionTreeClassifier(
            random_state=42,
            **best
        )
    
    return model_name, {
        'model': best_model,
        'best_params': best,
        'best_score': -trials.best_trial['result']['loss'],
        'time_taken': time_taken,
        'trials': trials
    }


# Parallel optimization across multiple models using your preferred method
def optimize_all_models_parallel(X_train, y_train, method='optuna', n_trials=50, models=None,verbose='medium'):
    """Run optimization for multiple models in parallel"""
    if models is None:
        models = ['XGBoost', 'LightGBM', 'Random Forest', 'Gradient Boosting', 'Neural Network', 'Logistic Regression', 'Decision Tree']
    
    print("Starting parallel optimization...")
    overall_start = time.time()
    
    if method == 'optuna':
        results = Parallel(n_jobs=-1)(
            delayed(optuna_optimize)(model_name, X_train, y_train, n_trials, verbose=verbose)
            for model_name in models
        )
    elif method == 'randomized':
        results = Parallel(n_jobs=-1)(
            delayed(randomized_search_optimize)(model_name, X_train, y_train, n_trials)
            for model_name in models
        )
    elif method == 'hyperopt':
        results = Parallel(n_jobs=-1)(
            delayed(hyperopt_optimize)(model_name, X_train, y_train, max_evals=n_trials)
            for model_name in models
        )

    # Convert results to dictionary
    optimized_models = dict(results)
    
    overall_time = time.time() - overall_start
    print(f"\nTotal optimization time: {overall_time:.2f} seconds")
    print(f"Successfully optimized {len(optimized_models)} models")
    
    return optimized_models

# Usage Examples:
# ===============

# QUIET MODE - Only start/end messages:
# results = optimize_all_models_parallel(X_train, y_train, method='optuna', n_trials=50, verbose='quiet')

# MEDIUM MODE - Default Optuna progress bars for each model (DEFAULT):
# results = optimize_all_models_parallel(X_train, y_train, method='optuna', n_trials=50, verbose='medium')

# FULL MODE - All trial details and progress bars:
# results = optimize_all_models_parallel(X_train, y_train, method='optuna', n_trials=50, verbose='full')

# For better progress display with many models, you can also run models one by one:
# models_to_run = ['XGBoost', 'LightGBM', 'Random Forest']
# results = {}
# for model in models_to_run:
#     result = optuna_optimize(model, X_train, y_train, n_trials=50, verbose='medium')
#     results[model] = result[1]  # result is (model_name, result_dict)