# Advanced hyperparameter optimization with imbalance handling
# Multiple methods: RandomizedSearch, Optuna, and Hyperopt

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
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
def randomized_search_optimize(model_name, X_train, y_train, n_iter=100, handle_imbalance=True):
    """
    Optimize using RandomizedSearchCV with continuous distributions
    
    Parameters:
    -----------
    handle_imbalance : bool, default=True
        Whether to handle class imbalance
    """
    
    # Define continuous parameter distributions
    param_dist = {
        'Random Forest': {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(5, 30),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': uniform(0.1, 0.9),
        },
        
        'Gradient Boosting': {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 10),
            'learning_rate': loguniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': uniform(0.1, 1.0),
            'validation_fraction': uniform(0.1, 0.2),
            'n_iter_no_change': randint(5, 20)
        },
        
        'XGBoost': {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 15),
            'learning_rate': loguniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': loguniform(1e-8, 1.0),
            'reg_alpha': loguniform(1e-8, 1.0),
            'reg_lambda': loguniform(1e-8, 1.0)
        },
        
        'LightGBM': {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 15),
            'learning_rate': loguniform(0.01, 0.3),
            'num_leaves': randint(10, 200),
            'feature_fraction': uniform(0.6, 0.4),
            'bagging_fraction': uniform(0.6, 0.4),
            'reg_alpha': loguniform(1e-8, 1.0),
            'reg_lambda': loguniform(1e-8, 1.0),
            'min_child_samples': randint(5, 100),
            'bagging_freq': [1, 5, 10]
        },
        
        'Neural Network': {
            'hidden_layer_sizes': [(randint(50, 300).rvs(),), 
                                  (randint(50, 300).rvs(), randint(25, 150).rvs()),
                                  (randint(100, 400).rvs(), randint(50, 200).rvs(), randint(25, 100).rvs())],
            'alpha': loguniform(1e-5, 1e-1),
            'learning_rate_init': loguniform(1e-4, 1e-1),
            'beta_1': uniform(0.8, 0.19),
            'beta_2': uniform(0.9, 0.099)
        },
        
        'Logistic Regression': {
            'C': loguniform(1e-4, 1e2),
            'penalty': ['l1', 'l2', 'elasticnet'],
            'l1_ratio': uniform(0.0, 1.0),
            'solver': ['liblinear', 'saga', 'lbfgs'],
            'max_iter': randint(100, 2000),
            'tol': loguniform(1e-6, 1e-2),
        },
        
        'Decision Tree': {
            'max_depth': randint(3, 30),
            'min_samples_split': randint(2, 50),
            'min_samples_leaf': randint(1, 20),
            'max_features': uniform(0.1, 0.9),
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'min_impurity_decrease': loguniform(1e-8, 1e-2),
            'ccp_alpha': loguniform(1e-8, 1e-1),
        }
    }
    
    # If handling imbalance, add class_weight/scale_pos_weight to param distributions
    if handle_imbalance:
        if model_name in ['Random Forest', 'Logistic Regression', 'Decision Tree']:
            param_dist[model_name]['class_weight'] = ['balanced', {0: 1, 1: 2}, {0: 1, 1: 5}, 
                                                      {0: 1, 1: 10}, {0: 1, 1: 15}, {0: 1, 1: 20}]
        elif model_name == 'XGBoost':
            param_dist[model_name]['scale_pos_weight'] = uniform(1, 20)
        elif model_name == 'LightGBM':
            # For LightGBM, we'll use is_unbalance as a categorical choice
            param_dist[model_name]['is_unbalance'] = [True, False]
            param_dist[model_name]['scale_pos_weight'] = uniform(1, 20)

    print(f"\nStarting RandomizedSearch optimization for {model_name}...")
    print(f"Handle imbalance: {handle_imbalance}")
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
                                   force_row_wise=True)
    elif model_name == 'Neural Network':
        base_model = MLPClassifier(random_state=42, max_iter=2000, early_stopping=True)
    elif model_name == 'Logistic Regression':
        base_model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_name == 'Decision Tree':
        base_model = DecisionTreeClassifier(random_state=42)
    
    # Prepare fit parameters for models that need sample weights
    fit_params = {}
    if handle_imbalance and model_name == 'Gradient Boosting':
        sample_weights = compute_sample_weight('balanced', y_train)
        fit_params['sample_weight'] = sample_weights
    
    # Randomized search
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist[model_name],
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True
    )
    
    random_search.fit(X_train, y_train, **fit_params)
    
    time_taken = time.time() - start_time
    print(f"Completed {model_name} RandomizedSearch in {time_taken:.2f} seconds")
    print(f"Best F1 Score: {random_search.best_score_:.4f}")
    print(f"Best Parameters: {random_search.best_params_}\n")
    
    return model_name, {
        'model': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'time_taken': time_taken,
        'handle_imbalance': handle_imbalance
    }


# Method 2: Optuna (Bayesian Optimization)
def optuna_optimize(model_name, X_train, y_train, n_trials=100, verbose='medium', handle_imbalance=True):
    """
    Optimize using Optuna with Bayesian optimization
    
    Parameters:
    -----------
    verbose : str, default='medium'
        - 'quiet': No progress output, only final results
        - 'medium': Progress bar with best score updates only
        - 'full': All trial details and progress bar
    handle_imbalance : bool, default=True
        Whether to handle class imbalance
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
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'random_state': 42,
                'eval_metric': 'logloss',
                'tree_method': 'hist'
            }
            
            # Handle imbalance
            if handle_imbalance:
                params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 1, 20)
            
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
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'random_state': 42,
                'verbosity': -1,
                'objective': 'binary',
                'force_row_wise': True,
                'bagging_freq': 1,
                'boost_from_average': False,
                'silent': True,
                'verbose': -1
            }
            
            # Handle imbalance
            if handle_imbalance:
                # Choose between is_unbalance and scale_pos_weight
                use_is_unbalance = trial.suggest_categorical('use_is_unbalance', [True, False])
                if use_is_unbalance:
                    params['is_unbalance'] = True
                else:
                    params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 1, 20)
            
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                model = LGBMClassifier(**params)
            
        elif model_name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Handle imbalance
            if handle_imbalance:
                class_weight_choice = trial.suggest_categorical('class_weight_type', ['balanced', 'custom'])
                if class_weight_choice == 'balanced':
                    params['class_weight'] = 'balanced'
                else:
                    pos_weight = trial.suggest_float('pos_weight', 2, 20)
                    params['class_weight'] = {0: 1, 1: pos_weight}
            
            model = RandomForestClassifier(**params)
        
        elif model_name == 'Neural Network':
            # Neural Network parameters (no imbalance handling as requested)
            n_layers = trial.suggest_int('n_layers', 1, 3)
            if n_layers == 1:
                hidden_layer_sizes = (trial.suggest_int('layer1_size', 50, 300),)
            elif n_layers == 2:
                hidden_layer_sizes = (
                    trial.suggest_int('layer1_size', 50, 300),
                    trial.suggest_int('layer2_size', 25, 150)
                )
            else:
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
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'solver': 'adam',
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'max_iter': trial.suggest_int('max_iter', 200, 1000),
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 15),
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
                'random_state': 42
            }
            
            # Handle imbalance
            if handle_imbalance:
                class_weight_choice = trial.suggest_categorical('class_weight_type', ['balanced', 'custom'])
                if class_weight_choice == 'balanced':
                    params['class_weight'] = 'balanced'
                else:
                    pos_weight = trial.suggest_float('pos_weight', 2, 20)
                    params['class_weight'] = {0: 1, 1: pos_weight}
            
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
                'random_state': 42
            }
            
            # Handle imbalance
            if handle_imbalance:
                class_weight_choice = trial.suggest_categorical('class_weight_type', ['balanced', 'custom'])
                if class_weight_choice == 'balanced':
                    params['class_weight'] = 'balanced'
                else:
                    pos_weight = trial.suggest_float('pos_weight', 2, 20)
                    params['class_weight'] = {0: 1, 1: pos_weight}
            
            model = DecisionTreeClassifier(**params)

        # Cross-validation with sample weights for Gradient Boosting
        if model_name == 'Gradient Boosting' and handle_imbalance:
            sample_weights = compute_sample_weight('balanced', y_train)
            # Custom cross-validation to handle sample weights
            cv_scores = []
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                sw_tr = sample_weights[train_idx]
                
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_tr, y_tr, sample_weight=sw_tr)
                pred = model_clone.predict(X_val)
                from sklearn.metrics import f1_score
                cv_scores.append(f1_score(y_val, pred))
            cv_scores = np.array(cv_scores)
        elif model_name == 'Neural Network':
            # Use fewer CV folds for Neural Network
            cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='f1',
                n_jobs=1
            )
        elif model_name == 'LightGBM':
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='f1',
                    n_jobs=1
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
    print(f"Handle imbalance: {handle_imbalance}")
    start_time = time.time()
    
    # Set logging level
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    # Create study
    study = optuna.create_study(
        direction='maximize', 
        sampler=optuna.samplers.TPESampler(n_startup_trials=10)
    )
    
    # Progress tracking code
    if verbose == 'quiet':
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    else:
        best_value = -float('inf')
        best_trial = 0
        completed_trials = 0
        
        def progress_callback(study, trial):
            nonlocal best_value, best_trial, completed_trials
            completed_trials += 1
            
            if study.best_value > best_value:
                best_value = study.best_value
                best_trial = study.best_trial.number
                
                progress_pct = (completed_trials / n_trials) * 100
                bar_length = 20
                filled_length = int(bar_length * completed_trials // n_trials)
                bar = '█' * filled_length + ' ' * (bar_length - filled_length)
                
                elapsed = time.time() - start_time
                avg_time = elapsed / completed_trials
                remaining = avg_time * (n_trials - completed_trials)
                
                print(f"\r{' ' * 100}", end='', flush=True)
                print(f"\r{model_name}: {progress_pct:.0f}%|{bar}| {completed_trials}/{n_trials} "
                      f"[{elapsed:.0f}s<{remaining:.0f}s, {avg_time:.2f}s/it] "
                      f"Best Trial: {best_trial}, Best F1: {best_value:.4f}", 
                      end='', flush=True)
            
            elif model_name == 'Neural Network' and completed_trials % 5 == 0:
                progress_pct = (completed_trials / n_trials) * 100
                bar_length = 20
                filled_length = int(bar_length * completed_trials // n_trials)
                bar = '█' * filled_length + ' ' * (bar_length - filled_length)
                
                elapsed = time.time() - start_time
                avg_time = elapsed / completed_trials
                remaining = avg_time * (n_trials - completed_trials)
                
                print(f"\r{' ' * 100}", end='', flush=True)
                print(f"\r{model_name}: {progress_pct:.0f}%|{bar}| {completed_trials}/{n_trials} "
                      f"[{elapsed:.0f}s<{remaining:.0f}s, {avg_time:.2f}s/it] "
                      f"Best Trial: {best_trial}, Best F1: {best_value:.4f}", 
                      end='', flush=True)
        
        study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])
        
        print(f"\r{' ' * 100}", end='', flush=True)
        elapsed = time.time() - start_time
        avg_time = elapsed / n_trials
        bar = '█' * 20
        print(f"\r{model_name}: 100%|{bar}| {n_trials}/{n_trials} "
              f"[{elapsed:.0f}s<00:00, {avg_time:.2f}s/it] "
              f"Best Trial: {best_trial}, Best F1: {best_value:.4f}")
        print()
    
    time_taken = time.time() - start_time
    
    print(f"Completed {model_name} Optuna optimization in {time_taken:.2f} seconds")
    print(f"Best F1 Score: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}\n")
    
    # Create final model with best parameters
    best_params = study.best_params.copy()
    
    # Handle special parameters for different models
    if model_name == 'LightGBM' and 'use_is_unbalance' in best_params:
        best_params.pop('use_is_unbalance')
    elif model_name in ['Random Forest', 'Logistic Regression', 'Decision Tree'] and 'class_weight_type' in best_params:
        best_params.pop('class_weight_type')
        if 'pos_weight' in best_params:
            best_params.pop('pos_weight')
    
    if model_name == 'XGBoost':
        best_model = XGBClassifier(**best_params)
    elif model_name == 'LightGBM':
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            best_model = LGBMClassifier(**best_params, force_row_wise=True)
    elif model_name == 'Random Forest':
        best_model = RandomForestClassifier(**best_params)
    elif model_name == 'Neural Network':
        # Handle Neural Network parameters
        nn_params = best_params.copy()
        n_layers = nn_params.pop('n_layers')
        layer1_size = nn_params.pop('layer1_size', 100)
        layer2_size = nn_params.pop('layer2_size', 50)
        layer3_size = nn_params.pop('layer3_size', 25)
        
        if n_layers == 1:
            hidden_layer_sizes = (layer1_size,)
        elif n_layers == 2:
            hidden_layer_sizes = (layer1_size, layer2_size)
        else:
            hidden_layer_sizes = (layer1_size, layer2_size, layer3_size)
        
        nn_params['hidden_layer_sizes'] = hidden_layer_sizes
        best_model = MLPClassifier(**nn_params)
    elif model_name == 'Gradient Boosting':
        best_model = GradientBoostingClassifier(**best_params)
    elif model_name == 'Logistic Regression':
        best_model = LogisticRegression(**best_params)
    elif model_name == 'Decision Tree':
        best_model = DecisionTreeClassifier(**best_params)
    
    # Fit the final model with sample weights if needed
    if model_name == 'Gradient Boosting' and handle_imbalance:
        sample_weights = compute_sample_weight('balanced', y_train)
        best_model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        best_model.fit(X_train, y_train)
    
    return model_name, {
        'model': best_model,
        'best_params': study.best_params,
        'best_score': study.best_value,
        'time_taken': time_taken,
        'study': study,
        'handle_imbalance': handle_imbalance
    }


# Method 3: Hyperopt (Tree-structured Parzen Estimator)
def hyperopt_optimize(model_name, X_train, y_train, max_evals=100, handle_imbalance=True):
    """
    Optimize using Hyperopt TPE algorithm
    
    Parameters:
    -----------
    handle_imbalance : bool, default=True
        Whether to handle class imbalance
    """
    
    # Define search spaces
    if model_name == 'XGBoost':
        space = {
            'n_estimators': hp.randint('n_estimators', 100, 1000),
            'max_depth': hp.randint('max_depth', 3, 15),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'gamma': hp.loguniform('gamma', np.log(1e-8), np.log(1.0)),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(1.0)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(1.0))
        }
        if handle_imbalance:
            space['scale_pos_weight'] = hp.uniform('scale_pos_weight', 1, 20)
            
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
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(1.0)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(1.0)),
            'min_child_samples': hp.randint('min_child_samples', 5, 100)
        }
        if handle_imbalance:
            space['use_is_unbalance'] = hp.choice('use_is_unbalance', [True, False])
            space['scale_pos_weight'] = hp.uniform('scale_pos_weight', 1, 20)
            
    elif model_name == 'Random Forest':
        space = {
            'n_estimators': hp.randint('n_estimators', 100, 1000),
            'max_depth': hp.randint('max_depth', 5, 30),
            'min_samples_split': hp.randint('min_samples_split', 2, 20),
            'min_samples_leaf': hp.randint('min_samples_leaf', 1, 10),
            'max_features': hp.uniform('max_features', 0.1, 1.0),
        }
        if handle_imbalance:
            space['class_weight_type'] = hp.choice('class_weight_type', ['balanced', 'custom'])
            space['pos_weight'] = hp.uniform('pos_weight', 2, 20)
            
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
        }
        if handle_imbalance:
            space['class_weight_type'] = hp.choice('class_weight_type', ['balanced', 'custom'])
            space['pos_weight'] = hp.uniform('pos_weight', 2, 20)
            
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
        }
        if handle_imbalance:
            space['class_weight_type'] = hp.choice('class_weight_type', ['balanced', 'custom'])
            space['pos_weight'] = hp.uniform('pos_weight', 2, 20)
    
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
            # Handle imbalance choice
            if handle_imbalance and 'use_is_unbalance' in params:
                use_is_unbalance = params.pop('use_is_unbalance')
                if use_is_unbalance == 0:  # hp.choice returns index
                    params['is_unbalance'] = True
                    params.pop('scale_pos_weight', None)
                else:
                    params.pop('is_unbalance', None)
                    # scale_pos_weight already in params
            
            model = LGBMClassifier(
                random_state=42,
                verbosity=-1,
                objective='binary',
                force_row_wise=True,
                **params
            )
        elif model_name == 'Random Forest':
            if handle_imbalance and 'class_weight_type' in params:
                class_weight_type = params.pop('class_weight_type')
                if class_weight_type == 0:  # 'balanced'
                    params['class_weight'] = 'balanced'
                    params.pop('pos_weight', None)
                else:  # 'custom'
                    pos_weight = params.pop('pos_weight', 10)
                    params['class_weight'] = {0: 1, 1: pos_weight}
            
            model = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                **params
            )
        elif model_name == 'Neural Network':
            n_layers = params.pop('n_layers')
            if n_layers == 0:
                hidden_layer_sizes = (params.pop('layer1_size'),)
            elif n_layers == 1:
                hidden_layer_sizes = (params.pop('layer1_size'), params.pop('layer2_size'))
            else:
                hidden_layer_sizes = (params.pop('layer1_size'), params.pop('layer2_size'), params.pop('layer3_size'))
            
            for layer_param in ['layer1_size', 'layer2_size', 'layer3_size']:
                params.pop(layer_param, None)
            
            params['hidden_layer_sizes'] = hidden_layer_sizes
            params['early_stopping'] = True
            
            model = MLPClassifier(
                random_state=42,
                **params
            )
        elif model_name == 'Logistic Regression':
            penalty_map = ['l1', 'l2', 'elasticnet']
            solver_map = ['liblinear', 'saga', 'lbfgs']
            
            penalty = penalty_map[params.pop('penalty')]
            solver = solver_map[params.pop('solver')]
            
            if penalty == 'elasticnet' and solver != 'saga':
                solver = 'saga'
            elif penalty == 'l1' and solver == 'lbfgs':
                solver = 'liblinear'
            
            if penalty != 'elasticnet':
                params.pop('l1_ratio', None)
            
            if handle_imbalance and 'class_weight_type' in params:
                class_weight_type = params.pop('class_weight_type')
                if class_weight_type == 0:  # 'balanced'
                    params['class_weight'] = 'balanced'
                    params.pop('pos_weight', None)
                else:  # 'custom'
                    pos_weight = params.pop('pos_weight', 10)
                    params['class_weight'] = {0: 1, 1: pos_weight}
            
            params.update({
                'penalty': penalty,
                'solver': solver
            })
            
            model = LogisticRegression(
                random_state=42,
                **params
            )
        elif model_name == 'Decision Tree':
            criterion_map = ['gini', 'entropy']
            splitter_map = ['best', 'random']
            
            criterion = criterion_map[params.pop('criterion')]
            splitter = splitter_map[params.pop('splitter')]
            
            if handle_imbalance and 'class_weight_type' in params:
                class_weight_type = params.pop('class_weight_type')
                if class_weight_type == 0:  # 'balanced'
                    params['class_weight'] = 'balanced'
                    params.pop('pos_weight', None)
                else:  # 'custom'
                    pos_weight = params.pop('pos_weight', 10)
                    params['class_weight'] = {0: 1, 1: pos_weight}
            
            params.update({
                'criterion': criterion,
                'splitter': splitter
            })
            
            model = DecisionTreeClassifier(
                random_state=42,
                **params
            )
        
        # Cross-validation with sample weights for Gradient Boosting
        if model_name == 'Gradient Boosting' and handle_imbalance:
            sample_weights = compute_sample_weight('balanced', y_train)
            cv_scores = []
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                sw_tr = sample_weights[train_idx]
                
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_tr, y_tr, sample_weight=sw_tr)
                pred = model_clone.predict(X_val)
                from sklearn.metrics import f1_score
                cv_scores.append(f1_score(y_val, pred))
            cv_scores = np.array(cv_scores)
        else:
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='f1',
                n_jobs=-1
            )
        
        return {'loss': -cv_scores.mean(), 'status': STATUS_OK}
    
    print(f"\nStarting Hyperopt optimization for {model_name}...")
    print(f"Handle imbalance: {handle_imbalance}")
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
    final_params = best.copy()
    
    if model_name == 'XGBoost':
        best_model = XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            tree_method='hist',
            **final_params
        )
    elif model_name == 'Gradient Boosting':
        best_model = GradientBoostingClassifier(
            random_state=42,
            **final_params
        )
    elif model_name == 'LightGBM':
        if handle_imbalance and 'use_is_unbalance' in final_params:
            use_is_unbalance = final_params.pop('use_is_unbalance')
            if use_is_unbalance == 0:
                final_params['is_unbalance'] = True
                final_params.pop('scale_pos_weight', None)
        
        best_model = LGBMClassifier(
            random_state=42,
            verbosity=-1,
            objective='binary',
            **final_params
        )
    elif model_name == 'Random Forest':
        if handle_imbalance and 'class_weight_type' in final_params:
            class_weight_type = final_params.pop('class_weight_type')
            if class_weight_type == 0:
                final_params['class_weight'] = 'balanced'
                final_params.pop('pos_weight', None)
            else:
                pos_weight = final_params.pop('pos_weight', 10)
                final_params['class_weight'] = {0: 1, 1: pos_weight}
        
        best_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **final_params
        )
    elif model_name == 'Neural Network':
        n_layers = final_params.pop('n_layers')
        if n_layers == 0:
            hidden_layer_sizes = (final_params.pop('layer1_size'),)
        elif n_layers == 1:
            hidden_layer_sizes = (final_params.pop('layer1_size'), final_params.pop('layer2_size'))
        else:
            hidden_layer_sizes = (final_params.pop('layer1_size'), final_params.pop('layer2_size'), final_params.pop('layer3_size'))
        
        for layer_param in ['layer1_size', 'layer2_size', 'layer3_size']:
            final_params.pop(layer_param, None)
        
        final_params['hidden_layer_sizes'] = hidden_layer_sizes
        final_params['early_stopping'] = True
        
        best_model = MLPClassifier(
            random_state=42,
            **final_params
        )
    elif model_name == 'Logistic Regression':
        penalty_map = ['l1', 'l2', 'elasticnet']
        solver_map = ['liblinear', 'saga', 'lbfgs']
        
        penalty = penalty_map[final_params.pop('penalty')]
        solver = solver_map[final_params.pop('solver')]
        
        if handle_imbalance and 'class_weight_type' in final_params:
            class_weight_type = final_params.pop('class_weight_type')
            if class_weight_type == 0:
                final_params['class_weight'] = 'balanced'
                final_params.pop('pos_weight', None)
            else:
                pos_weight = final_params.pop('pos_weight', 10)
                final_params['class_weight'] = {0: 1, 1: pos_weight}
        
        if penalty == 'elasticnet' and solver != 'saga':
            solver = 'saga'
        elif penalty == 'l1' and solver == 'lbfgs':
            solver = 'liblinear'
        
        if penalty != 'elasticnet':
            final_params.pop('l1_ratio', None)
        
        final_params.update({
            'penalty': penalty,
            'solver': solver
        })
        
        best_model = LogisticRegression(
            random_state=42,
            **final_params
        )
    elif model_name == 'Decision Tree':
        criterion_map = ['gini', 'entropy']
        splitter_map = ['best', 'random']
        
        criterion = criterion_map[final_params.pop('criterion')]
        splitter = splitter_map[final_params.pop('splitter')]
        
        if handle_imbalance and 'class_weight_type' in final_params:
            class_weight_type = final_params.pop('class_weight_type')
            if class_weight_type == 0:
                final_params['class_weight'] = 'balanced'
                final_params.pop('pos_weight', None)
            else:
                pos_weight = final_params.pop('pos_weight', 10)
                final_params['class_weight'] = {0: 1, 1: pos_weight}
        
        final_params.update({
            'criterion': criterion,
            'splitter': splitter
        })
        
        best_model = DecisionTreeClassifier(
            random_state=42,
            **final_params
        )
    
    # Fit the final model with sample weights if needed
    if model_name == 'Gradient Boosting' and handle_imbalance:
        sample_weights = compute_sample_weight('balanced', y_train)
        best_model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        best_model.fit(X_train, y_train)
    
    return model_name, {
        'model': best_model,
        'best_params': best,
        'best_score': -trials.best_trial['result']['loss'],
        'time_taken': time_taken,
        'trials': trials,
        'handle_imbalance': handle_imbalance
    }


# Parallel optimization across multiple models
def optimize_all_models_parallel(X_train, y_train, method='optuna', n_trials=50, 
                               models=None, verbose='medium', handle_imbalance=True):
    """
    Run optimization for multiple models in parallel
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    method : str, default='optuna'
        Optimization method: 'optuna', 'randomized', or 'hyperopt'
    n_trials : int, default=50
        Number of trials/iterations for optimization
    models : list, optional
        List of model names to optimize. If None, uses all available models
    verbose : str, default='medium'
        Verbosity level for Optuna
    handle_imbalance : bool, default=True
        Whether to handle class imbalance
    """
    
    if models is None:
        models = ['XGBoost', 'LightGBM', 'Random Forest', 'Gradient Boosting', 
                 'Neural Network', 'Logistic Regression', 'Decision Tree']
    
    print("Starting parallel optimization...")
    print(f"Optimization method: {method}")
    print(f"Handle imbalance: {handle_imbalance}")
    print(f"Models to optimize: {models}")
    overall_start = time.time()
    
    if method == 'optuna':
        results = Parallel(n_jobs=-1)(
            delayed(optuna_optimize)(
                model_name, X_train, y_train, n_trials, 
                verbose=verbose,
                handle_imbalance=handle_imbalance
            )
            for model_name in models
        )
    elif method == 'randomized':
        results = Parallel(n_jobs=-1)(
            delayed(randomized_search_optimize)(
                model_name, X_train, y_train, n_trials,
                handle_imbalance=handle_imbalance
            )
            for model_name in models
        )
    elif method == 'hyperopt':
        results = Parallel(n_jobs=-1)(
            delayed(hyperopt_optimize)(
                model_name, X_train, y_train, 
                max_evals=n_trials,
                handle_imbalance=handle_imbalance
            )
            for model_name in models
        )

    # Convert results to dictionary
    optimized_models = dict(results)
    
    overall_time = time.time() - overall_start
    print(f"\nTotal optimization time: {overall_time:.2f} seconds")
    print(f"Successfully optimized {len(optimized_models)} models")
    
    # Print summary of results
    print("\nOptimization Summary:")
    print("-" * 50)
    for model_name, result in optimized_models.items():
        print(f"{model_name}: Best F1 = {result['best_score']:.4f}, "
              f"Time = {result['time_taken']:.2f}s")
    
    return optimized_models


# Usage Examples:
# ===============

# Example 1: Basic usage with imbalance handling (default)
# results = optimize_all_models_parallel(X_train, y_train, method='optuna', n_trials=50)

# Example 2: Without handling imbalance
# results = optimize_all_models_parallel(
#     X_train, y_train, 
#     method='optuna', 
#     n_trials=50, 
#     handle_imbalance=False
# )

# Example 3: Optimize specific models only
# models_to_optimize = ['XGBoost', 'LightGBM', 'Random Forest']
# results = optimize_all_models_parallel(
#     X_train, y_train, 
#     method='optuna', 
#     n_trials=100,
#     models=models_to_optimize,
#     handle_imbalance=True
# )

# Example 4: Using different optimization methods
# # Optuna (Bayesian optimization)
# optuna_results = optimize_all_models_parallel(X_train, y_train, method='optuna', n_trials=50)
# 
# # RandomizedSearch
# random_results = optimize_all_models_parallel(X_train, y_train, method='randomized', n_trials=50)
# 
# # Hyperopt
# hyperopt_results = optimize_all_models_parallel(X_train, y_train, method='hyperopt', n_trials=50)

# Example 5: Sequential optimization with progress monitoring
# models_to_run = ['XGBoost', 'LightGBM', 'Random Forest']
# results = {}
# for model in models_to_run:
#     model_name, result = optuna_optimize(
#         model, X_train, y_train, 
#         n_trials=50, 
#         verbose='medium',
#         handle_imbalance=True
#     )
#     results[model_name] = result

# Example 6: Compare with and without imbalance handling
# # With imbalance handling
# balanced_results = optimize_all_models_parallel(
#     X_train, y_train, 
#     method='optuna', 
#     n_trials=50,
#     handle_imbalance=True
# )
# 
# # Without imbalance handling
# unbalanced_results = optimize_all_models_parallel(
#     X_train, y_train, 
#     method='optuna', 
#     n_trials=50,
#     handle_imbalance=False
# )
# 
# # Compare results
# print("\nComparison of F1 scores:")
# print("-" * 60)
# print(f"{'Model':<20} {'With Balance':>15} {'Without Balance':>15} {'Difference':>10}")
# print("-" * 60)
# for model in balanced_results:
#     bal_score = balanced_results[model]['best_score']
#     unbal_score = unbalanced_results[model]['best_score']
#     diff = bal_score - unbal_score
#     print(f"{model:<20} {bal_score:>15.4f} {unbal_score:>15.4f} {diff:>10.4f}")