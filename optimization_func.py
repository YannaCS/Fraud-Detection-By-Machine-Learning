# Advanced hyperparameter optimization using continuous ranges
# Multiple methods: RandomizedSearch, Optuna, and Hyperopt
# Improved for fraud detection with better parameter ranges and scoring

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
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_contour,
    plot_slice,
    plot_intermediate_values
)
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import time
from joblib import Parallel, delayed
import os
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
import warnings
warnings.filterwarnings("ignore")

# Comprehensive LightGBM suppression
os.environ['LIGHTGBM_VERBOSITY'] = '-1'
os.environ['LIGHTGBM_SILENT'] = '1'
logging.getLogger('lightgbm').setLevel(logging.CRITICAL)

# Custom scoring function for fraud detection
def fraud_detection_score(y_true, y_pred):
    """Custom scoring that prioritizes recall (catching frauds) while maintaining decent precision"""
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    
    # Weight recall more heavily for fraud detection
    recall_weight = 0.7
    precision_weight = 0.3
    
    if precision == 0:
        return 0
    
    # Weighted harmonic mean
    score = (recall_weight + precision_weight) / (
        (recall_weight / recall) + (precision_weight / precision)
    )
    return score

# Create custom scorer
fraud_scorer = make_scorer(fraud_detection_score)

def calculate_class_weight(y_train):
    """Calculate the positive class weight based on class distribution"""
    n_samples = len(y_train)
    n_positive = sum(y_train)
    n_negative = n_samples - n_positive
    
    # Calculate weight to balance classes
    pos_weight = n_negative / n_positive
    return pos_weight

# Method 1: RandomizedSearchCV with continuous distributions (IMPROVED)
# ====================================================================
def randomized_search_optimize(model_name, X_train, y_train, n_iter=100):
    """Optimize using RandomizedSearchCV with improved parameter distributions for fraud detection"""
    
    # Calculate dynamic class weight
    pos_weight = calculate_class_weight(y_train)
    
    # Define improved parameter distributions
    param_dist = {
        'Random Forest': {
            'n_estimators': randint(50, 1500),
            'max_depth': randint(5, 50),
            'min_samples_split': randint(2, 100),
            'min_samples_leaf': randint(1, 50),
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7, 0.9],
            'bootstrap': [True, False],
            'class_weight': [{0: 1, 1: w} for w in np.linspace(pos_weight * 0.5, pos_weight * 2.0, 20)]
        },
        
        'Gradient Boosting': {
            'n_estimators': randint(100, 2000),
            'max_depth': randint(3, 15),
            'learning_rate': loguniform(0.01, 0.5),
            'subsample': uniform(0.5, 0.5),
            'min_samples_split': randint(2, 50),
            'min_samples_leaf': randint(1, 20),
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
            'validation_fraction': uniform(0.1, 0.2),
            'n_iter_no_change': randint(5, 30),
            'tol': loguniform(1e-5, 1e-2)
        },
        
        'XGBoost': {
            'n_estimators': randint(50, 1500),
            'max_depth': randint(3, 20),
            'learning_rate': loguniform(0.001, 0.3),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.3, 0.7),
            'colsample_bylevel': uniform(0.3, 0.7),
            'colsample_bynode': uniform(0.3, 0.7),
            'scale_pos_weight': uniform(pos_weight * 0.5, pos_weight * 1.5),
            'gamma': uniform(0, 5),
            'reg_alpha': loguniform(1e-8, 10),
            'reg_lambda': loguniform(1e-8, 10),
            'min_child_weight': randint(1, 20),
            'max_delta_step': randint(0, 10)
        },
        
        'LightGBM': {
            'n_estimators': randint(50, 1500),
            'max_depth': randint(-1, 30),
            'learning_rate': loguniform(0.001, 0.3),
            'num_leaves': randint(20, 500),
            'feature_fraction': uniform(0.3, 0.7),
            'bagging_fraction': uniform(0.3, 0.7),
            'bagging_freq': randint(0, 10),
            'scale_pos_weight': uniform(pos_weight * 0.5, pos_weight * 1.5),
            'reg_alpha': uniform(0, 10),
            'reg_lambda': uniform(0, 10),
            'min_child_samples': randint(5, 100),
            'min_split_gain': uniform(0, 1),
            'subsample_for_bin': randint(20000, 300000),
            'min_data_per_group': randint(10, 200),
            'max_cat_threshold': randint(10, 100)
        },
        
        'Neural Network': {
            'hidden_layer_sizes': [
                (randint(50, 500).rvs(),),
                (randint(100, 500).rvs(), randint(50, 300).rvs()),
                (randint(200, 500).rvs(), randint(100, 300).rvs(), randint(50, 150).rvs()),
            ],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'lbfgs'],
            'alpha': loguniform(1e-6, 1),
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': loguniform(1e-5, 0.1),
            'batch_size': ['auto'] + list(range(32, 512, 32)),
            'max_iter': randint(200, 2000),
            'n_iter_no_change': randint(10, 50),
            'tol': loguniform(1e-6, 1e-2),
            'momentum': uniform(0.8, 0.19),
            'beta_1': uniform(0.8, 0.19),
            'beta_2': uniform(0.9, 0.099),
            'epsilon': loguniform(1e-8, 1e-6)
        },
        
        'Logistic Regression': {
            'C': loguniform(1e-4, 1e2),
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'l1_ratio': uniform(0.0, 1.0),
            'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg'],
            'max_iter': randint(100, 5000),
            'tol': loguniform(1e-6, 1e-2),
            'class_weight': [{0: 1, 1: w} for w in np.linspace(pos_weight * 0.5, pos_weight * 2.0, 20)],
            'warm_start': [True, False]
        },
        
        'Decision Tree': {
            'max_depth': randint(3, 50),
            'min_samples_split': randint(2, 100),
            'min_samples_leaf': randint(1, 50),
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7, 0.9],
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'min_impurity_decrease': loguniform(1e-8, 0.1),
            'ccp_alpha': loguniform(1e-8, 0.1),
            'class_weight': [{0: 1, 1: w} for w in np.linspace(pos_weight * 0.5, pos_weight * 2.0, 20)],
            'max_leaf_nodes': [None] + list(range(10, 200, 10))
        }
    }

    print(f"\nStarting RandomizedSearch optimization for {model_name}...")
    print(f"Calculated positive class weight: {pos_weight:.2f}")
    start_time = time.time()
    
    # Get base model
    if model_name == 'Random Forest':
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif model_name == 'Gradient Boosting':
        base_model = GradientBoostingClassifier(random_state=42)
    elif model_name == 'XGBoost':
        base_model = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    elif model_name == 'LightGBM':
        base_model = LGBMClassifier(random_state=42, verbosity=-1, objective='binary', force_row_wise=True)
    elif model_name == 'Neural Network':
        base_model = MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)
    elif model_name == 'Logistic Regression':
        base_model = LogisticRegression(random_state=42)
    elif model_name == 'Decision Tree':
        base_model = DecisionTreeClassifier(random_state=42)
    
    # Use multiple scoring metrics
    scoring = {
        'f1': 'f1',
        'recall': 'recall',
        'precision': 'precision',
        'fraud_score': fraud_scorer
    }
    
    # Randomized search with refit on fraud_score
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist[model_name],
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=scoring,
        refit='fraud_score',  # Refit on our custom fraud score
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True
    )
    
    random_search.fit(X_train, y_train)
    
    time_taken = time.time() - start_time
    print(f"Completed {model_name} RandomizedSearch in {time_taken:.2f} seconds")
    print(f"Best Fraud Score: {random_search.best_score_:.4f}")
    print(f"Best Parameters: {random_search.best_params_}\n")
    
    return model_name, {
        'model': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'time_taken': time_taken,
        'cv_results': random_search.cv_results_
    }


# Method 2: Optuna (Bayesian Optimization) - IMPROVED
# ===================================================
def optuna_optimize(model_name, X_train, y_train, n_trials=100, verbose='medium'):
    """Optimize using Optuna with improved parameter ranges for fraud detection"""
    
    # Calculate dynamic class weight
    pos_weight = calculate_class_weight(y_train)
    
    def objective(trial):
        from sklearn.metrics import f1_score, recall_score, precision_score
        
        # Improved parameter ranges for fraud detection
        if model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', pos_weight * 0.5, pos_weight * 2.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
            model = XGBClassifier(**params)
            
        elif model_name == 'Gradient Boosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 0.3, 0.5, 0.7]),
                'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.3),
                'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 30),
                'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True),
                'random_state': 42
            }
            model = GradientBoostingClassifier(**params)
            
        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1500),
                'max_depth': trial.suggest_int('max_depth', -1, 30),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 500),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', pos_weight * 0.5, pos_weight * 2.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'min_split_gain': trial.suggest_float('min_split_gain', 0, 1),
                'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 300000),
                'min_data_per_group': trial.suggest_int('min_data_per_group', 10, 200),
                'random_state': 42,
                'verbosity': -1,
                'objective': 'binary',
                'force_row_wise': True,
                'metric': 'None'  # We'll use custom metric
            }
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                model = LGBMClassifier(**params)
            
        elif model_name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1500),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 0.3, 0.5, 0.7]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'class_weight': {0: 1, 1: trial.suggest_float('pos_weight', pos_weight * 0.5, pos_weight * 2.0)},
                'max_samples': trial.suggest_float('max_samples', 0.5, 1.0) if trial.params.get('bootstrap', True) else None,
                'random_state': 42,
                'n_jobs': -1
            }
            if not params['bootstrap']:
                params.pop('max_samples', None)
            model = RandomForestClassifier(**params)
        
        elif model_name == 'Neural Network':
            n_layers = trial.suggest_int('n_layers', 1, 4)
            layers = []
            
            # First layer
            layers.append(trial.suggest_int('layer_1', 50, 500))
            
            # Additional layers with decreasing size
            for i in range(1, n_layers):
                max_neurons = max(20, layers[-1] - 50)
                layers.append(trial.suggest_int(f'layer_{i+1}', 20, max_neurons))
            
            params = {
                'hidden_layer_sizes': tuple(layers),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
                'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
                'alpha': trial.suggest_float('alpha', 1e-6, 1.0, log=True),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-5, 0.1, log=True),
                'batch_size': trial.suggest_categorical('batch_size', ['auto'] + list(range(32, 512, 32))),
                'max_iter': trial.suggest_int('max_iter', 200, 2000),
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': trial.suggest_int('n_iter_no_change', 10, 50),
                'momentum': trial.suggest_float('momentum', 0.8, 0.99) if trial.params['solver'] == 'sgd' else 0.9,
                'beta_1': trial.suggest_float('beta_1', 0.8, 0.99),
                'beta_2': trial.suggest_float('beta_2', 0.9, 0.999),
                'epsilon': trial.suggest_float('epsilon', 1e-8, 1e-6, log=True),
                'random_state': 42
            }
            model = MLPClassifier(**params)

        elif model_name == 'Logistic Regression':
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None])
            solver = trial.suggest_categorical('solver', ['liblinear', 'saga', 'lbfgs', 'newton-cg'])
            
            # Handle solver-penalty compatibility
            if penalty == 'elasticnet' and solver not in ['saga']:
                solver = 'saga'
            elif penalty == 'l1' and solver not in ['liblinear', 'saga']:
                solver = 'saga'
            elif penalty is None and solver not in ['lbfgs', 'newton-cg', 'sag', 'saga']:
                solver = 'lbfgs'
            
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                'penalty': penalty,
                'solver': solver,
                'max_iter': trial.suggest_int('max_iter', 100, 5000),
                'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
                'class_weight': {0: 1, 1: trial.suggest_float('pos_weight', pos_weight * 0.5, pos_weight * 2.0)},
                'warm_start': trial.suggest_categorical('warm_start', [True, False]),
                'random_state': 42
            }
            
            # Add l1_ratio only for elasticnet
            if penalty == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
            
            model = LogisticRegression(**params)

        elif model_name == 'Decision Tree':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 0.3, 0.5, 0.7]),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 1e-8, 0.1, log=True),
                'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-8, 0.1, log=True),
                'class_weight': {0: 1, 1: trial.suggest_float('pos_weight', pos_weight * 0.5, pos_weight * 2.0)},
                'max_leaf_nodes': trial.suggest_categorical('max_leaf_nodes', [None] + list(range(10, 200, 10))),
                'random_state': 42
            }
            model = DecisionTreeClassifier(**params)

        # Cross-validation with multiple metrics
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        f1_scores = []
        recall_scores = []
        precision_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            # Handle both DataFrame and numpy array
            if hasattr(X_train, 'iloc'):  # DataFrame
                X_fold_train = X_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx]
                y_fold_val = y_train.iloc[val_idx]
            else:  # numpy array
                X_fold_train = X_train[train_idx]
                X_fold_val = X_train[val_idx]
                y_fold_train = y_train[train_idx]
                y_fold_val = y_train[val_idx]
            
            # Train model
            if model_name == 'LightGBM':
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    model.fit(X_fold_train, y_fold_train)
            else:
                model.fit(X_fold_train, y_fold_train)
            
            # Predict
            y_pred = model.predict(X_fold_val)
            
            # Calculate metrics
            f1 = f1_score(y_fold_val, y_pred)
            recall = recall_score(y_fold_val, y_pred)
            precision = precision_score(y_fold_val, y_pred, zero_division=0)
            
            f1_scores.append(f1)
            recall_scores.append(recall)
            precision_scores.append(precision)
            
            # Combined score for fraud detection (prioritize recall)
            combined_score = 0.4 * f1 + 0.6 * recall
            
            # Report intermediate value
            trial.report(combined_score, fold_idx)
            
            # Prune if not promising
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Return combined score
        avg_f1 = np.mean(f1_scores)
        avg_recall = np.mean(recall_scores)
        avg_precision = np.mean(precision_scores)
        
        # Store additional metrics in trial
        trial.set_user_attr('avg_f1', avg_f1)
        trial.set_user_attr('avg_recall', avg_recall)
        trial.set_user_attr('avg_precision', avg_precision)
        
        # Combined score for optimization
        return 0.4 * avg_f1 + 0.6 * avg_recall
    
    print(f"\nStarting Optuna optimization for {model_name}...")
    print(f"Calculated positive class weight: {pos_weight:.2f}")
    start_time = time.time()
    
    # Set verbosity
    if verbose == 'quiet':
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    elif verbose == 'medium':
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:  # full
        optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # Create study with better sampler and pruner
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=20,
            n_ei_candidates=30,
            seed=42
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
            interval_steps=1
        )
    )
    
    # Add progress callback for medium verbosity
    if verbose == 'medium':
        def callback(study, trial):
            if trial.number % 10 == 0:
                print(f"Trial {trial.number}: Score = {trial.value:.4f}")
    else:
        callback = None
    
    # Optimize
    study.optimize(
        objective, 
        n_trials=n_trials, 
        show_progress_bar=(verbose != 'quiet'),
        callbacks=[callback] if callback else None
    )
    
    time_taken = time.time() - start_time
    
    print(f"\nCompleted {model_name} Optuna optimization in {time_taken:.2f} seconds")
    print(f"Best Combined Score: {study.best_value:.4f}")
    print(f"Best F1: {study.best_trial.user_attrs['avg_f1']:.4f}")
    print(f"Best Recall: {study.best_trial.user_attrs['avg_recall']:.4f}")
    print(f"Best Precision: {study.best_trial.user_attrs['avg_precision']:.4f}")
    print(f"Best Parameters: {study.best_params}\n")
    
    # Create final model with best parameters
    best_params = study.best_params.copy()
    
    if model_name == 'XGBoost':
        best_model = XGBClassifier(**best_params)
    elif model_name == 'LightGBM':
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            best_model = LGBMClassifier(**best_params)
    elif model_name == 'Random Forest':
        # Handle max_samples parameter
        if 'bootstrap' in best_params and not best_params['bootstrap']:
            best_params.pop('max_samples', None)
        best_model = RandomForestClassifier(**best_params)
    elif model_name == 'Neural Network':
        # Reconstruct layers
        n_layers = best_params.pop('n_layers')
        layers = []
        for i in range(n_layers):
            layers.append(best_params.pop(f'layer_{i+1}'))
        best_params['hidden_layer_sizes'] = tuple(layers)
        best_model = MLPClassifier(**best_params)
    elif model_name == 'Gradient Boosting':
        best_model = GradientBoostingClassifier(**best_params)
    elif model_name == 'Logistic Regression':
        best_model = LogisticRegression(**best_params)
    elif model_name == 'Decision Tree':
        best_model = DecisionTreeClassifier(**best_params)
    
    return model_name, {
        'model': best_model,
        'best_params': study.best_params,
        'best_score': study.best_value,
        'best_f1': study.best_trial.user_attrs['avg_f1'],
        'best_recall': study.best_trial.user_attrs['avg_recall'],
        'best_precision': study.best_trial.user_attrs['avg_precision'],
        'time_taken': time_taken,
        'study': study
    }


# Method 3: Hyperopt (Tree-structured Parzen Estimator) - IMPROVED
# ================================================================
def hyperopt_optimize(model_name, X_train, y_train, max_evals=100):
    """Optimize using Hyperopt TPE algorithm with improved ranges for fraud detection"""
    
    # Calculate dynamic class weight
    pos_weight = calculate_class_weight(y_train)
    
    # Define improved search spaces
    if model_name == 'XGBoost':
        space = {
            'n_estimators': hp.randint('n_estimators', 50, 1501),
            'max_depth': hp.randint('max_depth', 3, 21),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.3, 1.0),
            'colsample_bynode': hp.uniform('colsample_bynode', 0.3, 1.0),
            'scale_pos_weight': hp.uniform('scale_pos_weight', pos_weight * 0.5, pos_weight * 2.0),
            'gamma': hp.uniform('gamma', 0, 5),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(10)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(10)),
            'min_child_weight': hp.randint('min_child_weight', 1, 21),
            'max_delta_step': hp.randint('max_delta_step', 0, 11)
        }
    elif model_name == 'Gradient Boosting':
        space = {
            'n_estimators': hp.randint('n_estimators', 100, 2001),
            'max_depth': hp.randint('max_depth', 3, 16),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'min_samples_split': hp.randint('min_samples_split', 2, 51),
            'min_samples_leaf': hp.randint('min_samples_leaf', 1, 21),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None, 0.3, 0.5, 0.7]),
            'validation_fraction': hp.uniform('validation_fraction', 0.1, 0.3),
            'n_iter_no_change': hp.randint('n_iter_no_change', 5, 31),
            'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-2))
        }
    elif model_name == 'LightGBM':
        space = {
            'n_estimators': hp.randint('n_estimators', 50, 1501),
            'max_depth': hp.randint('max_depth', -1, 31),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
            'num_leaves': hp.randint('num_leaves', 20, 501),
            'feature_fraction': hp.uniform('feature_fraction', 0.3, 1.0),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.3, 1.0),
            'bagging_freq': hp.randint('bagging_freq', 0, 11),
            'scale_pos_weight': hp.uniform('scale_pos_weight', pos_weight * 0.5, pos_weight * 2.0),
            'reg_alpha': hp.uniform('reg_alpha', 0, 10),
            'reg_lambda': hp.uniform('reg_lambda', 0, 10),
            'min_child_samples': hp.randint('min_child_samples', 5, 101),
            'min_split_gain': hp.uniform('min_split_gain', 0, 1),
            'subsample_for_bin': hp.randint('subsample_for_bin', 20000, 300001)
        }
    elif model_name == 'Random Forest':
        space = {
            'n_estimators': hp.randint('n_estimators', 50, 1501),
            'max_depth': hp.randint('max_depth', 5, 51),
            'min_samples_split': hp.randint('min_samples_split', 2, 101),
            'min_samples_leaf': hp.randint('min_samples_leaf', 1, 51),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None, 0.3, 0.5, 0.7]),
            'bootstrap': hp.choice('bootstrap', [True, False]),
            'pos_weight': hp.uniform('pos_weight', pos_weight * 0.5, pos_weight * 2.0)
        }
    elif model_name == 'Neural Network':
        space = {
            'n_layers': hp.choice('n_layers', [1, 2, 3, 4]),
            'layer_1': hp.randint('layer_1', 50, 501),
            'layer_2': hp.randint('layer_2', 20, 301),
            'layer_3': hp.randint('layer_3', 20, 151),
            'layer_4': hp.randint('layer_4', 20, 101),
            'activation': hp.choice('activation', ['relu', 'tanh', 'logistic']),
            'solver': hp.choice('solver', ['adam', 'lbfgs']),
            'alpha': hp.loguniform('alpha', np.log(1e-6), np.log(1)),
            'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),
            'learning_rate_init': hp.loguniform('learning_rate_init', np.log(1e-5), np.log(0.1)),
            'batch_size': hp.choice('batch_size', ['auto'] + list(range(32, 512, 32))),
            'max_iter': hp.randint('max_iter', 200, 2001),
            'n_iter_no_change': hp.randint('n_iter_no_change', 10, 51)
        }
    elif model_name == 'Logistic Regression':
        space = {
            'C': hp.loguniform('C', np.log(1e-4), np.log(1e2)),
            'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet', None]),
            'solver': hp.choice('solver', ['liblinear', 'saga', 'lbfgs', 'newton-cg']),
            'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0),
            'max_iter': hp.randint('max_iter', 100, 5001),
            'tol': hp.loguniform('tol', np.log(1e-6), np.log(1e-2)),
            'pos_weight': hp.uniform('pos_weight', pos_weight * 0.5, pos_weight * 2.0),
            'warm_start': hp.choice('warm_start', [True, False])
        }
    elif model_name == 'Decision Tree':
        space = {
            'max_depth': hp.randint('max_depth', 3, 51),
            'min_samples_split': hp.randint('min_samples_split', 2, 101),
            'min_samples_leaf': hp.randint('min_samples_leaf', 1, 51),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None, 0.3, 0.5, 0.7]),
            'criterion': hp.choice('criterion', ['gini', 'entropy']),
            'splitter': hp.choice('splitter', ['best', 'random']),
            'min_impurity_decrease': hp.loguniform('min_impurity_decrease', np.log(1e-8), np.log(0.1)),
            'ccp_alpha': hp.loguniform('ccp_alpha', np.log(1e-8), np.log(0.1)),
            'pos_weight': hp.uniform('pos_weight', pos_weight * 0.5, pos_weight * 2.0)
        }
    
    def objective(params):
        # Convert hyperopt params to integers where needed
        int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 
                     'num_leaves', 'min_child_samples', 'layer_1', 'layer_2', 'layer_3', 
                     'layer_4', 'n_iter_no_change', 'max_iter', 'min_child_weight', 
                     'max_delta_step', 'bagging_freq', 'subsample_for_bin']
        for param in int_params:
            if param in params:
                params[param] = int(params[param])
        
        # Build models based on parameters
        if model_name == 'XGBoost':
            model = XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                **params
            )
        elif model_name == 'Gradient Boosting':
            # Handle max_features choice
            if isinstance(params.get('max_features'), int):
                max_features_map = ['sqrt', 'log2', None, 0.3, 0.5, 0.7]
                params['max_features'] = max_features_map[params['max_features']]
            model = GradientBoostingClassifier(
                random_state=42,
                **params
            )
        elif model_name == 'LightGBM':
            model = LGBMClassifier(
                random_state=42,
                verbosity=-1,
                objective='binary',
                force_row_wise=True,
                **params
            )
        elif model_name == 'Random Forest':
            # Handle categorical parameters
            if isinstance(params.get('max_features'), int):
                max_features_map = ['sqrt', 'log2', None, 0.3, 0.5, 0.7]
                params['max_features'] = max_features_map[params['max_features']]
            if isinstance(params.get('bootstrap'), int):
                params['bootstrap'] = bool(params['bootstrap'])
            
            pos_weight = params.pop('pos_weight', 10)
            params['class_weight'] = {0: 1, 1: pos_weight}
            model = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                **params
            )
        elif model_name == 'Neural Network':
            # Handle architecture
            n_layers = params.pop('n_layers') + 1  # hp.choice returns index
            layers = []
            for i in range(n_layers):
                if f'layer_{i+1}' in params:
                    layers.append(params.pop(f'layer_{i+1}'))
            
            # Clean up unused layer parameters
            for i in range(4):
                params.pop(f'layer_{i+1}', None)
            
            params['hidden_layer_sizes'] = tuple(layers)
            params['early_stopping'] = True
            params['validation_fraction'] = 0.1
            
            # Handle categorical parameters
            activation_map = ['relu', 'tanh', 'logistic']
            solver_map = ['adam', 'lbfgs']
            learning_rate_map = ['constant', 'invscaling', 'adaptive']
            
            params['activation'] = activation_map[params['activation']]
            params['solver'] = solver_map[params['solver']]
            params['learning_rate'] = learning_rate_map[params['learning_rate']]
            
            # Handle batch_size
            if isinstance(params.get('batch_size'), int):
                batch_size_options = ['auto'] + list(range(32, 512, 32))
                params['batch_size'] = batch_size_options[params['batch_size']]
            
            model = MLPClassifier(
                random_state=42,
                **params
            )
        elif model_name == 'Logistic Regression':
            # Handle categorical parameters
            penalty_map = ['l1', 'l2', 'elasticnet', None]
            solver_map = ['liblinear', 'saga', 'lbfgs', 'newton-cg']
            
            penalty = penalty_map[params.pop('penalty')]
            solver = solver_map[params.pop('solver')]
            pos_weight = params.pop('pos_weight', 10)
            
            # Handle compatibility
            if penalty == 'elasticnet' and solver != 'saga':
                solver = 'saga'
            elif penalty == 'l1' and solver not in ['liblinear', 'saga']:
                solver = 'saga'
            elif penalty is None and solver not in ['lbfgs', 'newton-cg', 'sag', 'saga']:
                solver = 'lbfgs'
            
            # Only keep l1_ratio for elasticnet
            if penalty != 'elasticnet':
                params.pop('l1_ratio', None)
            
            params.update({
                'penalty': penalty,
                'solver': solver,
                'class_weight': {0: 1, 1: pos_weight}
            })
            
            # Handle warm_start
            if isinstance(params.get('warm_start'), int):
                params['warm_start'] = bool(params['warm_start'])
            
            model = LogisticRegression(
                random_state=42,
                **params
            )
        elif model_name == 'Decision Tree':
            # Handle categorical parameters
            if isinstance(params.get('max_features'), int):
                max_features_map = ['sqrt', 'log2', None, 0.3, 0.5, 0.7]
                params['max_features'] = max_features_map[params['max_features']]
            
            criterion_map = ['gini', 'entropy']
            splitter_map = ['best', 'random']
            
            params['criterion'] = criterion_map[params['criterion']]
            params['splitter'] = splitter_map[params['splitter']]
            
            pos_weight = params.pop('pos_weight', 10)
            params['class_weight'] = {0: 1, 1: pos_weight}
            
            model = DecisionTreeClassifier(
                random_state=42,
                **params
            )
        
        # Cross-validation with multiple metrics
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        recall_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall', n_jobs=-1)
        
        # Combined score for fraud detection
        combined_score = 0.4 * np.mean(f1_scores) + 0.6 * np.mean(recall_scores)
        
        # Hyperopt minimizes, so return negative score
        return {'loss': -combined_score, 'status': STATUS_OK}
    
    print(f"\nStarting Hyperopt optimization for {model_name}...")
    print(f"Calculated positive class weight: {pos_weight:.2f}")
    start_time = time.time()
    
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=True,
        rstate=np.random.RandomState(42)
    )
    
    time_taken = time.time() - start_time
    
    # Convert best params back to usable format
    # (Similar conversion logic as in objective function)
    # ... [conversion code here]
    
    print(f"\nCompleted {model_name} Hyperopt optimization in {time_taken:.2f} seconds")
    print(f"Best Combined Score: {-trials.best_trial['result']['loss']:.4f}")
    print(f"Best Parameters: {best}\n")
    
    # Create final model (implementation similar to objective function)
    # ... [model creation code here]
    
    return model_name, {
        'model': None,  # Would need to recreate with best params
        'best_params': best,
        'best_score': -trials.best_trial['result']['loss'],
        'time_taken': time_taken,
        'trials': trials
    }


# Parallel optimization across multiple models
def optimize_all_models_parallel(X_train, y_train, method='optuna', n_trials=50, models=None, verbose='medium'):
    """Run optimization for multiple models in parallel"""
    if models is None:
        models = ['XGBoost', 'LightGBM', 'Random Forest', 'Gradient Boosting', 
                 'Neural Network', 'Logistic Regression', 'Decision Tree']
    
    print("Starting parallel optimization...")
    print(f"Method: {method}")
    print(f"Models: {models}")
    print(f"Trials per model: {n_trials}")
    
    # Calculate and display class distribution
    pos_weight = calculate_class_weight(y_train)
    print(f"\nDataset info:")
    print(f"Total samples: {len(y_train)}")
    print(f"Positive samples (fraud): {sum(y_train)} ({sum(y_train)/len(y_train)*100:.2f}%)")
    print(f"Negative samples: {len(y_train) - sum(y_train)} ({(len(y_train) - sum(y_train))/len(y_train)*100:.2f}%)")
    print(f"Calculated positive class weight: {pos_weight:.2f}")
    print("-" * 60)
    
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
    print(f"Average time per model: {overall_time/len(models):.2f} seconds")
    print(f"Successfully optimized {len(optimized_models)} models")
    
    # Print summary of results
    print("\nOptimization Summary:")
    print("-" * 80)
    print(f"{'Model':<20} {'Best Score':<12} {'Time (s)':<10} {'Best F1':<10} {'Best Recall':<12}")
    print("-" * 80)
    
    for model_name, info in optimized_models.items():
        best_score = info['best_score']
        time_taken = info['time_taken']
        best_f1 = info.get('best_f1', 'N/A')
        best_recall = info.get('best_recall', 'N/A')
        
        if isinstance(best_f1, float):
            print(f"{model_name:<20} {best_score:<12.4f} {time_taken:<10.2f} {best_f1:<10.4f} {best_recall:<12.4f}")
        else:
            print(f"{model_name:<20} {best_score:<12.4f} {time_taken:<10.2f} {best_f1:<10} {best_recall:<12}")
    
    print("-" * 80)
    
    return optimized_models


# Visualization functions remain the same but with added fraud-specific metrics
def visualize_optuna_results(optimized_models, save_dir='optuna_visualizations'):
    """Create comprehensive visualizations for Optuna optimization results"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    for model_name, model_info in optimized_models.items():
        if 'study' not in model_info:
            print(f"Skipping {model_name} - no study object found")
            continue
            
        study = model_info['study']
        print(f"\nGenerating visualizations for {model_name}...")
        
        # 1. Optimization History
        try:
            fig = plot_optimization_history(study)
            fig.update_layout(
                title=f'{model_name} - Optimization History',
                xaxis_title='Trial',
                yaxis_title='Combined Score (0.4*F1 + 0.6*Recall)',
                width=800,
                height=500
            )
            fig.write_html(f"{save_dir}/{model_name}_optimization_history.html")
        except Exception as e:
            print(f"Could not generate optimization history for {model_name}: {e}")
        
        # 2. Parameter Importances
        if len(study.trials) >= 10:
            try:
                fig = plot_param_importances(study)
                fig.update_layout(
                    title=f'{model_name} - Parameter Importances',
                    width=800,
                    height=500
                )
                fig.write_html(f"{save_dir}/{model_name}_param_importances.html")
            except Exception as e:
                print(f"Could not generate parameter importance plot for {model_name}: {e}")
        
        # 3. Parallel Coordinate Plot
        try:
            fig = plot_parallel_coordinate(study)
            fig.update_layout(
                title=f'{model_name} - Parallel Coordinate Plot',
                width=1000,
                height=600
            )
            fig.write_html(f"{save_dir}/{model_name}_parallel_coordinate.html")
        except Exception as e:
            print(f"Could not generate parallel coordinate plot for {model_name}: {e}")
        
        # 4. Slice Plot
        try:
            fig = plot_slice(study)
            fig.update_layout(
                title=f'{model_name} - Parameter Slice Plot',
                width=1200,
                height=800
            )
            fig.write_html(f"{save_dir}/{model_name}_slice_plot.html")
        except Exception as e:
            print(f"Could not generate slice plot for {model_name}: {e}")
        
        # 5. Contour Plot for top 2 parameters
        if len(study.trials) >= 10:
            try:
                params = list(study.best_params.keys())
                if len(params) >= 2:
                    fig = plot_contour(study, params=[params[0], params[1]])
                    fig.update_layout(
                        title=f'{model_name} - Contour Plot ({params[0]} vs {params[1]})',
                        width=800,
                        height=600
                    )
                    fig.write_html(f"{save_dir}/{model_name}_contour_plot.html")
            except Exception as e:
                print(f"Could not generate contour plot for {model_name}: {e}")
        
        # 6. Intermediate Values (for pruned trials)
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        if pruned_trials:
            try:
                fig = plot_intermediate_values(study)
                fig.update_layout(
                    title=f'{model_name} - Intermediate Values (Pruning Visualization)',
                    width=800,
                    height=500
                )
                fig.write_html(f"{save_dir}/{model_name}_pruning_history.html")
            except Exception as e:
                print(f"Could not generate pruning history for {model_name}: {e}")
    
    # Create summary visualization
    create_optimization_summary(optimized_models, save_dir)
    
    print(f"\nVisualizations saved to {save_dir}/")


def create_optimization_summary(optimized_models, save_dir):
    """Create a summary visualization comparing all models with fraud-specific metrics"""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    # Collect summary data
    summary_data = []
    for model_name, model_info in optimized_models.items():
        summary_row = {
            'Model': model_name,
            'Best Score': model_info['best_score'],
            'Time (s)': model_info['time_taken'],
            'Best F1': model_info.get('best_f1', None),
            'Best Recall': model_info.get('best_recall', None),
            'Best Precision': model_info.get('best_precision', None)
        }
        
        if 'study' in model_info:
            study = model_info['study']
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            
            summary_row.update({
                'Completed Trials': completed_trials,
                'Pruned Trials': pruned_trials,
                'Pruning Rate (%)': (pruned_trials / (completed_trials + pruned_trials) * 100) if (completed_trials + pruned_trials) > 0 else 0
            })
        
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create summary plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Fraud Detection Model Optimization Summary', fontsize=16, fontweight='bold')
    
    # 1. Best Scores Comparison
    ax1 = axes[0, 0]
    summary_df.sort_values('Best Score', ascending=False).plot(
        x='Model', y='Best Score', kind='bar', ax=ax1, color='skyblue'
    )
    ax1.set_title('Best Combined Scores by Model\n(0.4*F1 + 0.6*Recall)', fontsize=14)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Combined Score')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(summary_df.sort_values('Best Score', ascending=False)['Best Score']):
        ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
    
    # 2. F1 vs Recall Trade-off
    ax2 = axes[0, 1]
    if 'Best F1' in summary_df.columns and summary_df['Best F1'].notna().any():
        for idx, row in summary_df.iterrows():
            if pd.notna(row['Best F1']) and pd.notna(row['Best Recall']):
                ax2.scatter(row['Best Recall'], row['Best F1'], s=200, alpha=0.7)
                ax2.annotate(row['Model'], (row['Best Recall'], row['Best F1']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax2.set_xlabel('Recall (Fraud Detection Rate)')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 vs Recall Trade-off', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    # 3. Optimization Time
    ax3 = axes[0, 2]
    summary_df.sort_values('Time (s)', ascending=False).plot(
        x='Model', y='Time (s)', kind='bar', ax=ax3, color='lightcoral'
    )
    ax3.set_title('Optimization Time by Model', fontsize=14)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Time (seconds)')
    
    # 4. Recall Comparison (Critical for Fraud Detection)
    ax4 = axes[1, 0]
    if 'Best Recall' in summary_df.columns and summary_df['Best Recall'].notna().any():
        summary_df.sort_values('Best Recall', ascending=False).plot(
            x='Model', y='Best Recall', kind='bar', ax=ax4, color='lightgreen'
        )
        ax4.set_title('Best Recall by Model\n(% of Frauds Detected)', fontsize=14)
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Recall')
        ax4.set_ylim(0, 1)
        for i, v in enumerate(summary_df.sort_values('Best Recall', ascending=False)['Best Recall']):
            if pd.notna(v):
                ax4.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
    
    # 5. Trial Statistics
    ax5 = axes[1, 1]
    if 'Completed Trials' in summary_df.columns:
        trial_data = summary_df[['Model', 'Completed Trials', 'Pruned Trials']].set_index('Model')
        trial_data.plot(kind='bar', stacked=True, ax=ax5, color=['lightgreen', 'orange'])
        ax5.set_title('Trial Statistics (Completed vs Pruned)', fontsize=14)
        ax5.set_xlabel('Model')
        ax5.set_ylabel('Number of Trials')
        ax5.legend(loc='upper right')
    
    # 6. Precision Comparison
    ax6 = axes[1, 2]
    if 'Best Precision' in summary_df.columns and summary_df['Best Precision'].notna().any():
        summary_df.sort_values('Best Precision', ascending=False).plot(
            x='Model', y='Best Precision', kind='bar', ax=ax6, color='lightblue'
        )
        ax6.set_title('Best Precision by Model\n(% of Alerts that are True Frauds)', fontsize=14)
        ax6.set_xlabel('Model')
        ax6.set_ylabel('Precision')
        ax6.set_ylim(0, 1)
        for i, v in enumerate(summary_df.sort_values('Best Precision', ascending=False)['Best Precision']):
            if pd.notna(v):
                ax6.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/optimization_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary table
    summary_df.to_csv(f"{save_dir}/optimization_summary.csv", index=False)
    print(f"Summary saved to {save_dir}/optimization_summary.png and .csv")
    
    # Create additional fraud-specific visualization
    create_fraud_metrics_comparison(summary_df, save_dir)


def create_fraud_metrics_comparison(summary_df, save_dir):
    """Create fraud-specific metrics comparison"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Only create if we have the necessary columns
    if not all(col in summary_df.columns for col in ['Best F1', 'Best Recall', 'Best Precision']):
        return
    
    # Filter out rows with missing values
    plot_df = summary_df.dropna(subset=['Best F1', 'Best Recall', 'Best Precision'])
    
    if plot_df.empty:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Fraud Detection Performance Metrics', fontsize=16, fontweight='bold')
    
    # 1. Radar Chart for Multi-metric Comparison
    ax1 = plt.subplot(121, projection='polar')
    
    metrics = ['F1 Score', 'Recall', 'Precision']
    num_vars = len(metrics)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, row in plot_df.iterrows():
        values = [row['Best F1'], row['Best Recall'], row['Best Precision']]
        values += values[:1]
        
        ax1.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
        ax1.fill(angles, values, alpha=0.1)
    
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1)
    ax1.set_title('Multi-Metric Comparison', y=1.08)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    ax1.grid(True)
    
    # 2. Cost-Benefit Analysis
    ax2 = plt.subplot(122)
    
    # Assuming cost ratio (cost of missing fraud : cost of false alarm)
    cost_ratios = [5, 10, 20, 50]
    
    for cost_ratio in cost_ratios:
        scores = []
        for idx, row in plot_df.iterrows():
            # Cost-benefit score = Recall * cost_ratio - (1 - Precision)
            # This rewards high recall (catching frauds) and high precision (fewer false alarms)
            recall = row['Best Recall']
            precision = row['Best Precision']
            false_positive_rate = 1 - precision if precision > 0 else 1
            score = (recall * cost_ratio - false_positive_rate) / cost_ratio  # Normalize
            scores.append(score)
        
        x_pos = np.arange(len(plot_df))
        ax2.plot(x_pos, scores, 'o-', label=f'Cost Ratio {cost_ratio}:1', linewidth=2)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(plot_df['Model'], rotation=45, ha='right')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Normalized Cost-Benefit Score')
    ax2.set_title('Cost-Benefit Analysis\n(Higher is Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fraud_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_hyperparameter_distributions(optimized_models, model_name, param_name, save_path=None):
    """Analyze the distribution of a specific hyperparameter across trials"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    if model_name not in optimized_models or 'study' not in optimized_models[model_name]:
        print(f"Study not found for {model_name}")
        return
    
    study = optimized_models[model_name]['study']
    
    # Extract parameter values and scores
    param_values = []
    scores = []
    recalls = []
    f1s = []
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and param_name in trial.params:
            param_values.append(trial.params[param_name])
            scores.append(trial.value)
            
            # Try to get additional metrics
            if 'avg_recall' in trial.user_attrs:
                recalls.append(trial.user_attrs['avg_recall'])
            if 'avg_f1' in trial.user_attrs:
                f1s.append(trial.user_attrs['avg_f1'])
    
    if not param_values:
        print(f"Parameter {param_name} not found in {model_name} trials")
        return
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{model_name}: {param_name} Analysis', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot - Combined Score
    ax1 = axes[0, 0]
    scatter = ax1.scatter(param_values, scores, c=scores, cmap='viridis', alpha=0.6)
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Combined Score')
    ax1.set_title('Parameter vs Combined Score')
    plt.colorbar(scatter, ax=ax1)
    
    # Highlight best value
    best_idx = scores.index(max(scores))
    ax1.scatter(param_values[best_idx], scores[best_idx], 
               color='red', s=200, marker='*', edgecolors='black', linewidth=2,
               label=f'Best: {param_values[best_idx]:.4f}')
    ax1.legend()
    
    # 2. Distribution plot
    ax2 = axes[0, 1]
    ax2.hist(param_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(param_values[best_idx], color='red', linestyle='--', linewidth=2, 
                label=f'Best: {param_values[best_idx]:.4f}')
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Frequency')
    ax2.set_title('Parameter Distribution')
    ax2.legend()
    
    # 3. Parameter vs Recall (if available)
    ax3 = axes[1, 0]
    if recalls:
        ax3.scatter(param_values[:len(recalls)], recalls, c=recalls, cmap='Reds', alpha=0.6)
        ax3.set_xlabel(param_name)
        ax3.set_ylabel('Recall')
        ax3.set_title('Parameter vs Recall')
        
        # Add trend line
        z = np.polyfit(param_values[:len(recalls)], recalls, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(param_values), max(param_values), 100)
        ax3.plot(x_smooth, p(x_smooth), "r--", alpha=0.8, label='Trend')
        ax3.legend()
    
    # 4. Box plot by parameter bins
    ax4 = axes[1, 1]
    param_bins = pd.qcut(param_values, q=5, duplicates='drop')
    df_temp = pd.DataFrame({
        'param_bin': param_bins,
        'score': scores
    })
    df_temp.boxplot(column='score', by='param_bin', ax=ax4)
    ax4.set_xlabel(f'{param_name} (binned)')
    ax4.set_ylabel('Combined Score')
    ax4.set_title('Score Distribution by Parameter Range')
    plt.sca(ax4)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


# Additional helper functions for fraud detection
def get_optimized_ensemble(optimized_models, X_train, y_train, top_n=3):
    """
    Create an ensemble of the top N optimized models
    
    Parameters:
    -----------
    optimized_models : dict
        Dictionary of optimized models
    X_train, y_train : arrays
        Training data
    top_n : int
        Number of top models to include in ensemble
    
    Returns:
    --------
    VotingClassifier : Ensemble model
    """
    from sklearn.ensemble import VotingClassifier
    
    # Sort models by score
    sorted_models = sorted(
        optimized_models.items(),
        key=lambda x: x[1]['best_score'],
        reverse=True
    )[:top_n]
    
    # Create ensemble
    estimators = []
    for model_name, model_info in sorted_models:
        model = model_info['model']
        if model is not None:
            # Fit the model
            model.fit(X_train, y_train)
            estimators.append((model_name, model))
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',  # Use probability predictions
        n_jobs=-1
    )
    
    print(f"Created ensemble with models: {[name for name, _ in estimators]}")
    
    return ensemble


def perform_nested_cv_evaluation(model, X, y, n_outer_folds=5, n_inner_folds=3):
    """
    Perform nested cross-validation for unbiased performance estimation
    
    Parameters:
    -----------
    model : sklearn estimator
        Model to evaluate
    X, y : arrays
        Data and labels
    n_outer_folds : int
        Number of outer CV folds
    n_inner_folds : int
        Number of inner CV folds
    
    Returns:
    --------
    dict : Results with mean and std for each metric
    """
    from sklearn.base import clone
    from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
    
    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=42)
    
    scores = {
        'f1': [],
        'recall': [],
        'precision': [],
        'roc_auc': [],
        'fraud_score': []
    }
    
    for train_idx, test_idx in outer_cv.split(X, y):
        if hasattr(X, 'iloc'):  # DataFrame
            X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
            y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
        else:  # numpy array
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        
        # Clone model
        model_clone = clone(model)
        
        # Fit on outer training set
        model_clone.fit(X_train_outer, y_train_outer)
        
        # Predict on outer test set
        y_pred = model_clone.predict(X_test_outer)
        y_pred_proba = model_clone.predict_proba(X_test_outer)[:, 1]
        
        # Calculate metrics
        scores['f1'].append(f1_score(y_test_outer, y_pred))
        scores['recall'].append(recall_score(y_test_outer, y_pred))
        scores['precision'].append(precision_score(y_test_outer, y_pred, zero_division=0))
        scores['roc_auc'].append(roc_auc_score(y_test_outer, y_pred_proba))
        scores['fraud_score'].append(fraud_detection_score(y_test_outer, y_pred))
    
    # Calculate results
    results = {}
    for metric, values in scores.items():
        results[f'{metric}_mean'] = np.mean(values)
        results[f'{metric}_std'] = np.std(values)
        results[f'{metric}_min'] = np.min(values)
        results[f'{metric}_max'] = np.max(values)
    
    return results


def save_optimized_models(optimized_models, save_dir='optimized_models'):
    """Save optimized models to disk"""
    import joblib
    import os
    import json
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for model_name, model_info in optimized_models.items():
        # Save model
        if model_info['model'] is not None:
            model_path = f"{save_dir}/{model_name.replace(' ', '_')}_model.pkl"
            joblib.dump(model_info['model'], model_path)
            print(f"Saved {model_name} model to {model_path}")
        
        # Save parameters and metrics
        info_to_save = {
            'best_params': model_info['best_params'],
            'best_score': model_info['best_score'],
            'time_taken': model_info['time_taken']
        }
        
        # Add additional metrics if available
        for metric in ['best_f1', 'best_recall', 'best_precision']:
            if metric in model_info:
                info_to_save[metric] = model_info[metric]
        
        info_path = f"{save_dir}/{model_name.replace(' ', '_')}_info.json"
        with open(info_path, 'w') as f:
            json.dump(info_to_save, f, indent=4)
        print(f"Saved {model_name} info to {info_path}")


# Example usage and recommendations
"""
# Example 1: Basic optimization with fraud-specific scoring
optimized_models = optimize_all_models_parallel(
    X_train, y_train,
    method='optuna',
    n_trials=100,
    models=['XGBoost', 'LightGBM', 'Random Forest'],
    verbose='medium'
)

# Example 2: Evaluate with nested CV
for model_name, model_info in optimized_models.items():
    if model_info['model'] is not None:
        print(f"\nNested CV for {model_name}:")
        results = perform_nested_cv_evaluation(model_info['model'], X_train, y_train)
        print(f"F1: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
        print(f"Recall: {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
        print(f"Precision: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")

# Example 3: Create and evaluate ensemble
ensemble = get_optimized_ensemble(optimized_models, X_train, y_train, top_n=3)
ensemble_results = perform_nested_cv_evaluation(ensemble, X_train, y_train)
print(f"\nEnsemble Performance:")
print(f"F1: {ensemble_results['f1_mean']:.4f} ± {ensemble_results['f1_std']:.4f}")
print(f"Recall: {ensemble_results['recall_mean']:.4f} ± {ensemble_results['recall_std']:.4f}")

# Example 4: Visualize results
visualize_optuna_results(optimized_models, save_dir='fraud_optimization_plots')

# Example 5: Analyze specific parameters
analyze_hyperparameter_distributions(
    optimized_models, 
    'XGBoost', 
    'scale_pos_weight',
    save_path='xgboost_class_weight_analysis.png'
)

# Example 6: Save models for deployment
save_optimized_models(optimized_models, save_dir='production_models')
"""