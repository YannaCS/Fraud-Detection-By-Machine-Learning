import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint, loguniform
import optuna
from optimization_func import calculate_class_weight, fraud_scorer
import warnings
warnings.filterwarnings('ignore')
import logging
import time

def two_stage_optimization(X_train, y_train, model_name='XGBoost', 
                         stage1_trials=30, stage2_trials=70, 
                         range_factor=0.5, verbose=True, show_trial_details=False):
    """
    Two-stage optimization: RandomizedSearch for exploration, then Optuna for exploitation
    
    Parameters:
    -----------
    X_train, y_train : arrays
        Training data
    model_name : str
        Model to optimize
    stage1_trials : int
        Number of trials for RandomizedSearch (exploration)
    stage2_trials : int
        Number of trials for Optuna (exploitation)
    range_factor : float
        How much to narrow the range (0.5 = ±50% around best value)
    verbose : bool
        Print progress summary
    show_trial_details : bool
        Show detailed trial-by-trial results (default: False)
    
    Returns:
    --------
    dict : Optimization results with best model and parameters
    """
    
    # Import all model classes at the beginning
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    pos_weight = calculate_class_weight(y_train)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Two-Stage Optimization for {model_name}")
        print(f"{'='*60}")
        print(f"Stage 1: RandomizedSearch with {stage1_trials} trials (exploration)")
        print(f"Stage 2: Optuna with {stage2_trials} trials (exploitation)")
        print(f"Total trials: {stage1_trials + stage2_trials}")
        print(f"Positive class weight: {pos_weight:.2f}")
    
    # ========================================
    # STAGE 1: Broad exploration with RandomizedSearch
    # ========================================
    
    # Define broad parameter distributions for initial search
    if model_name == 'XGBoost':
        from xgboost import XGBClassifier
        broad_params = {
            'n_estimators': randint(50, 1000),
            'max_depth': randint(3, 15),
            'learning_rate': loguniform(0.001, 0.3),
            'subsample': uniform(0.5, 0.5),  # This generates values in [0.5, 1.0]
            'colsample_bytree': uniform(0.3, 0.7),  # This generates values in [0.3, 1.0]
            'scale_pos_weight': uniform(pos_weight * 0.5, pos_weight * 1.5),
            'gamma': uniform(0, 5),
            'reg_alpha': loguniform(1e-8, 10),
            'reg_lambda': loguniform(1e-8, 10),
            'min_child_weight': randint(1, 20)
        }
        # Configure for max performance
        base_params = configure_model_for_max_performance(model_name, {
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': -1,
            'tree_method': 'hist'
        })
        base_model = XGBClassifier(**base_params)
        
    elif model_name == 'LightGBM':
        from lightgbm import LGBMClassifier
        broad_params = {
            'n_estimators': randint(50, 1000),
            'max_depth': randint(3, 20),
            'learning_rate': loguniform(0.001, 0.3),
            'num_leaves': randint(20, 300),
            'feature_fraction': uniform(0.3, 0.7),
            'bagging_fraction': uniform(0.3, 0.7),
            'scale_pos_weight': uniform(pos_weight * 0.5, pos_weight * 1.5),
            'reg_alpha': uniform(0, 10),
            'reg_lambda': uniform(0, 10),
            'min_child_samples': randint(5, 100)
        }
        # Configure for max performance
        base_params = configure_model_for_max_performance(model_name, {
            'random_state': 42,
            'verbosity': -1,
            'objective': 'binary'
        })
        base_model = LGBMClassifier(**base_params)
        
    elif model_name == 'Random Forest':
        from sklearn.ensemble import RandomForestClassifier
        broad_params = {
            'n_estimators': randint(50, 1000),
            'max_depth': randint(5, 50),
            'min_samples_split': randint(2, 50),
            'min_samples_leaf': randint(1, 20),
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
            'class_weight': [{0: 1, 1: w} for w in np.linspace(pos_weight * 0.5, pos_weight * 2.0, 10)]
        }
        # Configure for max performance
        base_params = configure_model_for_max_performance(model_name, {'random_state': 42})
        base_model = RandomForestClassifier(**base_params)
        
    elif model_name == 'Gradient Boosting':
        from sklearn.ensemble import GradientBoostingClassifier
        broad_params = {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 10),
            'learning_rate': loguniform(0.01, 0.3),
            'subsample': uniform(0.5, 0.5),
            'min_samples_split': randint(2, 30),
            'min_samples_leaf': randint(1, 15),
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7]
        }
        base_params = configure_model_for_max_performance(model_name, {'random_state': 42})
        base_model = GradientBoostingClassifier(**base_params)
        
    elif model_name == 'Neural Network':
        from sklearn.neural_network import MLPClassifier
        broad_params = {
            'hidden_layer_sizes': [(randint(50, 300).rvs(),), 
                                   (randint(100, 300).rvs(), randint(50, 150).rvs()),
                                   (randint(200, 300).rvs(), randint(100, 200).rvs(), randint(50, 100).rvs())],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': loguniform(1e-5, 0.1),
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': loguniform(1e-4, 0.01),
            'max_iter': randint(500, 1500),
            'early_stopping': [True],
            'validation_fraction': [0.1],
            'n_iter_no_change': randint(10, 30)
        }
        base_params = configure_model_for_max_performance(model_name, {'random_state': 42})
        base_model = MLPClassifier(**base_params)
        
    elif model_name == 'Logistic Regression':
        from sklearn.linear_model import LogisticRegression
        broad_params = {
            'C': loguniform(1e-4, 1e2),
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg'],
            'l1_ratio': uniform(0.0, 1.0),  # For elasticnet
            'max_iter': randint(500, 3000),
            'tol': loguniform(1e-6, 1e-3),
            'class_weight': [{0: 1, 1: w} for w in np.linspace(pos_weight * 0.5, pos_weight * 2.0, 10)]
        }
        base_params = configure_model_for_max_performance(model_name, {'random_state': 42})
        base_model = LogisticRegression(**base_params)
        
    elif model_name == 'Decision Tree':
        from sklearn.tree import DecisionTreeClassifier
        broad_params = {
            'max_depth': randint(3, 30),
            'min_samples_split': randint(2, 50),
            'min_samples_leaf': randint(1, 20),
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'min_impurity_decrease': loguniform(1e-8, 0.01),
            'class_weight': [{0: 1, 1: w} for w in np.linspace(pos_weight * 0.5, pos_weight * 2.0, 10)]
        }
        base_params = configure_model_for_max_performance(model_name, {'random_state': 42})
        base_model = DecisionTreeClassifier(**base_params)
    
    # Run RandomizedSearch with all CPU cores
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=broad_params,
        n_iter=stage1_trials,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1',  # Use F1 for speed in stage 1
        n_jobs=-1,  # Use all CPU cores for CV
        verbose=1 if show_trial_details else 0,  # Show details only if requested
        random_state=42
    )
    
    if verbose:
        print(f"\nRunning Stage 1...")
    
    # Store trial history
    stage1_history = []
    
    stage1_start = time.time()
    random_search.fit(X_train, y_train)
    stage1_time = time.time() - stage1_start
    
    stage1_best_params = random_search.best_params_
    stage1_best_score = random_search.best_score_
    
    # Store Stage 1 results
    if hasattr(random_search, 'cv_results_'):
        stage1_history = {
            'mean_test_scores': random_search.cv_results_['mean_test_score'].tolist(),
            'params': random_search.cv_results_['params'],
            'time': stage1_time
        }
    
    if verbose:
        print(f"Stage 1 complete! (Time: {stage1_time:.1f}s)")
        print(f"Best F1 Score: {stage1_best_score:.4f}")
        if show_trial_details:
            print(f"Best parameters found:")
            for param, value in stage1_best_params.items():
                print(f"  {param}: {value}")
    
    # ========================================
    # STAGE 2: Focused search with Optuna around best parameters
    # ========================================
    
    if verbose:
        print(f"\n{'='*40}")
        print(f"Stage 2: Focused search around best parameters")
        print(f"{'='*40}")
    
    def create_narrow_ranges(best_params, range_factor=0.5):
        """Create narrow ranges around best parameters with bounds checking"""
        narrow_ranges = {}
        
        for param, value in best_params.items():
            if isinstance(value, (int, np.integer)):
                # For integers, search ±range_factor around best value
                min_val = max(1, int(value * (1 - range_factor)))
                max_val = int(value * (1 + range_factor))
                narrow_ranges[param] = ('int', min_val, max_val)
                
            elif isinstance(value, (float, np.floating)):
                # For floats, search ±range_factor around best value
                min_val = value * (1 - range_factor)
                max_val = value * (1 + range_factor)
                
                # Handle parameters with known bounds
                if param in ['subsample', 'colsample_bytree', 'colsample_bylevel', 'colsample_bynode', 
                            'feature_fraction', 'bagging_fraction', 'l1_ratio']:
                    # These parameters must be in [0, 1]
                    min_val = max(0.1, min(min_val, 0.999))
                    max_val = min(1.0, max_val)
                elif param in ['learning_rate', 'learning_rate_init']:
                    # Learning rate should be positive
                    min_val = max(0.0001, min_val)
                elif param in ['alpha', 'reg_alpha', 'reg_lambda', 'tol', 'gamma', 'min_impurity_decrease']:
                    # These should be non-negative
                    min_val = max(0.0, min_val)
                    
                # Check if this was likely a log-scale parameter
                if param in ['learning_rate', 'reg_alpha', 'reg_lambda', 'alpha', 'tol', 'learning_rate_init', 'C', 'min_impurity_decrease']:
                    narrow_ranges[param] = ('log', min_val, max_val)
                else:
                    narrow_ranges[param] = ('float', min_val, max_val)
                    
            elif isinstance(value, str) or value is None:
                # For categorical, keep the best value but allow some alternatives
                if param == 'max_features':
                    if value in ['sqrt', 'log2', None]:
                        narrow_ranges[param] = ('categorical', ['sqrt', 'log2', None, 0.3, 0.5])
                    else:
                        # For numeric max_features, ensure it's in valid range
                        if isinstance(value, (float, np.floating)):
                            alternatives = [max(0.1, value-0.1), value, min(1.0, value+0.1)]
                            narrow_ranges[param] = ('categorical', alternatives)
                        else:
                            narrow_ranges[param] = ('categorical', [value])
                else:
                    narrow_ranges[param] = ('fixed', value)
                    
            elif isinstance(value, dict):  # class_weight
                # Extract the positive class weight
                pos_weight_value = value.get(1, pos_weight)
                min_val = pos_weight_value * (1 - range_factor)
                max_val = pos_weight_value * (1 + range_factor)
                narrow_ranges['pos_weight'] = ('float', min_val, max_val)
        
        return narrow_ranges
    
    # Create narrow ranges
    narrow_ranges = create_narrow_ranges(stage1_best_params, range_factor)
    
    if verbose and show_trial_details:
        print("Narrow ranges for Stage 2:")
        for param, range_info in narrow_ranges.items():
            if range_info[0] in ['int', 'float', 'log']:
                print(f"  {param}: [{range_info[1]:.4f}, {range_info[2]:.4f}]")
            else:
                print(f"  {param}: {range_info[1]}")
    
    # Define Optuna objective with narrow ranges
    def objective(trial):
        params = {}
        
        # Convert narrow ranges to trial suggestions
        for param, range_info in narrow_ranges.items():
            if range_info[0] == 'int':
                params[param] = trial.suggest_int(param, range_info[1], range_info[2])
            elif range_info[0] == 'float':
                params[param] = trial.suggest_float(param, range_info[1], range_info[2])
            elif range_info[0] == 'log':
                params[param] = trial.suggest_float(param, range_info[1], range_info[2], log=True)
            elif range_info[0] == 'categorical':
                params[param] = trial.suggest_categorical(param, range_info[1])
            elif range_info[0] == 'fixed':
                params[param] = range_info[1]
        
        # Handle special cases
        if 'pos_weight' in params and model_name in ['Random Forest', 'XGBoost', 'Logistic Regression', 'Decision Tree']:
            if model_name == 'Random Forest':
                params['class_weight'] = {0: 1, 1: params.pop('pos_weight')}
            elif model_name == 'XGBoost':
                params['scale_pos_weight'] = params.pop('pos_weight')
            elif model_name in ['Logistic Regression', 'Decision Tree']:
                params['class_weight'] = {0: 1, 1: params.pop('pos_weight')}
        
        # Add fixed parameters and configure for max performance
        if model_name == 'XGBoost':
            params.update({'random_state': 42, 'eval_metric': 'logloss'})
            params = configure_model_for_max_performance(model_name, params)
            model = XGBClassifier(**params)
        elif model_name == 'LightGBM':
            params.update({'random_state': 42, 'verbosity': -1, 'objective': 'binary'})
            params = configure_model_for_max_performance(model_name, params)
            model = LGBMClassifier(**params)
        elif model_name == 'Random Forest':
            params.update({'random_state': 42})
            params = configure_model_for_max_performance(model_name, params)
            model = RandomForestClassifier(**params)
        elif model_name == 'Gradient Boosting':
            params.update({'random_state': 42})
            params = configure_model_for_max_performance(model_name, params)
            model = GradientBoostingClassifier(**params)
        elif model_name == 'Neural Network':
            params.update({'random_state': 42})
            params = configure_model_for_max_performance(model_name, params)
            model = MLPClassifier(**params)
        elif model_name == 'Logistic Regression':
            params.update({'random_state': 42})
            params = configure_model_for_max_performance(model_name, params)
            model = LogisticRegression(**params)
        elif model_name == 'Decision Tree':
            params.update({'random_state': 42})
            params = configure_model_for_max_performance(model_name, params)
            model = DecisionTreeClassifier(**params)
        
        # Cross-validation with multiple metrics
        from sklearn.metrics import f1_score, recall_score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        f1_scores = []
        recall_scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            X_fold_val = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
            y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            
            f1_scores.append(f1_score(y_fold_val, y_pred))
            recall_scores.append(recall_score(y_fold_val, y_pred))
        
        # Combined score for fraud detection
        return 0.4 * np.mean(f1_scores) + 0.6 * np.mean(recall_scores)
    
    # Configure Optuna logging
    if not show_trial_details:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # Create and run Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    if verbose:
        print(f"\nRunning Stage 2 optimization...")
    
    # Custom callback to show progress without trial details
    stage2_start = time.time()
    best_value_history = []
    
    def progress_callback(study, trial):
        if verbose and not show_trial_details:
            # Only show progress updates every 10% of trials
            if trial.number % max(1, stage2_trials // 10) == 0:
                elapsed = time.time() - stage2_start
                progress = (trial.number + 1) / stage2_trials * 100
                print(f"  Progress: {progress:.0f}% ({trial.number + 1}/{stage2_trials} trials) - "
                      f"Best: {study.best_value:.4f} - Time: {elapsed:.1f}s")
        
        # Store best value history
        best_value_history.append(study.best_value if len(study.trials) > 0 else 0)
    
    study.optimize(
        objective, 
        n_trials=stage2_trials, 
        show_progress_bar=show_trial_details,
        callbacks=[progress_callback] if not show_trial_details else []
    )
    
    stage2_time = time.time() - stage2_start
    
    # Get final best parameters
    final_best_params = study.best_params
    
    # Merge with fixed parameters from stage 1
    for param, range_info in narrow_ranges.items():
        if range_info[0] == 'fixed':
            final_best_params[param] = range_info[1]
    
    # Create final model with performance optimization
    if model_name == 'XGBoost':
        # Handle pos_weight conversion
        if 'pos_weight' in final_best_params:
            final_best_params['scale_pos_weight'] = final_best_params.pop('pos_weight')
        final_best_params.update({'random_state': 42, 'eval_metric': 'logloss'})
        final_best_params = configure_model_for_max_performance(model_name, final_best_params)
        final_model = XGBClassifier(**final_best_params)
    elif model_name == 'LightGBM':
        final_best_params.update({'random_state': 42, 'verbosity': -1, 'objective': 'binary'})
        final_best_params = configure_model_for_max_performance(model_name, final_best_params)
        final_model = LGBMClassifier(**final_best_params)
    elif model_name == 'Random Forest':
        if 'pos_weight' in final_best_params:
            final_best_params['class_weight'] = {0: 1, 1: final_best_params.pop('pos_weight')}
        final_best_params.update({'random_state': 42})
        final_best_params = configure_model_for_max_performance(model_name, final_best_params)
        final_model = RandomForestClassifier(**final_best_params)
    elif model_name == 'Gradient Boosting':
        final_best_params.update({'random_state': 42})
        final_best_params = configure_model_for_max_performance(model_name, final_best_params)
        final_model = GradientBoostingClassifier(**final_best_params)
    elif model_name == 'Neural Network':
        final_best_params.update({'random_state': 42})
        final_best_params = configure_model_for_max_performance(model_name, final_best_params)
        final_model = MLPClassifier(**final_best_params)
    elif model_name == 'Logistic Regression':
        if 'pos_weight' in final_best_params:
            final_best_params['class_weight'] = {0: 1, 1: final_best_params.pop('pos_weight')}
        final_best_params.update({'random_state': 42})
        final_best_params = configure_model_for_max_performance(model_name, final_best_params)
        final_model = LogisticRegression(**final_best_params)
    elif model_name == 'Decision Tree':
        if 'pos_weight' in final_best_params:
            final_best_params['class_weight'] = {0: 1, 1: final_best_params.pop('pos_weight')}
        final_best_params.update({'random_state': 42})
        final_best_params = configure_model_for_max_performance(model_name, final_best_params)
        final_model = DecisionTreeClassifier(**final_best_params)
    
    # Fit final model on full training data
    final_model.fit(X_train, y_train)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(f"Stage 1: {stage1_best_score:.4f} (Time: {stage1_time:.1f}s)")
        print(f"Stage 2: {study.best_value:.4f} (Time: {stage2_time:.1f}s)")
        print(f"Improvement: {(study.best_value - stage1_best_score):.4f}")
        print(f"Total time: {stage1_time + stage2_time:.1f}s")
        
        if show_trial_details:
            print(f"\nFinal best parameters:")
            for param, value in final_best_params.items():
                print(f"  {param}: {value}")
    
    return {
        'model': final_model,
        'best_params': final_best_params,
        'best_score': study.best_value,
        'stage1_score': stage1_best_score,
        'stage1_params': stage1_best_params,
        'stage1_history': stage1_history,
        'stage2_study': study,
        'stage2_history': best_value_history,
        'improvement': study.best_value - stage1_best_score,
        'total_time': stage1_time + stage2_time,
        'stage1_time': stage1_time,
        'stage2_time': stage2_time
    }


# Detect system capabilities
import platform
import os

def get_system_info():
    """Detect system capabilities including CPU cores and GPU availability"""
    import multiprocessing
    
    system_info = {
        'cpu_count': multiprocessing.cpu_count(),
        'platform': platform.system(),
        'processor': platform.processor(),
        'is_apple_silicon': False,
        'has_metal': False
    }
    
    # Check for Apple Silicon
    if system_info['platform'] == 'Darwin':  # macOS
        if 'arm' in system_info['processor'].lower() or 'apple' in system_info['processor'].lower():
            system_info['is_apple_silicon'] = True
            # Metal is available on Apple Silicon
            system_info['has_metal'] = True
    
    return system_info

# Get system info once
SYSTEM_INFO = get_system_info()

def configure_model_for_max_performance(model_name, params, system_info=SYSTEM_INFO):
    """Configure model parameters for maximum performance on the system"""
    
    cpu_count = system_info['cpu_count']
    
    # Add CPU/GPU optimization parameters
    if model_name == 'XGBoost':
        params.update({
            'n_jobs': -1,  # Use all CPU cores
            'tree_method': 'hist',  # Fast histogram-based method
        })
        # Remove deprecated parameters if they exist
        params.pop('use_label_encoder', None)
        params.pop('predictor', None)
        
        # On Apple Silicon, XGBoost can use GPU via Metal
        if system_info['is_apple_silicon']:
            # Note: Requires XGBoost built with GPU support
            # params['tree_method'] = 'gpu_hist'  # Uncomment if GPU-enabled XGBoost installed
            pass  # No need to set predictor anymore
            
    elif model_name == 'LightGBM':
        params.update({
            'n_jobs': -1,  # Use all CPU cores
            'num_threads': cpu_count,  # Explicit thread count
            'device_type': 'cpu',  # Explicit CPU usage
            'force_col_wise': True,  # Better for small datasets
            'force_row_wise': False,  # Better for large datasets
        })
        # LightGBM GPU support on Mac is limited
        
    elif model_name == 'Random Forest':
        params.update({
            'n_jobs': -1,  # Use all CPU cores
        })
        
    elif model_name == 'Gradient Boosting':
        # Sklearn GradientBoosting doesn't support n_jobs
        # But we can use parallel CV
        pass
        
    elif model_name == 'Neural Network':
        # MLPClassifier doesn't directly support GPU
        # But uses BLAS which can use multiple cores
        params.update({
            'max_iter': params.get('max_iter', 1000),
        })
        
    elif model_name == 'Logistic Regression':
        # Some solvers support n_jobs
        if params.get('solver') in ['liblinear', 'saga']:
            params.update({
                'n_jobs': -1,  # Use all CPU cores
            })
            
    elif model_name == 'Decision Tree':
        # DecisionTreeClassifier doesn't support n_jobs
        # But it's fast anyway
        pass
    
    return params

# Convenience function to optimize multiple models
def two_stage_optimize_multiple(X_train, y_train, models, n_trials=100, stage1_ratio=0.3, 
                               verbose=True, show_trial_details=False):
    """
    Run two-stage optimization for multiple models
    Ensures each model uses all available CPU cores (and GPU when possible)
    
    Parameters:
    -----------
    X_train, y_train : arrays
        Training data
    models : list
        List of model names to optimize
    n_trials : int
        Number of trials for EACH model (not total)
    stage1_ratio : float
        Proportion of trials for stage 1 (default: 0.3 = 30%)
    verbose : bool
        Show progress summary
    show_trial_details : bool
        Show detailed trial-by-trial results
    
    Returns:
    --------
    dict : Results for each model with optimization history
    """
    
    stage1_trials = int(n_trials * stage1_ratio)
    stage2_trials = n_trials - stage1_trials
    
    # Display system information
    print(f"System Information:")
    print(f"  Platform: {SYSTEM_INFO['platform']}")
    print(f"  Processor: {SYSTEM_INFO['processor']}")
    print(f"  CPU Cores: {SYSTEM_INFO['cpu_count']}")
    print(f"  Apple Silicon: {SYSTEM_INFO['is_apple_silicon']}")
    print(f"  Metal Available: {SYSTEM_INFO['has_metal']}")
    print()
    
    print(f"Optimizing {len(models)} models with {n_trials} trials each")
    print(f"Stage 1: {stage1_trials} trials, Stage 2: {stage2_trials} trials")
    print(f"Total trials: {n_trials * len(models)}")
    print(f"Each model will use all {SYSTEM_INFO['cpu_count']} CPU cores")
    print(f"Trial details: {'ON' if show_trial_details else 'OFF'}")
    
    results = {}
    
    for model_name in models:
        print(f"\n{'='*80}")
        print(f"Optimizing {model_name}")
        print(f"{'='*80}")
        
        # Set environment variables for maximum performance
        os.environ['OMP_NUM_THREADS'] = str(SYSTEM_INFO['cpu_count'])
        os.environ['OPENBLAS_NUM_THREADS'] = str(SYSTEM_INFO['cpu_count'])
        os.environ['MKL_NUM_THREADS'] = str(SYSTEM_INFO['cpu_count'])
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(SYSTEM_INFO['cpu_count'])
        os.environ['NUMEXPR_NUM_THREADS'] = str(SYSTEM_INFO['cpu_count'])
        
        result = two_stage_optimization(
            X_train, y_train,
            model_name=model_name,
            stage1_trials=stage1_trials,
            stage2_trials=stage2_trials,
            range_factor=0.5,  # Search ±50% around stage 1 best
            verbose=verbose,
            show_trial_details=show_trial_details
        )
        
        results[model_name] = result
        
        # Clear GPU memory if used (for next model)
        import gc
        gc.collect()
    
    # Summary
    print(f"\n{'='*80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Stage 1':<10} {'Stage 2':<10} {'Improvement':<12} {'Time (s)':<10}")
    print("-" * 80)
    
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['stage1_score']:<10.4f} {result['best_score']:<10.4f} "
              f"{result['improvement']:<12.4f} {result['total_time']:<10.1f}")
    
    return results


# Parallel version for faster execution
def two_stage_optimize_parallel(X_train, y_train, models, n_trials=100, stage1_ratio=0.3, n_jobs=-1):
    """
    Run two-stage optimization for multiple models in parallel
    
    Parameters:
    -----------
    X_train, y_train : arrays
        Training data
    models : list
        List of model names to optimize
    n_trials : int
        Number of trials for EACH model
    stage1_ratio : float
        Proportion of trials for stage 1
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    
    Returns:
    --------
    dict : Results for each model
    """
    from joblib import Parallel, delayed
    
    stage1_trials = int(n_trials * stage1_ratio)
    stage2_trials = n_trials - stage1_trials
    
    print(f"Parallel optimization of {len(models)} models")
    print(f"Trials per model: {n_trials} (Stage 1: {stage1_trials}, Stage 2: {stage2_trials})")
    print(f"Total trials: {n_trials * len(models)}")
    print(f"Using {n_jobs if n_jobs > 0 else 'all'} CPU cores")
    
    # Run optimization in parallel
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(two_stage_optimization)(
            X_train, y_train,
            model_name=model_name,
            stage1_trials=stage1_trials,
            stage2_trials=stage2_trials,
            range_factor=0.5,
            verbose=False,  # Less verbose in parallel mode
            show_trial_details=False
        )
        for model_name in models
    )
    
    # Convert to dictionary
    results = {model_name: result for model_name, result in zip(models, results_list)}
    
    # Summary
    print(f"\n{'='*80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Stage 1':<10} {'Stage 2':<10} {'Improvement':<12}")
    print("-" * 60)
    
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['stage1_score']:<10.4f} {result['best_score']:<10.4f} "
              f"{result['improvement']:<12.4f}")
    
    return results


# Example usage:
"""
# Option 1: Single model optimization with 100 trials
result = two_stage_optimization(
    X_train, y_train,
    model_name='XGBoost',
    stage1_trials=30,  # 30 trials for exploration
    stage2_trials=70,  # 70 trials for exploitation
    range_factor=0.5,  # Search ±50% around best values
    verbose=True,
    show_trial_details=False  # Clean output
)

# Option 2: Multiple models, each with 100 trials (200 total)
results = two_stage_optimize_multiple(
    X_train, y_train,
    models=['XGBoost', 'LightGBM'],
    n_trials=100,  # 100 trials per model
    stage1_ratio=0.3,  # 30% for exploration
    show_trial_details=False  # Clean output
)

# Option 3: Parallel execution for faster results
results = two_stage_optimize_parallel(
    X_train, y_train,
    models=['XGBoost', 'LightGBM', 'Random Forest'],
    n_trials=100,  # 100 trials per model (300 total)
    stage1_ratio=0.3,
    n_jobs=-1  # Use all CPU cores
)

# Access the best model
best_model = results['XGBoost']['model']
best_params = results['XGBoost']['best_params']
best_score = results['XGBoost']['best_score']

# Plot optimization history
import matplotlib.pyplot as plt
plt.plot(results['XGBoost']['stage2_history'])
plt.xlabel('Trial')
plt.ylabel('Best Score')
plt.title('XGBoost Optimization Progress')
plt.show()
"""