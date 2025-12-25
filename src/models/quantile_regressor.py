import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit
import joblib

class QuantileRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, quantiles=[0.1, 0.5, 0.9], params=None):
        self.quantiles = quantiles
        self.params = params or {}
        # Base params for LightGBM
        self.base_params = {
            'objective': 'quantile',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1,
            'metric': 'quantile'
        }
        # Update with user params
        self.base_params.update(self.params)
        
        self.models = {}
        self.feature_names = None

    def fit(self, X, y, eval_set=None):
        """
        Train a model for each quantile.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            eval_set: tuple (X_val, y_val) for early stopping validation
        """
        self.feature_names = list(X.columns)
        
        for q in self.quantiles:
            print(f"Training model for quantile {q}...")
            
            params = self.base_params.copy()
            params['alpha'] = q
            
            # Create datasets
            train_ds = lgb.Dataset(X, label=y)
            valid_sets = [train_ds]
            callbacks = []
            
            if eval_set:
                X_val, y_val = eval_set
                val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
                valid_sets.append(val_ds)
                callbacks = [
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0) # Squelch logs, too noisy for 3 models * folds
                ]
            
            model = lgb.train(
                params,
                train_ds,
                num_boost_round=1000,
                valid_sets=valid_sets,
                callbacks=callbacks
            )
            
            self.models[q] = model
            
        return self

    def predict(self, X):
        """
        Predict return quantiles.
        
        Returns:
            DataFrame with columns corresponding to quantiles (e.g., 'q_0.1', 'q_0.5', 'q_0.9')
        """
        preds = pd.DataFrame(index=X.index)
        
        for q, model in self.models.items():
            preds[f'q_{q}'] = model.predict(X)
            
        return preds
    
    def save_models(self, path_prefix):
        """Save models to disk"""
        joblib.dump(self.models, f"{path_prefix}_models.joblib")
        joblib.dump(self.feature_names, f"{path_prefix}_features.joblib")
        
    def load_models(self, path_prefix):
        self.models = joblib.load(f"{path_prefix}_models.joblib")
        self.feature_names = joblib.load(f"{path_prefix}_features.joblib")

    def get_feature_importance(self):
        """Aggregate feature importance across quantiles (mean)"""
        if not self.models:
            return None
            
        importances = pd.DataFrame(index=self.feature_names)
        
        for q, model in self.models.items():
            importances[f'q_{q}'] = model.feature_importance(importance_type='gain')
            
        importances['mean_gain'] = importances.mean(axis=1)
        return importances.sort_values('mean_gain', ascending=False)
