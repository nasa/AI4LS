# feature_importance_service/src/importance_methods.py
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class FeatureImportanceMethods:
    """Methods for computing feature importance"""
    
    @staticmethod
    def built_in_importance(model, feature_names: List[str]) -> List[Dict]:
        """
        Get built-in feature importance from tree-based models
        (Random Forest, Gradient Boosting, XGBoost, etc.)
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                results = []
                for i, (name, importance) in enumerate(zip(feature_names, importances)):
                    results.append({
                        "feature_name": name,
                        "importance": float(importance),
                        "rank": i + 1
                    })
                
                # Sort by importance descending
                results.sort(key=lambda x: x['importance'], reverse=True)
                
                # Update ranks after sorting
                for i, result in enumerate(results):
                    result['rank'] = i + 1
                
                return results
            else:
                logger.warning("Model does not have built-in feature importances")
                return []
                
        except Exception as e:
            logger.error(f"Error computing built-in importance: {e}")
            return []
    
    @staticmethod
    def recursive_feature_elimination(
        model, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_features_to_select: int = None,
        step: int = 1
    ) -> List[Dict]:
        """
        Recursive Feature Elimination (RFE)
        
        Args:
            model: Trained sklearn model
            X: Feature matrix
            y: Target variable
            n_features_to_select: Number of features to select (default: half)
            step: Number of features to remove at each iteration
        """
        try:
            if n_features_to_select is None:
                n_features_to_select = max(1, len(X.columns) // 2)
            
            logger.info(f"Running RFE: selecting {n_features_to_select} features")
            
            # Create RFE selector
            rfe = RFE(
                estimator=model,
                n_features_to_select=n_features_to_select,
                step=step
            )
            
            # Fit RFE
            rfe.fit(X, y)
            
            results = []
            for i, (name, selected, ranking) in enumerate(
                zip(X.columns, rfe.support_, rfe.ranking_)
            ):
                # RFE ranking: 1 = selected, >1 = eliminated (higher = eliminated earlier)
                # Convert to importance score (inverse of rank)
                importance = 1.0 / ranking if ranking > 0 else 0.0
                
                results.append({
                    "feature_name": name,
                    "importance": float(importance),
                    "rank": int(ranking),
                    "selected": bool(selected)
                })
            
            # Sort by rank (lower rank = more important)
            results.sort(key=lambda x: x['rank'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing RFE: {e}", exc_info=True)
            return []
    
    @staticmethod
    def permutation_feature_importance(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10,
        random_state: int = 42
    ) -> List[Dict]:
        """
        Permutation Feature Importance
        
        Measures importance by randomly shuffling each feature and
        measuring the decrease in model performance.
        
        Args:
            model: Trained sklearn model
            X: Feature matrix (test set recommended)
            y: Target variable
            n_repeats: Number of times to permute each feature
            random_state: Random seed for reproducibility
        """
        try:
            logger.info(f"Running permutation importance with {n_repeats} repeats")
            
            # Compute permutation importance
            perm_importance = permutation_importance(
                model, 
                X, 
                y,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=-1  # Use all CPU cores
            )
            
            results = []
            for i, name in enumerate(X.columns):
                results.append({
                    "feature_name": name,
                    "importance": float(perm_importance.importances_mean[i]),
                    "std": float(perm_importance.importances_std[i]),
                    "rank": i + 1
                })
            
            # Sort by importance descending
            results.sort(key=lambda x: x['importance'], reverse=True)
            
            # Update ranks after sorting
            for i, result in enumerate(results):
                result['rank'] = i + 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing permutation importance: {e}", exc_info=True)
            return []
