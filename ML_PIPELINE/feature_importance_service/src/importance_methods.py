from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.inspection import permutation_importance

import pandas as pd
from typing import List, Dict
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
                    "rank": int(ranking)
                    #"selected": bool(selected)
                })
            
            # Sort by rank (lower rank = more important)
            results.sort(key=lambda x: x['rank'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing RFE: {e}", exc_info=True)
            return []
    
    @staticmethod
    def sequential_feature_selection(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_features_to_select: int = None,
        direction: str = 'forward',
        cv: int = 5
    ) -> List[Dict]:
        """
        Sequential Feature Selection (SFS)
        
        Iteratively adds (forward) or removes (backward) features based on 
        cross-validated estimator performance. Works with any sklearn model.
        
        Args:
            model: Sklearn model to use for selection
            X: Feature matrix
            y: Target variable
            n_features_to_select: Number of features to select (default: half)
            direction: 'forward' (add features) or 'backward' (remove features)
            cv: Number of cross-validation folds
        
        Returns:
            List of dicts with feature name, importance score, rank, and selection order
        """
        try:
            if n_features_to_select is None:
                n_features_to_select = max(1, len(X.columns) // 2)
            
            if direction not in ['forward', 'backward']:
                raise ValueError("direction must be 'forward' or 'backward'")
            
            logger.info(f"Running SFS ({direction}): selecting {n_features_to_select} features")
            
            # Create Sequential Feature Selector
            sfs = SequentialFeatureSelector(
                estimator=model,
                n_features_to_select=n_features_to_select,
                direction=direction,
                cv=cv,
                n_jobs=-1,  # Use all CPU cores
                scoring=None  # Use estimator's default scoring
            )
            
            # Fit SFS
            sfs.fit(X, y)
            
            results = []
            
            # Get feature names that were selected
            selected_features = X.columns[sfs.get_support()].tolist()
            
            # Assign ranks based on selection order
            # Features selected earlier (in forward) or later (in backward) are more important
            for i, name in enumerate(X.columns):
                is_selected = name in selected_features
                
                if is_selected:
                    # Selected features ranked by order
                    rank = selected_features.index(name) + 1
                    importance = 1.0 / rank  # Earlier selection = higher importance
                else:
                    # Non-selected features ranked after selected ones
                    rank = len(selected_features) + 1
                    importance = 0.0
                
                results.append({
                    "feature_name": name,
                    "importance": float(importance),
                    "rank": int(rank)
                    #"selected": bool(is_selected)
                })
            
            # Sort by rank
            results.sort(key=lambda x: x['rank'])
            
            logger.info(f"SFS completed: selected {len(selected_features)} features")
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing Sequential Feature Selection: {e}", exc_info=True)
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
                    #"std": float(perm_importance.importances_std[i]),
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
