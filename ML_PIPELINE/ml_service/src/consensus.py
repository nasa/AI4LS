from typing import List
from typing import Dict

def compute_consensus_features(
    feature_importance_results: List[Dict],
    top_n: int = 100,
    consensus_threshold: int = 3
) -> Dict:
    """
    Aggregate feature importance across multiple models to find consensus features.
    
    Args:
        feature_importance_results: List of feature importance dicts from different models
        top_n: Number of top features to consider from each model
        consensus_threshold: Minimum number of models a feature must appear in
    
    Returns:
        Dict with consensus features and their statistics
    """
    from collections import defaultdict
    import numpy as np
    from typing import List
    
    # Track which models selected each feature
    feature_selections = defaultdict(list)  # feature_name -> [model_ids]
    feature_scores = defaultdict(list)      # feature_name -> [importance_scores]
    feature_ranks = defaultdict(list)       # feature_name -> [ranks in different models]
    
    # Process each model's results
    for result in feature_importance_results:
        model_id = result['model_id']
        algorithm = result.get('algorithm', 'unknown')
        features = result.get('features', [])
        
        # Take top N features from this model
        top_features = features[:top_n]
        
        for rank, feature_dict in enumerate(top_features, 1):
            feature_name = feature_dict['feature']
            importance = feature_dict['importance']
            
            feature_selections[feature_name].append(model_id)
            feature_scores[feature_name].append(importance)
            feature_ranks[feature_name].append(rank)
    
    # Find consensus features (appear in >= consensus_threshold models)
    consensus_features = []
    
    for feature_name, model_ids in feature_selections.items():
        num_models = len(model_ids)
        
        if num_models >= consensus_threshold:
            consensus_features.append({
                'feature': feature_name,
                'num_models': num_models,
                'models': model_ids,
                'avg_importance': float(np.mean(feature_scores[feature_name])),
                'std_importance': float(np.std(feature_scores[feature_name])),
                'avg_rank': float(np.mean(feature_ranks[feature_name])),
                'best_rank': int(min(feature_ranks[feature_name])),
                'worst_rank': int(max(feature_ranks[feature_name]))
            })
    
    # Sort by number of models (descending), then by average rank (ascending)
    consensus_features.sort(key=lambda x: (-x['num_models'], x['avg_rank']))
    
    return {
        'consensus_features': consensus_features,
        'num_consensus': len(consensus_features),
        'total_models': len(feature_importance_results),
        'top_n_per_model': top_n,
        'consensus_threshold': consensus_threshold,
        'summary': {
            'perfect_consensus': len([f for f in consensus_features if f['num_models'] == len(feature_importance_results)]),
            'high_consensus': len([f for f in consensus_features if f['num_models'] >= len(feature_importance_results) * 0.8]),
            'medium_consensus': len([f for f in consensus_features if f['num_models'] >= consensus_threshold])
        }
    }
