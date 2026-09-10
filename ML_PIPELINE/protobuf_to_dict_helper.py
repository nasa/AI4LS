"""
Helper function to convert protobuf objects to dictionaries
"""

def convert_importance_response_to_dict(importance_response):
    """
    Convert protobuf ImportanceResponse to a Python dict
    
    Args:
        importance_response: feature_importance_service_pb2.ImportanceResponse
    
    Returns:
        Dict with feature importance data
    """
    if not importance_response or not importance_response.success:
        return {}
    
    result = {
        'model_id': importance_response.model_id,
        'success': importance_response.success
    }
    
    # Convert importances map (protobuf map)
    importances_dict = {}
    
    for method_name, feature_importances in importance_response.importances.items():
        # Each feature_importances is a FeatureImportances object with:
        # - scores: repeated FeatureScore
        # - metadata: map of strings
        
        features_list = []
        for score in feature_importances.scores:
            feature_dict = {
                'feature_name': score.feature_name,
                'importance': float(score.importance),
                'rank': score.rank
            }
            features_list.append(feature_dict)
        
        importances_dict[method_name] = {
            'features': features_list,
            'metadata': dict(feature_importances.metadata)
        }
    
    result['importances'] = importances_dict
    return result
