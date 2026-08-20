#!/usr/bin/env python3
"""
Multi-Dataset Pipeline with Proper Filtering, Transformation, and Ensemble
Complete analysis: filter → transform → ensemble training with consensus features
"""

import sys
import argparse
from pathlib import Path
from utils.multi_dataset_combiner import MultiDatasetCombiner, TISSUE_REGISTRY
import pandas as pd
import numpy as np
import logging
import random
import requests
import json
import grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

from orchestration_service.src.clients.data_client import DataServiceClient

from feature_importance_service.generated import feature_importance_service_pb2, feature_importance_service_pb2_grpc

from ml_service.generated import ml_service_pb2_grpc, ml_service_pb2 

def _filter_cvs(df, start=None, step=None, min_features=1000):
    """
    NEW METHOD: Keep exactly top min_features most varied genes
    
    Args:
        df: DataFrame with samples x genes
        min_features: exact number of genes to keep (default: 1000)
        start, step: ignored (kept for backward compatibility)
    
    Returns:
        DataFrame with top min_features genes by coefficient of variation
    """
    logger.info("=" * 80)
    logger.info("FILTERING BY COEFFICIENT OF VARIATION")
    logger.info("=" * 80)
    
    # If we already have fewer genes than min_features, return as-is
    if df.shape[1] <= min_features:
        logger.info(f"Dataset has {df.shape[1]} genes, which is ≤ {min_features}. No filtering needed.")
        return df
    
    logger.info(f"Starting genes: {df.shape[1]}")
    logger.info(f"Target genes: {min_features}")
    
    # Step 1: Remove low-count genes (optional, but good for quality)
    logger.info(f"\nRemoving low-count genes...")
    min_count = df.shape[0] * 50
    numeric_sums = df.sum(numeric_only=True)
    cols_to_keep = numeric_sums[numeric_sums >= min_count].index
    df_filtered = df[cols_to_keep].copy()
    logger.info(f"After removing low-count: {df_filtered.shape[1]} genes")

    # Step 2: Calculate coefficient of variation for all genes
    logger.info(f"\nCalculating coefficient of variation...")
    cv_scores = {}

    for col in df_filtered.columns:
        mean_val = np.mean(df_filtered[col])
        std_val = np.std(df_filtered[col])
        
        # Avoid division by zero
        if mean_val != 0:
            cv = std_val / mean_val
            cv_scores[col] = cv
        else:
            cv_scores[col] = 0

    # Step 3: Sort by CV and select top min_features
    logger.info(f"Selecting top {min_features} genes by CV...")
    sorted_genes = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)
    top_genes = [gene for gene, cv in sorted_genes[:min_features]]
    
    logger.info(f"CV range: {sorted_genes[-1][1]:.6f} - {sorted_genes[0][1]:.6f}")
    logger.info(f"Top gene CV: {sorted_genes[0][0]} = {sorted_genes[0][1]:.6f}")
    logger.info(f"Bottom gene CV (rank {min_features}): {sorted_genes[min_features-1][0]} = {sorted_genes[min_features-1][1]:.6f}")
    
    # Step 4: Return DataFrame with top genes
    df_result = df_filtered[top_genes]
    logger.info(f"\nFiltered dataset: {df_result.shape[0]} samples × {df_result.shape[1]} genes")
    
    return df_result, top_genes

def filter_and_transform_data(df, target_column, cv_step=0.25, min_features=1000, trans_list=None):
    """
    Filter by coefficient of variation and apply transformations
    """
    if trans_list is None:
        trans_list = []
    
    logger.info("=" * 80)
    logger.info("FILTERING AND TRANSFORMING DATA")
    logger.info("=" * 80)
    
    # Separate target from features
    feature_cols = [col for col in df.columns if col != target_column and col != 'source_dataset' ]
    
    X = df[feature_cols]
    y = df[target_column]

    
    logger.info(f"Original features: {len(feature_cols)}")
    
    # Step 1: CV Filtering - Keep exactly top min_features by CV
    logger.info(f"\nApplying CV filtering...")

    X_filtered, selected_features = _filter_cvs(X, start=0.25, step=0.25, min_features=1000)

    print('before trans: ', X_filtered.head())
    
    # Step 2: Apply transformations
    if trans_list:
        logger.info(f"\nApplying transformations: {trans_list}...")
        
        X_transformed = X_filtered.copy()
        
        if 't' in trans_list or 'tpm' in trans_list:
            logger.info("  Applying tpm transformation...")
        
        if 'l' in trans_list or 'log' in trans_list:
            logger.info("  Applying log transformation...")
            # Add pseudocount to avoid log(0)
            X_transformed = np.log2(X_transformed + 1)
        
        if 's' in trans_list or 'std' in trans_list:
            logger.info("  Applying scaling...")
            # Min-max scaling
            X_transformed = (X_transformed - X_transformed.min()) / (X_transformed.max() - X_transformed.min() + 1e-8)
    else:
        logger.info("No transformations specified")
        X_transformed = X_filtered.copy()
    

    print('after trans: ', X_transformed.head())

    # Combine back with target
    df_filtered_transformed = X_transformed.copy()
    df_filtered_transformed[target_column] = df[target_column]
    
    # Add back source_dataset if present
    if 'source_dataset' in df.columns:
        df_filtered_transformed['source_dataset'] = df['source_dataset']
    
    logger.info(f"\nFiltered & Transformed dataset:")
    logger.info(f"  Samples: {len(df_filtered_transformed)}")

    print('after adding back source_dataset: ', df_filtered_transformed.head())
    
    return df_filtered_transformed, selected_features


def combine_and_run_pipeline(
    tissue_name=None,
    osd_ids=None,
    target_column=None,
    factor_name="Factor Value[Spaceflight]",
    factor_values=None,
    patterns=None,
    task_type="classification",
    algorithm="random_forest",
    test_size=0.2,
    trans_list=None,
    cv_step=0.25,
    min_features=1000,
    fi_methods=None,
    do_feature_importance=True,
    do_ensemble=True,
    do_kegg_analysis=True,
    organism_name="mmu",
    pvalue_threshold=0.1,
    qvalue_threshold=0.1,
    kegg_max_genes=500,
    consensus_threshold=3,
    top_features=100,
    **kwargs
):
    """
    Combine multiple datasets, filter, transform, then run ensemble
    """
    
    print("=" * 80)
    print("MULTI-DATASET PIPELINE (Filtered & Transformed)")
    print("=" * 80)
    
    # Validate inputs
    if not tissue_name and not osd_ids:
        raise ValueError("Either --tissue or --osd-ids must be specified")
    
    if tissue_name and osd_ids:
        raise ValueError("Cannot specify both --tissue and --osd-ids")
    
    # Set defaults
    if factor_values is None:
        factor_values = ["Ground Control", "Space Flight"]
    
    if patterns is None:
        patterns = ["unnormalized", "RSEM"]
    
    '''if trans_list is None:
        trans_list = ['t', 's']  # tpm + standardize '''
    
    if fi_methods is None:
        fi_methods = ["built_in"]
    
    # Filter out RFE for neural network algorithms (they don't have feature_importances_)
    if algorithm in ["neural_network", "mlp", "nn"]:
        fi_methods = [m for m in fi_methods if m != "rfe"]
        logger.info(f"Removed RFE from feature importance methods for {algorithm} algorithm")
    
    # Step 1: Get OSD IDs
    print("\n" + "=" * 80)
    print("STEP 1: RESOLVE DATASET IDS")
    print("=" * 80)
    
    combiner = MultiDatasetCombiner(data_client=None)
    
    if tissue_name:
        print(f"Looking up datasets for tissue: {tissue_name}")
        osd_ids = combiner.get_osd_ids_for_tissue(tissue_name)
    else:
        print(f"Using specified OSD IDs: {osd_ids}")
    
    # Step 2: Download datasets
    print("\n" + "=" * 80)
    print("STEP 2: DOWNLOAD DATASETS")
    print("=" * 80)
    
    data_client = get_data_client()
    combiner = MultiDatasetCombiner(data_client=data_client)
    
    try:
        datasets = combiner.download_multiple_datasets(
            osd_ids=osd_ids,
            patterns=patterns,
            factor_name=factor_name,
            factor_values=factor_values
        )
    except Exception as e:
        print(f"✗ Failed to download datasets: {e}")
        return None
    
    # Step 3: Find common genes and combine
    print("\n" + "=" * 80)
    print("STEP 3: COMBINE DATASETS")
    print("=" * 80)
    
    try:
        common_genes = combiner.find_common_genes(datasets)
        combined_df, dataset_map = combiner.combine_datasets(datasets, common_genes)
        combiner.print_dataset_summary(combined_df, dataset_map)
    except Exception as e:
        print(f"✗ Failed to combine datasets: {e}")
        return None
    
    # Step 4: Determine target column
    print("\n" + "=" * 80)
    print("STEP 4: PREPARE DATA")
    print("=" * 80)
    
    print(f"Target column: {target_column}")
    
    # Ensure target has string values
    df_clean = combined_df.copy()
    if df_clean[target_column].dtype in ['int64', 'float64']:
        if set(df_clean[target_column].unique()) <= {0, 1}:
            print(f"Converting {target_column} from 0/1 to string values")
            df_clean[target_column] = df_clean[target_column].map({
                0: factor_values[0],
                1: factor_values[1]
            })
    
    # Step 5: Filter and Transform
    print("\n" + "=" * 80)
    print("STEP 5: FILTER AND TRANSFORM DATA")
    print("=" * 80)

    try:
        df_filtered_transformed, selected_genes = filter_and_transform_data(
            df_clean,
            target_column=target_column,
            cv_step=cv_step,
            min_features=min_features,
            trans_list=trans_list
        )
    except Exception as e:
        print(f"✗ Failed to filter/transform data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 6: Save filtered & transformed dataset
    print("\n" + "=" * 80)
    print("STEP 6: SAVE FILTERED & TRANSFORMED DATASET")
    print("=" * 80)
    
    try:
        import uuid
        dataset_id = str(uuid.uuid4())
        dataset_path = Path("./datasets") / f"{dataset_id}.parquet"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove source_dataset for upload
        df_for_upload = df_filtered_transformed.copy()
        if 'source_dataset' in df_for_upload.columns:
            df_for_upload = df_for_upload.drop(columns=['source_dataset'])

        #df_for_upload.to_csv('/Users/jcasalet/Desktop/df.csv', sep=',', index=None)
        
        df_for_upload.to_parquet(dataset_path)
        print(f"✓ Filtered & transformed dataset saved: {dataset_id}")
        print(f"  Samples: {len(df_for_upload)}")
        print(f"  Features: {len([c for c in df_for_upload.columns if c != target_column])}")
        
        columns = list(df_for_upload.columns)
    except Exception as e:
        print(f"✗ Failed to save dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

    # TODO fix combine+transform
    df_for_upload
    
    # Step 7: Train single model
    print("\n" + "=" * 80)
    print("STEP 7: TRAIN SINGLE MODEL")
    print("=" * 80)
    
    try:
        model_id, metrics = run_pipeline(
            dataset_id=dataset_id,
            target_column=target_column,
            sample_column=None,
            columns=columns,
            task_type=task_type,
            algorithm=algorithm,
            test_size=test_size,
            trans_list=[],  # Already transformed
            factor_name=factor_name,
            factor_values=factor_values,
            min_features=len(selected_genes),
            fi_methods=fi_methods,
            exclude_columns=["source_dataset"],
            cv_step=0.0  # Already filtered
        )
        
        if not model_id:
            print("\n✗ Single model training failed")
            return None
        
        print(f"\n✓ Model trained: {model_id}")
        if metrics:
            print(f"✓ Metrics: {metrics}")
        
    except Exception as e:
        print(f"✗ Model training error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 8: Feature Importance (single model)
    if do_feature_importance:
        print("\n" + "=" * 80)
        print("STEP 9: COMPUTE FEATURE IMPORTANCE (Single Model)")
        print("=" * 80)
        
        try:
            try:
                feature_importance_response = compute_feature_importance(
                    model_id=model_id,
                    dataset_id=dataset_id,
                    methods=fi_methods
                )
            except Exception as e:
                # If feature importance fails, log but continue
                logger.warning(f"Feature importance computation failed: {e}")
                logger.warning("Continuing without feature importance...")
                feature_importance_response = None
            
            if feature_importance_response:
                print("✓ Feature importance computed successfully")
                
                # Show top features
                if "built_in" in feature_importance_response.importances:
                    print("\nTop 10 Features (Built-in Importance):")
                    scores = feature_importance_response.importances["built_in"].scores
                    for i, score in enumerate(sorted(scores, key=lambda x: x.importance, reverse=True)[:10]):
                        print(f"  {i+1}. {score.feature_name}: {score.importance:.4f}")
            else:
                print("✗ Feature importance computation failed")
                feature_importance_response = None
        
        except Exception as e:
            print(f"✗ Feature importance error: {e}")
            import traceback
            traceback.print_exc()
            feature_importance_response = None
    else:
        feature_importance_response = None
    
    # Step 9: Ensemble Training with Consensus Features
    if do_ensemble:
        print("\n" + "=" * 80)
        print("STEP 11: ENSEMBLE TRAINING & CONSENSUS FEATURES")
        print("=" * 80)
        
        try:
            ensemble_result = run_ensemble_pipeline(
                dataset_id=dataset_id,
                target_column=target_column,
                factor_values=factor_values,
                top_n=top_features,
                consensus_threshold=consensus_threshold
            )
            
            if ensemble_result:
                print("✓ Ensemble training completed successfully")
                
                # Show consensus features
                print(f"\nConsensus Features ({consensus_threshold}+ models):")
                print(f"  Total consensus features: {len(ensemble_result.get('consensus_features', []))}")
                
                consensus_features = ensemble_result.get('consensus_features', [])
                if consensus_features:
                    print("\n  Top 10 Consensus Features:")
                    for i, feature in enumerate(consensus_features[:10]):
                        models_count = feature.get('num_models', 0)
                        avg_rank = feature.get('avg_rank', 0)
                        avg_importance = feature.get('avg_importance', 0)
                        print(f"    {i+1}. {feature.get('feature', 'N/A')}")
                        print(f"       - Selected by {models_count}/{5} models")
                        print(f"       - Average rank: {avg_rank:.1f}")
                        print(f"       - Average importance: {avg_importance:.4f}")
                
                # Show model performance comparison
                print(f"\n  Individual Model Performance:")
                for model_info in ensemble_result.get('models', []):
                    acc = model_info.get('accuracy', 0)
                    algo = model_info.get('algorithm', 'N/A')
                    print(f"    - {algo}: {acc:.4f}")
            else:
                print("✗ Ensemble training failed")
        
        except Exception as e:
            print(f"✗ Ensemble error: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 11: KEGG Pathway Enrichment Analysis
    if do_kegg_analysis and feature_importance_response:
        print("\n" + "=" * 80)
        print("STEP 12: KEGG PATHWAY ENRICHMENT")
        print("=" * 80)
        
        try:
            kegg_response = run_kegg_analysis(
                feature_importance_response,
                organism=organism_name,
                pvalue_cutoff=pvalue_threshold,
                qvalue_cutoff=qvalue_threshold,
                max_genes=kegg_max_genes,
                method=None,
                min_importance=0.0
            )
            
            if kegg_response:
                print(f"✓ KEGG analysis completed")
                print(f"  Organism: {organism_name}")
                print(f"  P-value cutoff: {pvalue_threshold}")
                print(f"  Q-value cutoff: {qvalue_threshold}")
            else:
                print("✗ KEGG analysis failed")
        
        except Exception as e:
            print(f"✗ KEGG analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 80)
    print("✓ MULTI-DATASET PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nResults Summary:")
    print(f"  Datasets combined: {len(datasets)}")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Raw genes: {len(common_genes)}")
    print(f"  Filtered & transformed genes: {len(selected_genes)}")
    print(f"  Single model ID: {model_id}")
    if metrics:
        print(f"  Single model metrics: {metrics}")
    print(f"  Feature importance: {'✓' if feature_importance_response else '✗'}")
    print(f"  Ensemble analysis: {'✓' if do_ensemble else '✗'}")
    print(f"  KEGG enrichment: {'✓' if do_kegg_analysis else '✗'}")
    
    return model_id


def get_data_client():
    import importlib.util
    # REMOVE experiment_service and ml_service from sys.path temporarily
    paths_to_remove = [p for p in sys.path if 'experiment_service' in p or 'ml_service' in p]
    for path in paths_to_remove:
        sys.path.remove(path)

    # Add orchestration_service to the FRONT of sys.path
    orchestration_path = "/Users/jcasalet/Desktop/CODES/NASA/AI4LS/ML_PIPELINE/orchestration_service"
    sys.path.insert(0, orchestration_path)
    print(f"sys.path after cleanup: {sys.path[:3]}")


    # Load data_client module directly
    data_client_path = orchestration_path + "/src/clients/data_client.py"
    spec = importlib.util.spec_from_file_location("data_client", data_client_path)
    data_client_module = importlib.util.module_from_spec(spec)


    # Now execute the module
    spec.loader.exec_module(data_client_module)

    # Get the class
    DataServiceClient = data_client_module.DataServiceClient

    # CREATE the data_client object
    #data_client = DataServiceClient(service_url="data_service:50051")
    data_client = DataServiceClient(service_url="localhost:50051")

    return data_client

def run_pipeline(dataset_id, target_column, sample_column, columns, task_type, algorithm, test_size, trans_list, factor_name, factor_values, min_features, fi_methods, exclude_columns, cv_step):
    """Run full ML pipeline"""
    print("\n" + "=" * 60)
    print("STEP 8: Run ML Pipeline")
    print("=" * 60)

    # remove sample and target from features
    if target_column in columns:
        columns.remove(target_column)

    transformations = []
    if 'l' in trans_list:
        transformations.append({"type": "log", "columns": columns, "params": {}})
    if 'n' in trans_list:
        transformations.append({"type": "normalize", "columns": columns, "params": {}})
    if 's' in trans_list:
        transformations.append({"type": "standardize", "columns": columns, "params": {}})
    if 't' in trans_list:
        transformations.append({"type": "tpm", "columns": columns, "params": {}})

    payload = {
        "dataset_id": dataset_id,
        "config": {
            "target_column": target_column,
            "task_type": task_type,
            "feature_columns": [],
            "transformations": transformations,
            "algorithm": algorithm,
            "hyperparameters": {},
            "metrics": ["accuracy", "f1_score"],
            "test_size": test_size,
            #"random_state": 42,
            "random_state": random.randint(0, 100), 
            "factor_name": factor_name,
            "factor_values": factor_values,
            "min_features": min_features,
            "fi_methods": fi_methods,
            "exclude_columns": exclude_columns,
            "cv_step": cv_step,
        }
    }

    print("\nConfiguration:")
    print(f"  Algorithm: {payload['config']['algorithm']}")
    print(f"  Target: {payload['config']['target_column']}")
    print(f"  Features: All except target")
    print(f"  Test size: {payload['config']['test_size']}")

    print("\nStreaming progress:")
    print("-" * 60)

    response = requests.post(
        f"{BASE_URL}/api/pipeline/run",
        json=payload,
        stream=True
    )

    model_id = None
    final_metrics = None

    for line in response.iter_lines():
        if line:
            progress = json.loads(line)

            status = progress.get('status', 'unknown')
            message = progress.get('message', '')
            percent = progress.get('progress_percent', 0)

            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)

            print(f"[{bar}] {percent:3d}% | {status:12s} | {message}")

            if progress.get('test_metrics'):
                final_metrics = progress['test_metrics']

            if progress.get('model_id'):
                model_id = progress['model_id']

            if progress.get('error'):
                print(f"  ❌ Error: {progress['error']}")
                return None, None

    print("-" * 60)
    return model_id, final_metrics

def compute_feature_importance(model_id, dataset_id, methods=['built_in']):
    """Compute feature importance for a trained model"""
    import grpc
    import sys

    # Import ModelStore to get feature count
    ml_service_path = Path(__file__).parent / "ml_service"
    sys.path.insert(0, str(ml_service_path))
    #from ml_service.generated import ml_service_pb2, ml_service_pb2_grpc
    from src.model_store import ModelStore  # ← ADD THIS
    
    channel = grpc.insecure_channel('localhost:50053')
    stub = feature_importance_service_pb2_grpc.FeatureImportanceServiceStub(channel)
    
    print("\n" + "=" * 60)
    print("STEP 10: COMPUTE FEATURE IMPORTANCE")
    print("=" * 60)
    
    print(f"Computing feature importance for model: {model_id}")
    print(f"Methods: {methods}")
    
    # Set default parameters for each method
    params = {}
    
    
    if 'permutation' in methods:
        params['n_repeats'] = '10'
        params['random_state'] = '42'
        #params['random_state'] = random.randint(0,100) 

    if 'recursive' in methods:
        # Get number of features to calculate better defaults
        model_store = ModelStore(base_path="./models")
        model_info = model_store.get_model_info(model_id)
        num_features = len(model_info.get('feature_columns', []))
        step = max(1, num_features // 10)
        n_features_to_select = min(100, max(10, num_features // 10))
        params['step'] = str(step)
        params['n_features_to_select'] = str(n_features_to_select)
    
    request = feature_importance_service_pb2.ImportanceRequest(
        model_id=model_id,
        dataset_id=dataset_id,
        methods=methods,
        params=params
    ) 
    response = stub.ComputeImportance(request)
    
    print(f"Success: {response.success}")
    
    if response.success:
        print(f"Computed importance using methods: {list(response.importances.keys())}")
        
        # Show summary for each method
        for method_name, importances in response.importances.items():
            num_features = len(importances.scores)
            print(f"\n{method_name.upper()}:")
            print(f"  Total features: {num_features}")
            
            if num_features > 0:
                top_5 = list(importances.scores)[:5]
                print(f"  Top 5 features:")
                for i, score in enumerate(top_5, 1):
                    print(f"    {i}. {score.feature_name}: {score.importance:.6f}")
    else:
        print(f"Error: {response.error_message}")
    
    return response

def run_kegg_analysis(feature_importance_response, organism="mmu", pvalue_cutoff=0.05, qvalue_cutoff=0.2, max_genes=500, method=None, min_importance=0.0):  # ← method is now optional
    """Run KEGG pathway enrichment on important features from ML model"""
    import grpc
    import sys
    from pathlib import Path
    
    # Lazy import to avoid path conflicts
    _bio_path = str(Path(__file__).parent / "bioinformatics_service")
    sys.path.insert(0, _bio_path)
    try:
        #from generated import bioinformatics_service_pb2, bioinformatics_service_pb2_grpc
        from bioinformatics_service.generated import bioinformatics_service_pb2, bioinformatics_service_pb2_grpc
    finally:
        sys.path.remove(_bio_path)
    
    channel = grpc.insecure_channel('localhost:50054')
    stub = bioinformatics_service_pb2_grpc.BioinformaticsServiceStub(channel)
    
    print("\n" + "=" * 60)
    print("KEGG PATHWAY ENRICHMENT")
    print("=" * 60)
    
    # Extract model_id to use as analysis_id
    model_id = feature_importance_response.model_id
    print(f"Model ID: {model_id}")
    
    # Get available methods
    if not hasattr(feature_importance_response, 'importances'):
        print("ERROR: Response doesn't have importances")
        return None
    
    available_methods = list(feature_importance_response.importances.keys())
    print(f"Available methods: {available_methods}")
    
    if len(available_methods) == 0:
        print("ERROR: No feature importance methods computed")
        return None
    
    # Use specified method or first available
    if method is None:
        method = available_methods[0]
        print(f"No method specified, using: {method}")
    elif method not in available_methods:
        print(f"ERROR: Method '{method}' not found")
        print(f"Using first available method: {available_methods[0]}")
        method = available_methods[0]
    
    # Get the FeatureImportances object for this method
    feature_importances = feature_importance_response.importances[method]
    
    # Extract scores (list of FeatureScore objects)
    scores = list(feature_importances.scores)
    
    print(f"\nMethod: {method}")
    print(f"Total features: {len(scores)}")
    
    # Filter by minimum importance if specified
    if min_importance > 0:
        filtered_scores = [s for s in scores if s.importance >= min_importance]
        print(f"Features above importance threshold ({min_importance}): {len(filtered_scores)}")
    else:
        filtered_scores = scores
    
    # Sort by importance (descending)
    sorted_scores = sorted(filtered_scores, key=lambda x: x.importance, reverse=True)
    
    # Take top N
    top_scores = sorted_scores[:max_genes]
    gene_list = [s.feature_name for s in top_scores]
    
    if len(gene_list) == 0:
        print("ERROR: No genes passed filters")
        print(f"Try lowering min_importance (current: {min_importance})")
        return None
    
    print(f"Using top {len(gene_list)} genes for enrichment")
    if len(top_scores) > 0:
        print(f"Importance range: {top_scores[0].importance:.6f} to {top_scores[-1].importance:.6f}")
    
    print(f"\nTop 10 genes by importance:")
    for i, score in enumerate(top_scores[:10], 1):
        print(f"  {i}. {score.feature_name}: {score.importance:.6f} (rank {score.rank})")
    
    # Run KEGG enrichment
    print(f"\nRunning KEGG enrichment...")
    print(f"  Analysis ID: {model_id}")
    print(f"  Organism: {organism}")
    print(f"  Number of genes: {len(gene_list)}")
    print(f"  P-value cutoff: {pvalue_cutoff}")
    print(f"  Q-value cutoff: {qvalue_cutoff}")
    
    kegg_request = bioinformatics_service_pb2.KEGGRequest(
        analysis_id=model_id,
        gene_list=gene_list,
        organism=organism,
        pvalue_cutoff=pvalue_cutoff,
        qvalue_cutoff=qvalue_cutoff
    )
    
    kegg_response = stub.RunKEGGEnrichment(kegg_request)
    
    if kegg_response.success:
        print(f"\n✓ KEGG Enrichment Complete!")
        print(f"  Enriched pathways: {kegg_response.results.num_pathways}")
        
        if kegg_response.results.num_pathways > 0:
            print(f"\nTop Enriched KEGG Pathways:")
            for i, pathway in enumerate(kegg_response.results.pathways[:15], 1):
                print(f"\n{i}. {pathway.pathway_id}: {pathway.description}")
                print(f"   P-value: {pathway.pvalue:.2e}, Adjusted p-value: {pathway.padj:.2e}")
                print(f"   Genes in pathway: {pathway.gene_count}/{len(gene_list)}")
                print(f"   Genes: {', '.join(pathway.genes[:5])}{'...' if len(pathway.genes) > 5 else ''}")
            
            print(f"\n📊 Visualization files:")
            print(f"  {kegg_response.results.dotplot_path}")
            print(f"  {kegg_response.results.barplot_path}")
            if hasattr(kegg_response.results, 'conversion_path') and kegg_response.results.conversion_path:
                print(f"  {kegg_response.results.conversion_path}")
        else:
            print("\n  ⚠️  No significantly enriched pathways found")
            print("\n  Possible reasons:")
            print("    - Gene IDs may not be in the correct format (need ENSEMBL or Gene Symbols)")
            print("    - Not enough genes for statistical power")
            print("    - Genes are not involved in well-characterized pathways")
            print("\n  Suggestions:")
            print(f"    - Relax p-value cutoff (current: {pvalue_cutoff})")
            print(f"    - Relax q-value cutoff (current: {qvalue_cutoff})")
            print(f"    - Include more genes (current: {len(gene_list)})")
    else:
        print(f"\n✗ KEGG enrichment failed: {kegg_response.error_message}")
    
    return kegg_response


def run_ensemble_pipeline(dataset_id, target_column, factor_values, 
                          top_n=100, consensus_threshold=3):
    """Run ensemble training and compute consensus features"""

    # Lazy import to avoid path conflicts
    _ml_path = str(Path(__file__).parent / "ml_service")
    sys.path.insert(0, _ml_path)
    
    # 1. Train ensemble of models
    ml_channel = grpc.insecure_channel('localhost:50052')
    ml_stub = ml_service_pb2_grpc.MLServiceStub(ml_channel)
    
    ensemble_request = ml_service_pb2.EnsembleRequest(
        dataset_id=dataset_id,
        target_column=target_column,
        algorithms=["random_forest", "svm", "logistic_regression", "neural_network"]

    )
    
    ensemble_response = ml_stub.TrainEnsemble(ensemble_request)
    
    if not ensemble_response.success:
        print(f"✗ Ensemble training failed: {ensemble_response.error_message}")
        return None
    
    print(f"✓ Trained {ensemble_response.num_models} models")
    
    # 2. Compute feature importance for each model
    # Lazy import to avoid path conflicts
    _fi_path = str(Path(__file__).parent / "feature_importance_service")

    fi_channel = grpc.insecure_channel('localhost:50053')
    fi_stub = feature_importance_service_pb2_grpc.FeatureImportanceServiceStub(fi_channel)
    
    feature_importance_results = []
    
    for model_result in ensemble_response.models:
        print(f"Computing importance for {model_result.algorithm} ({model_result.model_id})...")
        
        fi_request = feature_importance_service_pb2.ImportanceRequest(
            model_id=model_result.model_id,
            dataset_id=dataset_id,
            methods=["permutation", "recursive"]
        )
        
        fi_response = fi_stub.ComputeImportance(fi_request)
        if fi_response.success:
            # importances is a map: {"permutation": FeatureImportances}
            if "permutation" in fi_response.importances:
                perm_results = fi_response.importances["permutation"]
        
                # Access the scores from FeatureImportances
                features = [
                    {
                        'feature': score.feature_name,
                        'importance': score.importance,
                        'rank': score.rank
                    }
                    for score in perm_results.scores
                ]
        
                feature_importance_results.append({
                    'model_id': model_result.model_id,
                    'algorithm': model_result.algorithm,
                    'features': features
                })
            else:
                print(f"Warning: No permutation results for {model_result.model_id}")
        else:
            print(f"Failed to compute importance for {model_result.model_id}: {fi_response.error_message}")     
    
    # 3. Compute consensus features
    ml_service_path = Path(__file__).parent / "ml_service"
    sys.path.insert(0, str(ml_service_path))
    from ml_service.src.consensus import compute_consensus_features
    
    consensus_result = compute_consensus_features(
        feature_importance_results,
        top_n=top_n,
        consensus_threshold=consensus_threshold
    )
    
    print(f"\n✓ Consensus Analysis:")
    print(f"  Total models: {consensus_result['total_models']}")
    print(f"  Consensus features: {consensus_result['num_consensus']}")
    print(f"  Perfect consensus: {consensus_result['summary']['perfect_consensus']}")
    print(f"  High consensus: {consensus_result['summary']['high_consensus']}")
    
    print(f"\nTop 10 Consensus Features:")
    for i, feature in enumerate(consensus_result['consensus_features'][:10], 1):
        print(f"  {i}. {feature['feature']}")
        print(f"     - Selected by {feature['num_models']}/{consensus_result['total_models']} models")
        print(f"     - Avg rank: {feature['avg_rank']:.1f} (best: {feature['best_rank']})")
    
    return consensus_result

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Dataset Pipeline with Filtering, Transformation, and Ensemble"
    )
    
    # Multi-dataset options
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        '--tissue',
        type=str,
        help=f"Tissue type to combine (available: {', '.join(TISSUE_REGISTRY.keys())})"
    )
    dataset_group.add_argument(
        '--osd-ids',
        type=str,
        help="Comma-separated list of OSD IDs (e.g., 'OSD-48,OSD-51,OSD-71')"
    )
    
    # Pipeline parameters
    parser.add_argument('-tt', '--task_type', default='classification', help='classification|regression')
    parser.add_argument('-al', '--algorithm', default='random_forest', help='ML algorithm')
    parser.add_argument('-ts', '--test_size', type=float, default=0.2, help='test set fraction')
    parser.add_argument('-tc', '--target_column', default=None, help='target column name', required=True)
    parser.add_argument('-fl', '--factor_name', default='Factor Value[Spaceflight]', help='factor name')
    parser.add_argument('-fv', '--factor_values', default='Ground Control,Space Flight', help='factor values')
    parser.add_argument('-pa', '--patterns', default='unnormalized,RSEM', help='data patterns')
    parser.add_argument('-tl', '--trans_list', default='', help='transformations (t=tpm, l=log, s=stdize)')
    parser.add_argument('-cs', '--cv_step', type=float, default=0.25, help='CV filtering step')
    parser.add_argument('-mf', '--min_features', type=int, default=1000, help='minimum features after filtering')
    parser.add_argument('-fi', '--fi_methods', default='built_in', help='feature importance methods')
    
    # Feature Importance options
    parser.add_argument('--no-feature-importance', action='store_true', help='Skip feature importance')
    
    # Ensemble options
    parser.add_argument('--no-ensemble', action='store_true', help='Skip ensemble training')
    parser.add_argument('--consensus-threshold', type=int, default=3, help='consensus threshold')
    parser.add_argument('--top-features', type=int, default=100, help='top N features per model')
    
    # KEGG options
    parser.add_argument('--no-kegg', action='store_true', help='Skip KEGG enrichment')
    parser.add_argument('--organism', default='mmu', help='organism code')
    parser.add_argument('--pvalue', type=float, default=0.1, help='KEGG p-value cutoff')
    parser.add_argument('--qvalue', type=float, default=0.1, help='KEGG q-value cutoff')
    parser.add_argument('--kegg-max-genes', type=int, default=500, help='max genes for KEGG')
    
    args = parser.parse_args()
    
    # Parse comma-separated values
    factor_values = [v.strip() for v in args.factor_values.split(',')]
    patterns = [p.strip() for p in args.patterns.split(',')]
    trans_list = [t.strip() for t in args.trans_list.split(',')]
    fi_methods = [f.strip() for f in args.fi_methods.split(',')]
    
    # Parse OSD IDs
    osd_ids = None
    if args.osd_ids:
        osd_ids = [id.strip() for id in args.osd_ids.split(',')]
    
    # Run the pipeline
    result = combine_and_run_pipeline(
        tissue_name=args.tissue,
        osd_ids=osd_ids,
        target_column=args.target_column,
        factor_name=args.factor_name,
        factor_values=factor_values,
        patterns=patterns,
        task_type=args.task_type,
        algorithm=args.algorithm,
        test_size=args.test_size,
        trans_list=trans_list,
        cv_step=args.cv_step,
        min_features=args.min_features,
        fi_methods=fi_methods,
        do_feature_importance=not args.no_feature_importance,
        do_ensemble=not args.no_ensemble,
        do_kegg_analysis=not args.no_kegg,
        organism_name=args.organism,
        pvalue_threshold=args.pvalue,
        qvalue_threshold=args.qvalue,
        kegg_max_genes=args.kegg_max_genes,
        consensus_threshold=args.consensus_threshold,
        top_features=args.top_features
    )
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
