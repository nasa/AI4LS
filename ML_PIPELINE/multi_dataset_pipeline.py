#!/usr/bin/env python3
"""
Multi-Dataset Pipeline Runner
Extends run_docker_pipeline.py to support combining multiple OSD datasets
"""

import sys
import argparse
from pathlib import Path
from utils.multi_dataset_combiner import MultiDatasetCombiner, TISSUE_REGISTRY

# Import the original pipeline
import importlib.util
spec = importlib.util.spec_from_file_location("run_docker_pipeline", "./run_docker_pipeline.py")
pipeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_module)

from orchestration_service.src.clients.data_client import DataServiceClient


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
    **kwargs
):
    """
    Combine multiple datasets and run the pipeline
    """
    
    print("=" * 80)
    print("MULTI-DATASET PIPELINE RUNNER")
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
    
    if trans_list is None:
        trans_list = []
    
    if fi_methods is None:
        fi_methods = ["built_in"]
    
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
    
    data_client = pipeline_module.get_data_client()
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
    
    # Step 4: Prepare combined dataset for upload
    print("\n" + "=" * 80)
    print("STEP 4: PREPARE COMBINED DATASET")
    print("=" * 80)
    
    # Create a clean copy for upload (no metadata columns)
    df_for_upload = combined_df.copy()
    
    # Remove metadata columns
    if 'source_dataset' in df_for_upload.columns:
        df_for_upload = df_for_upload.drop(columns=['source_dataset'])
    
    # Ensure condition column has proper string values
    condition_cols = [col for col in df_for_upload.columns if 'Factor' in col or 'Condition' in col]
    if condition_cols:
        cond_col = condition_cols[0]
        # Convert 0/1 to string values if needed
        if df_for_upload[cond_col].dtype in ['int64', 'float64']:
            if set(df_for_upload[cond_col].unique()) <= {0, 1}:
                print(f"Converting {cond_col} from 0/1 to string values")
                df_for_upload[cond_col] = df_for_upload[cond_col].map({
                    0: factor_values[0],
                    1: factor_values[1]
                })
    
    combined_path = "./combined_dataset_temp.parquet"
    df_for_upload.to_parquet(combined_path)
    print(f"✓ Saved combined dataset for upload: {combined_path}")
    
    # Step 5: Upload combined dataset to pipeline
    print("\n" + "=" * 80)
    # Step 5: Save combined dataset directly
    print("\n" + "=" * 80)
    print("STEP 5: SAVE COMBINED DATASET")
    print("=" * 80)

    try:
        # Generate a unique dataset_id for the combined dataset
        import uuid
        dataset_id = str(uuid.uuid4())
        dataset_path = Path("./datasets") / f"{dataset_id}.parquet"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        df_for_upload.to_parquet(dataset_path)
        print(f"✓ Combined dataset saved: {dataset_id}")
        print(f"  Path: {dataset_path}")
        columns = list(df_for_upload.columns)
    except Exception as e:
        print(f"✗ Failed to save dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Step 6: Run the rest of the pipeline
    print("=" * 80)
    
    try:
        # Get target column from the combined dataset
        if target_column is None:
            condition_cols = [col for col in combined_df.columns if 'Factor' in col or 'Condition' in col]
            if condition_cols:
                target_column = condition_cols[0]
            else:
                raise ValueError("Could not determine target column")
        
        print(f"Target column: {target_column}")
        print(f"Task type: {task_type}")
        print(f"Algorithm: {algorithm}")
        
        # Run the main pipeline
        model_id, metrics = pipeline_module.run_pipeline(
            dataset_id=dataset_id,
            target_column=target_column,
            sample_column=None,
            columns=columns,
            task_type=task_type,
            algorithm=algorithm,
            test_size=test_size,
            trans_list=trans_list,
            factor_name=factor_name,
            factor_values=factor_values,
            min_features=min_features,
            fi_methods=fi_methods,
            exclude_columns=["source_dataset"],
            cv_step=cv_step
        )
        
        if model_id:
            print(f"\n✓ Pipeline completed successfully!")
            print(f"Model ID: {model_id}")
            if metrics:
                print(f"Metrics: {metrics}")
            return model_id
        else:
            print("\n✗ Pipeline failed")
            return None
            
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Dataset Pipeline Runner - Combine OSD datasets by tissue or ID"
    )
    
    # Multi-dataset options (mutually exclusive)
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
    parser.add_argument('-tt', '--task_type', help='classification|regression', default='classification', required=False)
    parser.add_argument('-al', '--algorithm', help='name of ML algorithm', default='random_forest', required=False)
    parser.add_argument('-ts', '--test_size', help='decimal amount of data for testing', type=float, default=0.2, required=False)
    parser.add_argument('-tc', '--target_column', help='name of target column', default=None, required=False)
    parser.add_argument('-fl', '--factor_name', help='metadata factor name', default='Factor Value[Spaceflight]', required=False)
    parser.add_argument('-fv', '--factor_values', type=str, help='comma-separated factor values', default='Ground Control,Space Flight', required=False)
    parser.add_argument('-pa', '--patterns', type=str, help='comma-separated patterns to match', default='unnormalized,RSEM', required=False)
    parser.add_argument('-tl', '--trans_list', type=str, help='comma-separated transformations', default='', required=False)
    parser.add_argument('-cs', '--cv_step', type=float, default=0.25, required=False)
    parser.add_argument('-mf', '--min_features', type=int, default=1000, required=False)
    parser.add_argument('-fi', '--fi_methods', type=str, help='comma-separated feature importance methods', default='built_in', required=False)
    
    args = parser.parse_args()
    
    # Parse comma-separated values
    factor_values = [v.strip() for v in args.factor_values.split(',')]
    patterns = [p.strip() for p in args.patterns.split(',')]
    trans_list = [t.strip() for t in args.trans_list.split(',')] if args.trans_list else []
    fi_methods = [f.strip() for f in args.fi_methods.split(',')]
    
    # Parse OSD IDs if provided
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
        fi_methods=fi_methods
    )
    
    if result:
        print("\n" + "=" * 80)
        print("✓ MULTI-DATASET PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ MULTI-DATASET PIPELINE FAILED")
        print("=" * 80)
        sys.exit(1)


if __name__ == '__main__':
    main()
