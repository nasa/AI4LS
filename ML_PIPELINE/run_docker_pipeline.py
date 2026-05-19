# run_docker_pipeline.py
import requests
import json
import time
import sys
import argparse
import grpc
import sys
from pathlib import Path
import pandas as pd

# Add paths
bio_service_path = Path(__file__).parent / "bioinformatics_service"
sys.path.insert(0, str(bio_service_path))
from bioinformatics_service.generated import bioinformatics_service_pb2, bioinformatics_service_pb2_grpc

fi_service_path = Path(__file__).parent / "feature_importance_service"
sys.path.insert(0, str(fi_service_path))
from feature_importance_service.generated import feature_importance_service_pb2, feature_importance_service_pb2_grpc

ec_service_path = Path(__file__).parent / "experiment_service"
sys.path.insert(0, str(ec_service_path))
from experiment_service.src.experiment_client import ExperimentClient


BASE_URL = "http://localhost:8000"


def run_deseq2(dataset_id, target_column="Factor Value[Spaceflight]", control_group="Ground Control", treatment_group="Space Flight", padj_threshold=0.1, log2fc_threshold=0):
    print("\n" + "=" * 60)
    print("STEP 6: Run DESEQ2")
    print("=" * 60)
    # Connect to service
    channel = grpc.insecure_channel('localhost:50054')
    stub = bioinformatics_service_pb2_grpc.BioinformaticsServiceStub(channel)
    
    # you need a dataset with non-zero count data and conditions
    
    
    padj_threshold=padj_threshold
    log2fc_threshold=log2fc_threshold
    # Run DESeq2
    request = bioinformatics_service_pb2.DESeq2Request(
        dataset_id=dataset_id,
        condition_column=target_column,
        control_group=control_group,
        treatment_group=treatment_group,
        padj_threshold=padj_threshold,
        log2fc_threshold=log2fc_threshold,
    )
    
    print("Running DESeq2 analysis...")
    response = stub.RunDESeq2(request)
    
    if response.success:
        print(f"\n✓ DESeq2 Analysis Complete!")
        print(f"  Analysis ID: {response.analysis_id}")
        print(f"\nResults:")
        print(f"  Total genes: {response.results.num_genes}")
        print(f"  Significant genes: {response.results.num_significant}")
        print(f"  Upregulated: {response.results.num_upregulated}")
        print(f"  Downregulated: {response.results.num_downregulated}")
        
        print(f"\nTop 10 Differentially Expressed Genes:")
        for gene in response.results.differential_genes[:10]:
            if gene.log2_fold_change >= log2fc_threshold and gene.padj <= padj_threshold:
                print(f"  {gene.gene_id}: log2FC={gene.log2_fold_change:.2f}, padj={gene.padj:.2e}")
        
        print(f"\nPlots:")
        print(f"  Volcano plot: {response.results.volcano_plot_path}")
        print(f"  MA plot: {response.results.ma_plot_path}")

        return response


def run_kegg_analysis(feature_importance_response, organism="mmu", pvalue_cutoff=0.05, qvalue_cutoff=0.2, 
                     max_genes=500, method=None, min_importance=0.0):  # ← method is now optional
    """Run KEGG pathway enrichment on important features from ML model"""
    import grpc
    import sys
    sys.path.insert(0, 'bioinformatics_service')
    from generated import bioinformatics_service_pb2, bioinformatics_service_pb2_grpc
    
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

def compute_feature_importance(model_id, dataset_id, methods=['built_in']):
    """Compute feature importance for a trained model"""
    import grpc
    import sys

    # Import ModelStore to get feature count
    ml_service_path = Path(__file__).parent / "ml_service"
    sys.path.insert(0, str(ml_service_path))
    from ml_service.generated import ml_service_pb2, ml_service_pb2_grpc
    from src.model_store import ModelStore  # ← ADD THIS
    
    channel = grpc.insecure_channel('localhost:50053')
    stub = feature_importance_service_pb2_grpc.FeatureImportanceServiceStub(channel)
    
    print("\n" + "=" * 60)
    print("STEP 5: COMPUTE FEATURE IMPORTANCE")
    print("=" * 60)
    
    print(f"Computing feature importance for model: {model_id}")
    print(f"Methods: {methods}")
    
    # Set default parameters for each method
    params = {}
    
    if 'permutation' in methods:
        params['n_repeats'] = '10'
        params['random_state'] = '42'

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

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("STEP 1: Health Check")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/health")
    result = response.json()
    print(json.dumps(result, indent=2))

    if result['status'] != 'healthy':
        print("\n⚠️  Services not healthy!")
        return False

    print("\n✓ All services healthy")
    return True


def download_dataset(args):
    """Download dataset from NASA OSDR via the data-service"""
    print("\n" + "=" * 60)
    print("STEP 2: Download Dataset from NASA OSDR")
    print("=" * 60)

    print(f"  OSD ID   : OSD-{args.osd_id}")
    print(f"  Patterns : {args.patterns}")
    print(f"  Factor Name : {args.factor_name}")
    print(f"  Factor Values : {args.factor_values}")

    payload = {
        "osd_id": args.osd_id,
        "patterns": args.patterns,
        "exclude_columns": args.exclude_columns,
        "factor_name": args.factor_name,
        "factor_values": args.factor_values,
        "min_features": args.min_features,
        "cv_step": args.cv_step,
    }

    response = requests.post(f"{BASE_URL}/api/datasets/download", json=payload)

    if response.status_code != 200:
        print(f"\n❌ Download request failed: HTTP {response.status_code}")
        print(response.text)
        return None, None

    result = response.json()
    print(f"Valid: {result['is_valid']}")

    if result.get('errors'):
        print(f"Errors: {result['errors']}")

    if result.get('warnings'):
        print(f"Warnings: {result['warnings']}")

    columns = []
    if result.get('dataset_info'):
        dataset_id = result['dataset_info']['dataset_id']
        print(f"Dataset ID: {dataset_id}")
        print(f"Shape: {result['dataset_info']['num_rows']} x {result['dataset_info']['num_columns']}")
        #print(f"\nColumns:")
        for col in result['dataset_info']['columns']:
            #print(f"  - {col['name']} ({col['dtype']})")
            columns.append(col['name'])

        return dataset_id, columns

    return None, None


def upload_dataset(file_name, exclude_columns, cv_step):
    """Upload and validate dataset from local file"""
    print("\n" + "=" * 60)
    print("STEP 2: Upload and Validate Dataset")
    print("=" * 60)

    with open(file_name, 'rb') as f:
        #files = {'file': ('data.csv', f, 'text/csv')}
        files = {'file': ('data.csv', f, 'csv')}
        response = requests.post(f"{BASE_URL}/api/datasets/validate", files=files)

    result = response.json()
    print(f"Valid: {result['is_valid']}")

    columns = []
    if result['dataset_info']:
        dataset_id = result['dataset_info']['dataset_id']
        print(f"Dataset ID: {dataset_id}")
        print(f"Shape: {result['dataset_info']['num_rows']} x {result['dataset_info']['num_columns']}")
        #print(f"\nColumns:")
        for col in result['dataset_info']['columns']:
            #print(f"  - {col['name']} ({col['dtype']})")
            columns.append(col['name'])

        return dataset_id, columns

    return None, None

def run_pipeline(dataset_id, target_column, sample_column, columns, task_type, algorithm, test_size, trans_list, factor_name, factor_values, min_features, fi_methods, exclude_columns, cv_step):
    """Run full ML pipeline"""
    print("\n" + "=" * 60)
    print("STEP 3: Run ML Pipeline")
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
            "random_state": 42,
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


def get_model_info(model_id):
    """Get model information"""
    print("\n" + "=" * 60)
    print("STEP 4: Get Model Information")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/api/models/{model_id}")
    result = response.json()

    print(f"Model ID: {result['model_id']}")
    print(f"Algorithm: {result['algorithm']}")
    print(f"Task Type: {result['task_type']}")
    print(f"Target: {result['target_column']}")
    print(f"Features: {len(result['feature_columns'])} columns")
    print(f"Samples: {result['num_samples']}")

    print("\nTest Metrics:")
    for metric, value in result['test_metrics'].items():
        print(f"  {metric}: {value:.4f}")


def list_models():
    """List all models"""
    print("\n" + "=" * 60)
    print("STEP 5: List All Models")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/api/models?limit=10")
    result = response.json()

    print(f"Total models: {result['total_count']}")

    if result['models']:
        print("\nModels:")
        for i, model in enumerate(result['models'], 1):
            acc = model['test_metrics'].get('accuracy', 'N/A')
            print(f"{i}. {model['model_id']} | {model['algorithm']} | Acc: {acc}")

def list_of_strings(arg):
    return arg.split(",")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-op', '--operation', help='download|upload', default=None, required=True)
    parser.add_argument('-tt', '--task_type', help='classification|regression', default=None, required=True)
    parser.add_argument('-al', '--algorithm', help='name of ML algorithm', default=None, required=True)
    parser.add_argument('-ts', '--test_size', help='decimal amount of data for testing', default=0.2, required=False)
    parser.add_argument('-if', '--input_file', help='custom file to provide', default=None, required=False)
    parser.add_argument('-oi', '--osd_id', help='OSDR dataset ID', default=None, required=False)
    parser.add_argument('-tl', '--trans_list', help='list of transformations', default=[], type=list_of_strings, required=False)
    parser.add_argument('-ec', '--exclude_columns', help='list of columns to exclude', default=[], type=list_of_strings, required=False)
    parser.add_argument('-cs', '--cv_step', help='float value to start coef of variation dim reduction', default=0.25, type=float, required=False)
    parser.add_argument('-pa', '--patterns', help='string patterns to identify RNA-seq files', default=['unnormalized', 'RSEM'], type=list_of_strings, required=False)
    parser.add_argument('-sc', '--sample_column', help='sample column name', default=None, required=False)
    parser.add_argument('-tc', '--target_column', help='name of target column', default=None, required=False)
    parser.add_argument('-fl', '--factor_name', help='metadata factor name', default='Factor Value[Spaceflight]', required=False)
    parser.add_argument('-fv', '--factor_values', help='metadata factor values', type=list_of_strings, default=['Ground Control', 'Space Flight'], required=False)
    parser.add_argument('-mf', '--min_features', help='minimum number of features to keep after CVS dimensionality reduction', default=1000, required=False)
    parser.add_argument('-fi', '--fi_methods', help='list of feature importance methods to use', type=list_of_strings, default=['recursive', 'permutation', 'built_in'] , required=False)
    parser.add_argument('-rs', '--random_state', help='random state(seed)', type=int, default=42, required=False)
    parser.add_argument('-de', '--dgea', help='do DGEA', type=bool, default=False, required=False)
    parser.add_argument('-pv', '--pvalue_threshold', help='pvalue cutoff', type=float, default=0.1, required=False)
    parser.add_argument('-qv', '--qvalue_threshold', help='FDR cutoff', type=float, default=0.1, required=False)
    parser.add_argument('-fc', '--l2fc_threshold', help='log2 fold change', type=float, default=0.0, required=False)
    parser.add_argument('-ka', '--kegg_analysis', help='do kegg analysis', type=bool, default=False, required=False)


    args = parser.parse_args()


    # Map short names to full method names
    METHOD_ALIASES = {
        'bi': 'built_in',
        'builtin': 'built_in',
        'built_in': 'built_in',
        'pfi': 'permutation',
        'permutation': 'permutation',
        'rfe': 'recursive',
        'recursive': 'recursive'
    }

    # Parse feature importance methods
    fi_methods_input = list(args.fi_methods)
    fi_methods = []
    for method in fi_methods_input:
        method_lower = method.strip().lower()
        if method_lower in METHOD_ALIASES:
            fi_methods.append(METHOD_ALIASES[method_lower])
        else:
            print(f"WARNING: Unknown feature importance method '{method}', skipping")

    if not fi_methods:
        fi_methods = []

    print(f"Feature importance methods: {fi_methods}")
    

    operation = args.operation
    osd_id        = args.osd_id 
    target_column = args.target_column
    sample_column = args.sample_column 
    task_type     = args.task_type 
    algorithm     = args.algorithm 
    test_size     = args.test_size 
    trans_list    = list(args.trans_list)
    exclude_columns    = list(args.exclude_columns)
    cv_step    = float(args.cv_step)
    patterns      = list(args.patterns) 
    input_file = args.input_file
    factor_name = args.factor_name
    factor_values = args.factor_values
    min_features = args.min_features
    do_kegg_analysis = bool(args.kegg_analysis)
    dgea = bool(args.dgea)

    random_state = int(args.random_state)
    pvalue_threshold = float(args.pvalue_threshold) 
    qvalue_threshold = float(args.qvalue_threshold) 
    l2fc_threshold = float(args.l2fc_threshold) 

    if operation == 'upload':
        if input_file is None:
            print("must provide input_file if using upload operation")
            sys.exit(1)
        if target_column is None:
            print("must provide target_column if using upload operation")
            sys.exit(1)
        if sample_column is None:
            print("must provide sample_column if using upload operation")
            sys.exit(1)
    elif operation == 'download':
        if osd_id is None:
            print("must provide osd_id if using download operation")
            sys.exit(1)
        if target_column is None:
            target_column = factor_name
            print(f"  ℹ️  Using factor_name as target_column: {target_column}")


    # Add after argument parsing
    print(f"DEBUG - Arguments:")
    print(f"  operation: {operation}")
    print(f"  osd_id: {osd_id}")
    print(f"  target_column: {target_column}")
    print(f"  task_type: {task_type}")
    print(f"  algorithm: {algorithm}")
    print(f"  transformations: {trans_list}")
    print(f"  factor_name: {factor_name}")
    print(f"  factor_values: {factor_values}")
    print(f"  min_features: {min_features}")
    print()


    # Step 1: Health check
    if not test_health():
        print("\n❌ Services not healthy. Check logs:")
        print("   docker-compose logs")
        return

    time.sleep(2)


    # create experiment client
    experiment_client = ExperimentClient('localhost:50055')
    
    # Create new experiment
    experiment_name = f"Pipeline Run - OSD-{osd_id} - {task_type} - {algorithm}"
    experiment_description = f"Dataset OSD-{osd_id}, Algorithm: {algorithm}, Target: {target_column}"
    
    metadata = {
        "osd_id": str(osd_id),
        "algorithm": algorithm,
        "task_type": task_type,
        "target_column": target_column,
        "pattern": ",".join(patterns)
    }
    experiment_id = experiment_client.create_experiment(
        name=experiment_name,
        description=experiment_description,
        metadata=metadata
    )
    
    if not experiment_id:
        print("Failed to create experiment")
        return
    
    print("\n" + "=" * 60)
    print(f"EXPERIMENT ID: {experiment_id}")
    print("=" * 60)    
     

    try:
        # Update status
        experiment_client.update_experiment(experiment_id, status="running")

        # Step 2: Get dataset (download or upload)
        if operation == 'download':
            dataset_id, columns = download_dataset(args)
        elif operation == 'upload':
            dataset_id, columns = upload_dataset(input_file, exclude_columns, cv_step)
        else:
            print("unknown operation: ", operation)
            sys.exit(1)

        if not dataset_id:
            print("\n❌ Failed to load dataset")
            return

        print(f"\n✓ Dataset ready: {dataset_id}")

        # Update experiment with dataset_id
        experiment_client.update_experiment(experiment_id, dataset_id=dataset_id)

        # Step 3: Run pipeline
        model_id, metrics = run_pipeline(
            dataset_id, target_column, sample_column, 
            columns, task_type, algorithm, test_size, trans_list,
            factor_name, factor_values,min_features, fi_methods,
            exclude_columns, cv_step
        )
        if not model_id:
            print("\n❌ Pipeline failed")
            return

        print(f"\n✓ Model trained: {model_id}")
        if metrics:
            if task_type == 'classification':
                print(f"✓ Test Accuracy: {metrics.get('accuracy', 0):.4f}")
            elif task_type == 'regression':
                print(f"✓ Test RMSE: {metrics.get('rmse', 0):.4f}")

        # Step 4: Get model info
        get_model_info(model_id)

        # Update experiment with model_id
        experiment_client.update_experiment(experiment_id, model_id=model_id)

        # Step 5: Feature Importance
        #if do_feature_importance:
        if len(fi_methods) != 0:
            feature_importance_response = compute_feature_importance(
                model_id=model_id,
                dataset_id=dataset_id,
                methods=fi_methods  # ← Pass the parsed methods
            )
        
            if feature_importance_response:
                experiment_client.update_experiment(
                    experiment_id,
                    feature_importance_id=model_id
                )
    
        # Step 6: KEGG Analysis
        if do_kegg_analysis and feature_importance_response:
            kegg_response = run_kegg_analysis(
                feature_importance_response,
                organism=organism,
                pvalue_cutoff=pvalue_threshold,
                qvalue_cutoff=qvalue_threshold,
                max_genes=500,
                method=None,  # ← Auto-select first available method
                min_importance=0.0
            )


    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        experiment_client.update_experiment(experiment_id, status="failed")
        raise

if __name__ == "__main__":
    main()
