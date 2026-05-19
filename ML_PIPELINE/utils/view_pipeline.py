# view_pipeline.py

import sys

def main():
    if len(sys.argv) < 2:
        print("\nPipeline Viewer - View all pipeline components")
        print("\nUsage:")
        print("  python view_pipeline.py datasets          # View all datasets")
        print("  python view_pipeline.py datasets <id>     # View specific dataset")
        print("  python view_pipeline.py models            # View all models")
        print("  python view_pipeline.py models <id>       # View specific model")
        print("  python view_pipeline.py importance        # View feature importance")
        print("  python view_pipeline.py importance <id>   # View for specific model")
        print("  python view_pipeline.py kegg              # View KEGG analyses")
        print("  python view_pipeline.py kegg <id>         # View specific analysis")
        print("  python view_pipeline.py experiments       # View all experiments")
        print("  python view_pipeline.py experiments <id>  # View specific experiment")
        print()
        return
    
    component = sys.argv[1]
    
    if component == "datasets":
        import view_datasets
        view_datasets.main()
    elif component == "models":
        import view_models
        view_models.main()
    elif component == "importance":
        import view_feature_importance
        view_feature_importance.main()
    elif component == "kegg":
        import view_kegg_analyses
        view_kegg_analyses.main()
    elif component == "experiments":
        import view_experiments
        view_experiments.main()
    else:
        print(f"Unknown component: {component}")
        print("Valid components: datasets, models, importance, kegg, experiments")

if __name__ == "__main__":
    main()
