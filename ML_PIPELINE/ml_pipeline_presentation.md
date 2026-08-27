# ML Bioinformatics Pipeline Presentation

## Table of Contents
1. System Overview & Data Flow
2. gRPC + Docker Architecture
3. Data Service
4. ML Service
5. Feature Importance Service
6. Orchestration Service
7. Experiment Service
8. Bioinformatics Service
9. Strengths & Weaknesses
10. Future Directions

---

## 1. System Overview & Data Flow

### Complete Data/Control Flow Diagram

```
                    ┌─────────────────────────────────────┐
                    │         USER / CLIENT               │
                    │  (Web UI, CLI, or API client)       │
                    └───────────────┬─────────────────────┘
                                    │
                                    │ HTTP POST /api/pipeline/run
                                    │ {osd_id, factor_name, algorithms, ...}
                                    │
                                    ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃         ORCHESTRATION SERVICE (Port 8000)                  ┃
        ┃         FastAPI REST API - Coordinates Pipeline           ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
        ┌───────────────────┐  ┌────────────┐  ┌─────────────────┐
        │ Create Experiment │  │ Download   │  │ Filter & Transform│
        │ Record            │  │ Raw Data   │  │ Data Prep        │
        └───────────────────┘  └────────────┘  └─────────────────┘
                    │               │                │
                    ▼               ▼                ▼
        
═══════════════════════════════════════════════════════════════════════════
STEP 1: EXPERIMENT INITIALIZATION
═══════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────┐
    │  Orchestration Service                                           │
    │  ──────────────────────                                          │
    │  1. Generate experiment_id                                       │
    │  2. Call Experiment Service: CreateExperiment()                  │
    │  3. Store parameters: {osd_id, algorithms, cv_step, ...}         │
    └──────────────────────────────────────────────────────────────────┘
                                    │
                                    │ gRPC: CreateExperiment
                                    ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃    EXPERIMENT SERVICE (Port 50055)                   ┃
        ┃    Tracks experiments, parameters, results          ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                    │
                                    │ Saves to: /app/experiments/{exp_id}.json
                                    ▼
                        {
                          "experiment_id": "exp_abc123",
                          "parameters": {...},
                          "status": "running",
                          "datasets": {},
                          "models": [],
                          "results": {}
                        }

═══════════════════════════════════════════════════════════════════════════
STEP 2: DATA ACQUISITION (Raw Counts from NASA)
═══════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────┐
    │  Orchestration Service                                           │
    │  ──────────────────────                                          │
    │  Call Data Service: DownloadDataset()                            │
    │  Parameters: osd_id="137", factor_name="Spaceflight"             │
    └──────────────────────────────────────────────────────────────────┘
                                    │
                                    │ gRPC: DownloadDataset
                                    ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃         DATA SERVICE (Port 50051)                    ┃
        ┃   Data acquisition, storage, transformation          ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            Check Cache     NASA OSDR API    Parse Metadata
            (MD5 hash)      Download Files   Map Conditions
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │  Raw Dataset                      │
                    │  ────────────────                 │
                    │  • 55,536 genes × 18 samples      │
                    │  • Raw integer counts             │
                    │  • Condition column (0/1)         │
                    │  • Saved as Parquet               │
                    │  • dataset_id: raw_uuid           │
                    └───────────────────────────────────┘
                                    │
                                    │ Return dataset_id
                                    ▼
            ┌────────────────────────────────────────────┐
            │  Orchestration: Store raw_dataset_id       │
            │  Update Experiment: datasets.raw = raw_id  │
            └────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
STEP 3A: DATA PREPARATION FOR ML (Filter + Transform)
═══════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────┐
    │  Orchestration Service                                           │
    │  ──────────────────────                                          │
    │  Call Data Service: TransformDataset()                           │
    │  Parameters: raw_dataset_id, cv_step=0.25, min_features=1000     │
    │              transformations=["log", "standardize"]              │
    └──────────────────────────────────────────────────────────────────┘
                                    │
                                    │ gRPC: TransformDataset
                                    ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃         DATA SERVICE (Port 50051)                    ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            Extract Condition   CV Filtering    Apply Transforms
            (save for later)    (55K → 3K)      (log, standardize)
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │  Transformed Dataset              │
                    │  ────────────────────             │
                    │  • 3,179 genes × 18 samples       │
                    │  • Log-transformed                │
                    │  • Standardized (z-score)         │
                    │  • Condition column restored      │
                    │  • Saved as Parquet               │
                    │  • dataset_id: transformed_uuid   │
                    └───────────────────────────────────┘
                                    │
                                    │ Return transformed_dataset_id
                                    ▼
            ┌────────────────────────────────────────────────┐
            │  Orchestration: Store transformed_dataset_id   │
            │  Update Experiment: datasets.transformed       │
            └────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
STEP 3B: DATA PREPARATION FOR DESeq2 (Filter Only, NO Transform)
═══════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────┐
    │  Orchestration Service                                           │
    │  ──────────────────────                                          │
    │  Call Data Service: FilterDataset()                              │
    │  Parameters: raw_dataset_id, cv_step=0.25, min_features=1000     │
    │  (Same filtering as ML, but NO transformations)                  │
    └──────────────────────────────────────────────────────────────────┘
                                    │
                                    │ gRPC: FilterDataset
                                    ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃         DATA SERVICE (Port 50051)                    ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            Extract Condition   CV Filtering    Restore Condition
            (save for later)    (55K → 3K)      (RAW counts!)
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │  Filtered Dataset                 │
                    │  ────────────────                 │
                    │  • 3,179 genes × 18 samples       │
                    │  • RAW COUNTS (not transformed!)  │
                    │  • Same genes as ML dataset       │
                    │  • Condition column restored      │
                    │  • Saved as Parquet               │
                    │  • dataset_id: filtered_uuid      │
                    └───────────────────────────────────┘
                                    │
                                    │ Return filtered_dataset_id
                                    ▼
            ┌────────────────────────────────────────────────┐
            │  Orchestration: Store filtered_dataset_id      │
            │  Update Experiment: datasets.filtered          │
            └────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
STEP 4: ENSEMBLE ML TRAINING (5 Algorithms)
═══════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────┐
    │  Orchestration Service                                           │
    │  ──────────────────────                                          │
    │  Call ML Service: TrainEnsemble()                                │
    │  Parameters: transformed_dataset_id, target="Condition"          │
    │              algorithms=["random_forest", "xgboost", ...]        │
    └──────────────────────────────────────────────────────────────────┘
                                    │
                                    │ gRPC: TrainEnsemble
                                    ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃            ML SERVICE (Port 50052)                   ┃
        ┃       Model training, storage, serving               ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                    │
                                    │ gRPC: StreamDataset(transformed_dataset_id)
                                    ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃         DATA SERVICE (Port 50051)                    ┃
        ┃         Streams dataset in chunks                    ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                    │
                                    │ Streaming chunks (10K rows each)
                                    ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃            ML SERVICE (Port 50052)                   ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            Assemble Dataset   Train/Test Split   Train 5 Models
            (18 × 3179)        (14 train, 4 test) (Sequential)
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
    Random Forest              XGBoost                    SVM
    (n_est=100)             (max_depth=6)              (kernel=rbf)
    Acc: 0.95               Acc: 0.93                  Acc: 0.91
    model_id: rf_123        model_id: xgb_456          model_id: svm_789
            │                       │                       │
            └───────────────────────┴───────────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            │                                               │
            ▼                                               ▼
    Logistic Regression                          Neural Network (MLP)
    (C=1.0, penalty=l2)                          (hidden=(100,50))
    Acc: 0.89                                    Acc: 0.92
    model_id: lr_012                             model_id: mlp_345
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │  Save Models & Metadata           │
                    │  ────────────────────             │
                    │  • model_rf_123.joblib            │
                    │  • model_xgb_456.joblib           │
                    │  • model_svm_789.joblib           │
                    │  • model_lr_012.joblib            │
                    │  • model_mlp_345.joblib           │
                    │                                   │
                    │  Metadata includes:               │
                    │  - algorithm, dataset_id          │
                    │  - feature_names (3179 genes)     │
                    │  - train/test metrics             │
                    │  - created_at timestamp           │
                    └───────────────────────────────────┘
                                    │
                                    │ Return: [model_id, algorithm, metrics] × 5
                                    ▼
            ┌────────────────────────────────────────────────┐
            │  Orchestration: Store model_ids                │
            │  Update Experiment: models = [5 model records] │
            └────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
STEP 5: FEATURE IMPORTANCE (for each of 5 models)
═══════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────┐
    │  Orchestration Service                                           │
    │  ──────────────────────                                          │
    │  FOR EACH model_id in [rf_123, xgb_456, svm_789, lr_012, mlp_345]:│
    │      Call Feature Importance Service: ComputeImportance()        │
    └──────────────────────────────────────────────────────────────────┘
                                    │
                                    │ gRPC: ComputeImportance (5 times, one per model)
                                    ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃   FEATURE IMPORTANCE SERVICE (Port 50053)            ┃
        ┃   Compute permutation importance for models          ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            Load Model          Load Dataset    Compute Permutation
            (from ML Service)   (from Data Svc) Importance (n_repeats=10)
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │  Feature Rankings (per model)     │
                    │  ────────────────────             │
                    │                                   │
                    │  Random Forest:                   │
                    │  1. GENE_A (importance: 0.42)     │
                    │  2. GENE_C (importance: 0.28)     │
                    │  3. GENE_B (importance: 0.19)     │
                    │  ... (3179 features total)        │
                    │                                   │
                    │  XGBoost:                         │
                    │  1. GENE_B (importance: 0.38)     │
                    │  2. GENE_A (importance: 0.31)     │
                    │  3. GENE_C (importance: 0.24)     │
                    │  ...                              │
                    │                                   │
                    │  (Similar for SVM, LR, MLP)       │
                    └───────────────────────────────────┘
                                    │
                                    │ Return: ranked features (5 separate lists)
                                    ▼
            ┌────────────────────────────────────────────────┐
            │  Orchestration: Store all 5 feature rankings   │
            │  feature_importance_results = [...]            │
            └────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
STEP 6: CONSENSUS FEATURE AGGREGATION
═══════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────┐
    │  Orchestration Service (or run_docker_pipeline.py)              │
    │  ──────────────────────                                          │
    │  Call: compute_consensus_features()                              │
    │  Parameters: feature_importance_results (5 lists)                │
    │              top_n=100, consensus_threshold=3                    │
    └──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃     CONSENSUS ALGORITHM (Python function)            ┃
        ┃     ml_service/src/consensus.py                      ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            Collect top 100     Track which      Filter genes in
            from each model     models selected   ≥3 models
                                each gene
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │  Consensus Features               │
                    │  ────────────────                 │
                    │                                   │
                    │  Sort by:                         │
                    │  1. # models that selected it     │
                    │  2. Average rank across models    │
                    │                                   │
                    │  Results:                         │
                    │  1. GENE_A (5/5 models, rank 2.1) │
                    │  2. GENE_B (5/5 models, rank 4.3) │
                    │  3. GENE_C (4/5 models, rank 6.8) │
                    │  4. GENE_D (3/5 models, rank 12.4)│
                    │  ...                              │
                    │  Total: 127 consensus genes       │
                    │                                   │
                    │  Summary:                         │
                    │  - Perfect consensus: 43 genes    │
                    │  - High consensus: 89 genes       │
                    │  - Medium consensus: 127 genes    │
                    └───────────────────────────────────┘
                                    │
                                    │ Return consensus features
                                    ▼
            ┌────────────────────────────────────────────────┐
            │  Orchestration: Store consensus features       │
            │  Update Experiment: consensus_features = [...]  │
            └────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
STEP 7: DIFFERENTIAL EXPRESSION ANALYSIS (DESeq2 in R)
═══════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────┐
    │  Orchestration Service                                           │
    │  ──────────────────────                                          │
    │  Call Bioinformatics Service: RunDESeq2()                        │
    │  Parameters: filtered_dataset_id (RAW COUNTS!)                   │
    │              control_group="Ground Control"                      │
    │              treatment_group="Space Flight"                      │
    └──────────────────────────────────────────────────────────────────┘
                                    │
                                    │ gRPC: RunDESeq2
                                    ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃   BIOINFORMATICS SERVICE (Port 50054)                ┃
        ┃   R-based analyses: DESeq2, KEGG, GO                 ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                    │
                                    │ gRPC: StreamDataset(filtered_dataset_id)
                                    ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃         DATA SERVICE (Port 50051)                    ┃
        ┃         Returns filtered RAW counts                  ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                    │
                                    │ Filtered dataset (3179 genes, RAW counts)
                                    ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃   BIOINFORMATICS SERVICE                             ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            Extract Condition   Transpose to     Call R DESeq2
            Metadata           genes × samples   (via rpy2)
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │  R: DESeq2 Analysis            │
                    │  ─────────────────             │
                    │  library(DESeq2)               │
                    │                                │
                    │  dds <- DESeqDataSet(...)      │
                    │  dds <- DESeq(dds)             │
                    │  res <- results(dds)           │
                    │                                │
                    │  Apply filters:                │
                    │  - padj < 0.05                 │
                    │  - |log2FC| > 1.0              │
                    │                                │
                    │  Generate plots:               │
                    │  - Volcano plot                │
                    │  - MA plot                     │
                    └────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │  Differential Expression Results  │
                    │  ────────────────────             │
                    │                                   │
                    │  Total genes: 3,179               │
                    │  Significant: 847 genes           │
                    │  - Upregulated: 423               │
                    │  - Downregulated: 424             │
                    │                                   │
                    │  Top genes:                       │
                    │  1. GENE_X (log2FC: 3.2, padj: 1e-15)│
                    │  2. GENE_Y (log2FC: -2.8, padj: 2e-12)│
                    │  3. GENE_Z (log2FC: 2.1, padj: 5e-10) │
                    │  ...                              │
                    │                                   │
                    │  Plots saved:                     │
                    │  - volcano_plot.png               │
                    │  - ma_plot.png                    │
                    └───────────────────────────────────┘
                                    │
                                    │ Return DESeq2 results
                                    ▼
            ┌────────────────────────────────────────────────┐
            │  Orchestration: Store DESeq2 results           │
            │  Update Experiment: deseq2_results = {...}     │
            └────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
STEP 8: COMPARISON & FINAL RESULTS
═══════════════════════════════════════════════════════════════════════════

    ┌──────────────────────────────────────────────────────────────────┐
    │  Orchestration Service                                           │
    │  ──────────────────────                                          │
    │  Compute overlap between:                                        │
    │  - ML consensus features (127 genes)                             │
    │  - DESeq2 significant genes (847 genes)                          │
    └──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │  Overlap Analysis                 │
                    │  ────────────────                 │
                    │                                   │
                    │  ML consensus: 127 genes          │
                    │  DESeq2 significant: 847 genes    │
                    │  Overlap (ML ∩ DESeq2): 43 genes  │
                    │                                   │
                    │  These 43 genes are:              │
                    │  ✓ Selected by ≥3 ML algorithms   │
                    │  ✓ Statistically significant in   │
                    │    differential expression        │
                    │  → HIGH CONFIDENCE biomarkers     │
                    │                                   │
                    │  ML-only (84 genes):              │
                    │  - May be important for           │
                    │    classification but not DE      │
                    │                                   │
                    │  DESeq2-only (804 genes):         │
                    │  - Differentially expressed but   │
                    │    not selected by ML             │
                    └───────────────────────────────────┘
                                    │
                                    │ Final results package
                                    ▼
            ┌────────────────────────────────────────────────┐
            │  Update Experiment: Final Results              │
            │  ─────────────────────────────                 │
            │  - overlap_genes: [43 gene IDs]                │
            │  - ml_only_genes: [84 gene IDs]                │
            │  - deseq2_only_genes: [804 gene IDs]           │
            │  - plots: [volcano.png, ma.png]                │
            │  - status: "completed"                         │
            └────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃    EXPERIMENT SERVICE (Port 50055)                   ┃
        ┃    Persists complete experiment record               ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                    │
                                    │ Save to /app/experiments/exp_abc123.json
                                    ▼
                    {
                      "experiment_id": "exp_abc123",
                      "status": "completed",
                      "parameters": {...},
                      "datasets": {
                        "raw": "raw_uuid",
                        "filtered": "filtered_uuid",
                        "transformed": "transformed_uuid"
                      },
                      "models": [5 model records],
                      "consensus_features": 127,
                      "deseq2_results": {...},
                      "overlap": 43
                    }
                                    │
                                    │ HTTP Response
                                    ▼
                    ┌───────────────────────────────────┐
                    │  Return to USER                   │
                    │  ──────────────                   │
                    │                                   │
                    │  {                                │
                    │    "status": "success",           │
                    │    "experiment_id": "exp_abc123", │
                    │    "models_trained": 5,           │
                    │    "consensus_genes": 127,        │
                    │    "deseq2_significant": 847,     │
                    │    "high_confidence": 43,         │
                    │    "plots": [...]                 │
                    │  }                                │
                    └───────────────────────────────────┘
```

### Data Flow Summary

**Input:** User parameters (OSD ID, factor, algorithms, etc.)  
↓  
**Orchestration:** Coordinates all services via gRPC  
↓  
**Data Service:** Download, cache, filter, transform  
↓  
**ML Service:** Train ensemble, save models  
↓  
**Feature Importance:** Rank features per model  
↓  
**Consensus:** Aggregate rankings across models  
↓  
**Bioinformatics:** DESeq2 differential expression  
↓  
**Experiment Service:** Persist all results  
↓  
**Output:** Consensus features + DESeq2 results + overlap analysis

---

### End-to-End Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: DATA ACQUISITION                                                │
└─────────────────────────────────────────────────────────────────────────┘
   User Input: OSD-137, Factor: "Spaceflight", Values: ["Space Flight", "Ground Control"]
        │
        ▼
   ┌──────────────────┐
   │  Data Service    │  Downloads from NASA OSDR
   │   (Port 50051)   │  • Raw count matrix (genes × samples)
   └──────────────────┘  • Metadata with condition labels
        │                 • Saves as: raw_dataset_id
        │
        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Raw Dataset: 55,536 genes × 18 samples + condition column   │
   │ Format: samples as rows, genes as columns, raw counts       │
   └─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2A: PREPARE FOR ML (Filtering + Transformation)                    │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   ┌──────────────────┐
   │  Data Service    │  TransformDataset(raw_dataset_id)
   │  Transform RPC   │  • Extract condition column
   └──────────────────┘  • Apply CV filtering (55K → 3-10K genes)
        │                 • Apply transformations (log, standardize)
        │                 • Add condition back
        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Transformed Dataset: 18 samples × 3,179 genes + condition   │
   │ Format: samples × genes, log-transformed, standardized      │
   │ Saved as: transformed_dataset_id                            │
   └─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2B: PREPARE FOR DESeq2 (Filtering only, NO transformation)        │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   ┌──────────────────┐
   │  Data Service    │  FilterDataset(raw_dataset_id)
   │   Filter RPC     │  • Extract condition column
   └──────────────────┘  • Apply CV filtering (same as ML)
        │                 • NO transformations (keeps raw counts)
        │                 • Add condition back
        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Filtered Dataset: 18 samples × 3,179 genes + condition      │
   │ Format: samples × genes, RAW COUNTS (for DESeq2)            │
   │ Saved as: filtered_dataset_id                               │
   └─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: ENSEMBLE ML TRAINING                                            │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   ┌──────────────────┐
   │   ML Service     │  TrainEnsemble(transformed_dataset_id)
   │   (Port 50052)   │  • Fetch dataset from Data Service (streaming)
   └──────────────────┘  • Train 5 algorithms:
        │                   - Random Forest
        │                   - XGBoost
        │                   - SVM
        │                   - Logistic Regression
        │                   - Neural Network (MLP)
        │                 • Save models with metadata
        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 5 Trained Models (stored as .joblib files)                  │
   │ • model_abc123 (Random Forest)   - Accuracy: 0.95          │
   │ • model_def456 (XGBoost)         - Accuracy: 0.93          │
   │ • model_ghi789 (SVM)             - Accuracy: 0.91          │
   │ • model_jkl012 (Logistic Reg)    - Accuracy: 0.89          │
   │ • model_mno345 (MLP)             - Accuracy: 0.92          │
   └─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: FEATURE IMPORTANCE (for each model)                             │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   ┌───────────────────────┐
   │ Feature Importance    │  ComputeImportance(model_id, dataset_id)
   │    Service            │  • Load model from ML Service
   │   (Port 50053)        │  • Fetch dataset from Data Service
   └───────────────────────┘  • Compute permutation importance
        │                      • Return ranked features
        │
        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Per-Model Feature Rankings (5 separate results)             │
   │                                                              │
   │ Random Forest:        XGBoost:           SVM:               │
   │ 1. GENE_A (0.42)     1. GENE_B (0.38)   1. GENE_A (0.35)   │
   │ 2. GENE_C (0.28)     2. GENE_A (0.31)   2. GENE_D (0.29)   │
   │ 3. GENE_B (0.19)     3. GENE_C (0.24)   3. GENE_B (0.21)   │
   │ ...                  ...                ...                 │
   └─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 5: CONSENSUS FEATURE AGGREGATION                                   │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   ┌──────────────────────┐
   │  Consensus Algorithm │  compute_consensus_features()
   │  (Python function)   │  • Collect top-N from each model (default: 100)
   └──────────────────────┘  • Find genes in ≥ M models (default: 3)
        │                     • Rank by: # models, avg rank, avg importance
        │
        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Consensus Features (Genes selected by multiple models)      │
   │                                                              │
   │ 1. GENE_A - Selected by 5/5 models, avg rank: 2.1          │
   │ 2. GENE_B - Selected by 5/5 models, avg rank: 4.3          │
   │ 3. GENE_C - Selected by 4/5 models, avg rank: 6.8          │
   │ 4. GENE_D - Selected by 3/5 models, avg rank: 12.4         │
   │ ...                                                          │
   └─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 6: DIFFERENTIAL EXPRESSION (DESeq2)                                │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   ┌────────────────────────┐
   │  Bioinformatics        │  RunDESeq2(filtered_dataset_id)
   │      Service           │  • Fetch filtered dataset (raw counts!)
   │   (Port 50054)         │  • Transpose to genes × samples
   └────────────────────────┘  • Extract condition metadata
        │  [R + DESeq2]        • Run DESeq2 differential expression
        │                      • Generate volcano & MA plots
        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Differentially Expressed Genes (DESeq2 Results)             │
   │                                                              │
   │ 1. GENE_X - log2FC: 3.2, padj: 1e-15  (upregulated)        │
   │ 2. GENE_Y - log2FC: -2.8, padj: 2e-12 (downregulated)      │
   │ 3. GENE_Z - log2FC: 2.1, padj: 5e-10                        │
   │ ...                                                          │
   │ Total: 847 significant genes (padj < 0.05, |log2FC| > 1)   │
   └─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 7: EXPERIMENT TRACKING & RESULTS STORAGE                           │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   ┌───────────────────────┐
   │  Experiment Service   │  CreateExperiment(), UpdateExperiment()
   │   (Port 50055)        │  • Store all parameters & results
   └───────────────────────┘  • Track model performance
        │                     • Link datasets, models, features
        │                     • Persist to JSON
        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Experiment Record (experiment_id: exp_abc123)               │
   │                                                              │
   │ Parameters:                                                  │
   │   - Dataset: OSD-137 (Spaceflight vs Ground Control)        │
   │   - Algorithms: RF, XGBoost, SVM, LogReg, MLP               │
   │   - CV filtering: 0.25 step, 1000 min features              │
   │   - Transformations: log, standardize                        │
   │                                                              │
   │ Results:                                                     │
   │   - 5 models trained, avg accuracy: 0.92                    │
   │   - 127 consensus ML features (≥3 models)                   │
   │   - 847 DESeq2 significant genes                            │
   │   - Overlap: 43 genes (ML ∩ DESeq2)                         │
   └─────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

**Input:** OSD ID + Factor Name + Factor Values  
↓  
**Raw Data:** 55K genes × 18 samples (from NASA OSDR)  
↓  
**Branch 1 (ML Path):** Filter + Transform → Ensemble Training → Feature Importance → Consensus  
**Branch 2 (Bio Path):** Filter Only → DESeq2 → Differential Expression  
↓  
**Output:** Consensus ML features + DESeq2 results + Overlap analysis

---

## 2. gRPC + Docker Architecture

### Why Microservices?

**Separation of Concerns:**
- Each service has a single responsibility
- Can be developed, tested, and deployed independently
- Easier to understand and maintain

**Technology Flexibility:**
- Data/ML services: Python (scikit-learn, pandas, XGBoost)
- Bioinformatics service: R (DESeq2, clusterProfiler)
- Each service picks the best tool for the job

**Scalability:**
- Can scale individual services based on load
- Heavy computation (ML training) can get more resources
- Lightweight services (orchestration) stay small

### Service Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                         DOCKER HOST                            │
│                                                                │
│  ┌──────────────┐                                             │
│  │Orchestration │  REST API (FastAPI)                         │
│  │   Service    │  Port: 8000                                 │
│  │              │  Role: Coordinate pipeline, expose HTTP     │
│  └──────┬───────┘                                             │
│         │ gRPC calls                                          │
│         ├─────────────┬──────────────┬──────────────┐         │
│         ▼             ▼              ▼              ▼         │
│  ┌───────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────┐  │
│  │   Data    │ │     ML     │ │ Feature  │ │Bioinformatics│  │
│  │  Service  │ │  Service   │ │Importance│ │   Service    │  │
│  │  :50051   │ │   :50052   │ │  :50053  │ │    :50054    │  │
│  └───────────┘ └──────┬─────┘ └────┬─────┘ └──────────────┘  │
│         │             │            │                          │
│         │             │            │                          │
│         │             ▼            ▼                          │
│         │      ┌─────────────┐  ┌────────────┐               │
│         │      │Model Storage│  │   Model    │               │
│         │      │  (.joblib)  │  │   Store    │               │
│         │      └─────────────┘  └────────────┘               │
│         ▼                                                     │
│  ┌────────────────┐         ┌───────────────┐                │
│  │Dataset Storage │         │  Experiment   │                │
│  │  (.parquet)    │         │    Service    │                │
│  │   + cache      │         │    :50055     │                │
│  └────────────────┘         └───────────────┘                │
│                                                               │
│  ┌─────────────────────────────────────────┐                 │
│  │        Docker Network: ml-pipeline      │                 │
│  │  Services discover each other by name   │                 │
│  └─────────────────────────────────────────┘                 │
└────────────────────────────────────────────────────────────────┘
```

### gRPC Benefits

**1. Type Safety via Protocol Buffers**
```protobuf
message TrainRequest {
    string dataset_id = 1;
    string target_column = 2;
    string algorithm = 3;
}
```
- Compiler-enforced contracts
- Auto-generated client/server code
- No runtime type errors

**2. Performance**
- Binary protocol (vs JSON text)
- HTTP/2 multiplexing
- ~5-10x faster than REST for large datasets

**3. Streaming**
```python
def StreamDataset(self, request, context):
    for chunk in dataset_chunks:
        yield DataChunk(data=chunk)
```
- Efficient for large datasets
- No need to load entire dataset in memory
- Progressive processing on client

**4. Language Interoperability**
- Python ML service calls R bioinformatics service seamlessly
- Same .proto file generates code for Python, R, Go, Java, etc.

### Docker Compose Configuration

```yaml
services:
  data_service:
    build: ./data_service
    ports:
      - "50051:50051"
    volumes:
      - ./datasets:/app/datasets
    networks:
      - ml-pipeline
  
  ml_service:
    build: ./ml_service
    ports:
      - "50052:50052"
    depends_on:
      - data_service
    networks:
      - ml-pipeline
```

**Benefits:**
- One command startup: `docker-compose up`
- Automatic service discovery via DNS
- Volume mounts for persistent storage
- Network isolation

---

## 3. Data Service

### Core Responsibilities

1. **Data Acquisition** - NASA OSDR API integration
2. **Data Storage** - Parquet-based persistence with caching
3. **Data Preparation** - CV filtering, transformations
4. **Data Distribution** - gRPC streaming to other services

### Key RPCs

#### DownloadDataset
**Purpose:** Download raw RNA-seq data from NASA Open Science Data Repository

**Input:**
```python
{
    "osd_id": "137",
    "patterns": ["Unnormalized", "RSEM"],
    "factor_name": "Factor Value[Spaceflight]",
    "factor_values": ["Space Flight", "Ground Control"],
    "exclude_columns": []
}
```

**Process:**
1. Check cache (MD5 hash of download params)
2. If cached → return immediately
3. If not cached:
   - Fetch file list from NASA OSDR API
   - Download count matrix (.csv or .tsv)
   - Download metadata (.zip)
   - Parse and merge metadata
   - Map factor values to numeric labels (0/1)
   - Transpose to samples × genes
   - Save to parquet + update cache

**Output:**
- Dataset ID (UUID)
- Raw counts: samples × genes + condition column
- Cached for future use

**Caching Strategy:**
- Cache key = hash(osd_id, patterns, factor_name, factor_values)
- NO cv_step or min_features in cache key
- Raw data cached once, transformations separate

#### FilterDataset
**Purpose:** Apply CV filtering WITHOUT transformations (for DESeq2)

**Process:**
1. Load raw dataset
2. Extract condition column
3. Apply CV filtering algorithm:
   ```python
   start_cv = 1.0
   while num_features(cv > start_cv) >= min_features:
       start_cv += cv_step
   keep features where cv > (start_cv - cv_step)
   ```
4. Add condition column back
5. Save as new dataset (raw counts preserved)

**Why separate from Transform?**
- DESeq2 needs raw counts (not log-transformed)
- ML and DESeq2 should use same filtered genes
- Allows fair comparison of results

#### TransformDataset
**Purpose:** Prepare data for ML (filtering + transformations)

**Input:**
```python
{
    "dataset_id": "raw_dataset_uuid",
    "transformations": ["log", "standardize"],
    "cv_step": 0.25,
    "min_features": 1000
}
```

**Process:**
1. Load raw dataset
2. Extract condition column
3. Apply CV filtering (same as FilterDataset)
4. Apply transformations sequentially:
   - **log**: np.log1p(X) - handles zeros
   - **standardize**: (X - mean) / std per feature
   - **normalize**: min-max scaling
   - **tpm**: Transcripts Per Million
5. Add condition column back
6. Save as new dataset

**Why these transformations?**
- **log**: Makes data more normal, reduces skew
- **standardize**: Puts features on same scale (important for SVM, LogReg)
- Not needed for tree-based (RF, XGBoost)

#### StreamDataset
**Purpose:** Efficiently transfer large datasets to other services

**Process:**
1. Load dataset from parquet
2. Convert to CSV
3. Split into chunks (default 10K rows)
4. Stream chunks sequentially
5. Client reassembles

**Why streaming?**
- gRPC has message size limits (~4MB)
- Datasets can be 100+ MB
- Memory efficient on both sides

### CV Filtering Algorithm Details

```python
def _filter_cvs(self, df, start=1, step=0.25, min_features=1000):
    """
    Iteratively increase CV threshold until feature count drops below min
    
    CV = coefficient of variation = std / mean
    High CV = high variability = biologically interesting
    """
    if df.shape[1] <= min_features:
        return df  # Already small enough
    
    keep_columns_use = list(df.columns)
    
    while True:
        keep_columns = []
        for col in df.columns:
            mean = np.mean(df[col])
            std = np.std(df[col])
            if mean != 0 and std/mean > start:
                keep_columns.append(col)
        
        if len(keep_columns) < min_features:
            break  # Would drop too many
        else:
            keep_columns_use = keep_columns
            start += step  # Increase threshold
    
    return df[keep_columns_use]
```

**Example:**
- Start: 55,536 genes
- CV > 1.00: 15,000 genes remain
- CV > 1.25: 8,000 genes remain
- CV > 1.50: 3,179 genes remain ← STOP (below 1000 would be too few)
- Final: 3,179 genes with CV > 1.25

### Data Storage

**Format:** Apache Parquet
- Columnar storage (fast column access)
- Compression (10x smaller than CSV)
- Preserves dtypes (no int→float conversion)
- Fast I/O with pandas

**Directory Structure:**
```
/app/datasets/
├── {uuid}.parquet          # Raw datasets
├── {uuid}_filtered_{hash}.parquet  # Filtered datasets
├── {uuid}_transformed_{hash}.parquet  # Transformed datasets
└── download_cache.json     # Cache mapping
```

### NASA OSDR API Integration

**Old API** (before 2024):
```json
{
  "OSD-137": {
    "files": ["file1.csv", "file2.txt"]
  }
}
```

**New API** (current):
```json
{
  "studies": {
    "OSD-137": {
      "study_files": [
        {
          "file_name": "counts.csv",
          "remote_url": "/geode-py/ws/studies/OSD-137/download?file=counts.csv"
        }
      ]
    }
  }
}
```

**Adaptation:**
- Parse nested structure
- Extract `remote_url` instead of constructing URL
- Prepend `https://osdr.nasa.gov` to remote_url

---

## 4. ML Service

### Core Responsibilities

1. **Model Training** - Train classification/regression models
2. **Ensemble Training** - Train multiple algorithms simultaneously
3. **Model Storage** - Persist models with metadata
4. **Model Serving** - Load and provide models to other services

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     ML Service                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐         ┌──────────────────┐         │
│  │  Service     │         │   ModelTrainer   │         │
│  │  (gRPC)      │────────▶│   (Factory)      │         │
│  └──────────────┘         └──────────────────┘         │
│         │                          │                    │
│         │                          ▼                    │
│         │                 ┌─────────────────┐           │
│         │                 │  Sklearn Models │           │
│         │                 ├─────────────────┤           │
│         │                 │ RandomForest    │           │
│         │                 │ XGBoost         │           │
│         │                 │ SVM             │           │
│         │                 │ LogisticReg     │           │
│         │                 │ MLP             │           │
│         │                 └─────────────────┘           │
│         │                                               │
│         ▼                                               │
│  ┌──────────────┐         ┌──────────────────┐         │
│  │ ModelStore   │────────▶│ Persistent       │         │
│  │              │         │ Storage          │         │
│  └──────────────┘         │ (.joblib files)  │         │
│         │                 └──────────────────┘         │
│         ▼                                               │
│  ┌──────────────┐                                      │
│  │ DataClient   │─────────────────────────────────────▶│
│  │  (gRPC)      │   Fetch datasets from Data Service   │
│  └──────────────┘                                      │
└─────────────────────────────────────────────────────────┘
```

### Key RPCs

#### TrainModel
**Purpose:** Train a single model

**Input:**
```python
{
    "dataset_id": "transformed_uuid",
    "target_column": "Factor Value[Spaceflight]",
    "algorithm": "random_forest",
    "task_type": "classification",
    "hyperparameters": {
        "n_estimators": "100",
        "max_depth": "10"
    }
}
```

**Process:**
1. Fetch dataset from Data Service (streaming)
2. Split into train/test (80/20, stratified)
3. Create model via ModelTrainer factory
4. Train model
5. Compute metrics (accuracy, precision, recall, F1)
6. Save model + metadata
7. Return model_id + metrics

#### TrainEnsemble
**Purpose:** Train multiple algorithms on same dataset

**Input:**
```python
{
    "dataset_id": "transformed_uuid",
    "target_column": "Factor Value[Spaceflight]",
    "algorithms": ["random_forest", "xgboost", "svm", "logistic_regression", "neural_network"]
}
```

**Process:**
1. Fetch dataset ONCE (shared across all models)
2. Split ONCE (same train/test for all)
3. For each algorithm:
   - Create model
   - Train on same data split
   - Compute metrics
   - Save model
4. Return list of trained models with metrics

**Benefits:**
- Fair comparison (same data split)
- Efficient (one data fetch)
- Parallel training potential

**Current Algorithms:**
- **Random Forest:** Ensemble of decision trees, handles non-linear relationships
- **XGBoost:** Gradient boosting, often best performance
- **SVM:** Support Vector Machine, good for high-dimensional data
- **Logistic Regression:** Linear model, interpretable
- **Neural Network (MLP):** Multi-layer perceptron, captures complex patterns

### Model Storage

**Storage Format:** joblib
- Python object serialization
- Preserves sklearn models perfectly
- Fast load/save

**Metadata Storage:** JSON
```json
{
  "model_id": "model_abc123",
  "algorithm": "random_forest",
  "task_type": "classification",
  "dataset_id": "transformed_uuid",
  "target_column": "Factor Value[Spaceflight]",
  "feature_columns": ["GENE_001", "GENE_002", ...],
  "num_samples": 18,
  "num_features": 3179,
  "hyperparameters": {},
  "training_metrics": {
    "accuracy": 0.93,
    "precision": 0.92,
    "recall": 0.91,
    "f1_score": 0.91
  },
  "test_metrics": {
    "accuracy": 0.95,
    "precision": 0.94,
    "recall": 0.93,
    "f1_score": 0.94
  },
  "created_at": "2025-05-19T10:30:00"
}
```

### ModelTrainer Factory Pattern

```python
CLASSIFICATION_MODELS = {
    "random_forest": RandomForestClassifier,
    "svm": SVC,
    "logistic_regression": LogisticRegression,
    "xgboost": xgb.XGBClassifier,
    "neural_network": MLPClassifier
}

def create_model(algorithm, task_type, hyperparameters):
    model_class = CLASSIFICATION_MODELS[algorithm]
    return model_class(**hyperparameters)
```

**Benefits:**
- Easy to add new algorithms
- Consistent interface
- Type checking via config

### Data Client Integration

```python
class DataServiceClient:
    def get_dataset(self, dataset_id):
        # Stream dataset in chunks
        request = StreamDatasetRequest(dataset_id=dataset_id)
        
        frames = []
        for chunk in self.stub.StreamDataset(request):
            chunk_df = pd.read_csv(StringIO(chunk.data), index_col=0)
            frames.append(chunk_df)
        
        return pd.concat(frames, ignore_index=False)
```

**Why streaming?**
- Datasets > gRPC message limit
- Memory efficient
- Progressive loading

---

## 5. Feature Importance Service

### Core Responsibility
**Compute feature importance scores for trained models**

### Why Separate Service?

1. **Computation Intensive:** Permutation importance is slow (requires many model predictions)
2. **Optional:** Not all workflows need importance
3. **Reusable:** Can compute importance for any model, anytime
4. **Scalable:** Can run on separate hardware/GPU

### Key RPC: ComputeImportance

**Input:**
```python
{
    "model_id": "model_abc123",
    "dataset_id": "transformed_uuid",
    "methods": ["permutation"],
    "params": {
        "n_repeats": "10",
        "random_state": "42"
    }
}
```

**Methods Supported:**

#### 1. Permutation Importance
**How it works:**
1. Get baseline model score on original data
2. For each feature:
   - Randomly shuffle that feature's values
   - Re-score model
   - Importance = baseline - shuffled_score
3. Repeat N times (n_repeats), average results

**Pros:**
- Works for any model
- Model-agnostic
- Reliable

**Cons:**
- Slow (N_features × N_repeats predictions)
- Can be unstable with correlated features

**Speed:** ~30 seconds for 3K features, 5 repeats

#### 2. Built-in Importance (tree models only)
**How it works:**
- Random Forest: Gini importance from splits
- XGBoost: Gain/cover from tree structure

**Pros:**
- Instant (already computed during training)
- Stable

**Cons:**
- Only works for tree-based models
- Can be biased toward high-cardinality features

**Speed:** <1 second

### Implementation Details

```python
from sklearn.inspection import permutation_importance

def compute_permutation_importance(model, X, y, n_repeats=10):
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    # Sort by importance
    importance_scores = []
    for idx in result.importances_mean.argsort()[::-1]:
        importance_scores.append({
            'feature_name': X.columns[idx],
            'importance': result.importances_mean[idx],
            'std': result.importances_std[idx],
            'rank': len(importance_scores) + 1
        })
    
    return importance_scores
```

### Output Format

```python
{
    "success": True,
    "model_id": "model_abc123",
    "importances": {
        "permutation": {
            "scores": [
                {
                    "feature_name": "ENSMUSG00000001",
                    "importance": 0.42,
                    "rank": 1
                },
                {
                    "feature_name": "ENSMUSG00000045",
                    "importance": 0.28,
                    "rank": 2
                },
                ...
            ],
            "metadata": {
                "execution_time": "28.3s",
                "n_features": "3179"
            }
        }
    }
}
```

### Optimization Strategies

**For debugging/testing:**
1. Reduce n_repeats: 10 → 5 or 3
2. Use built-in for tree models
3. Sample dataset (500 samples instead of all)
4. Parallel processing (n_jobs=-1)

**For production:**
1. Higher n_repeats (10-30) for stability
2. Full dataset
3. Cache results (feature importance rarely changes)

---

## 6. Orchestration Service

### Core Responsibility
**HTTP REST API that coordinates the entire pipeline**

### Why Orchestration?

1. **User-Friendly:** REST API easier than gRPC for external clients
2. **Workflow Management:** Coordinates multiple services in sequence
3. **Error Handling:** Centralized error reporting and retry logic
4. **State Management:** Tracks pipeline progress

### Technology Stack

- **FastAPI:** Modern Python web framework
- **Uvicorn:** ASGI server
- **gRPC Clients:** Connect to all backend services

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Orchestration Service (FastAPI)             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  HTTP Endpoints:                                         │
│  ┌────────────────────────────────────────────────┐     │
│  │ POST /api/pipeline/run                         │     │
│  │ GET  /api/models/{model_id}                    │     │
│  │ GET  /api/experiments/{experiment_id}          │     │
│  │ GET  /health                                   │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
│  gRPC Clients:                                           │
│  ┌────────────┐  ┌────────────┐  ┌───────────────┐     │
│  │DataService │  │ MLService  │  │FeatureImport  │     │
│  │Client      │  │Client      │  │Client         │     │
│  └────────────┘  └────────────┘  └───────────────┘     │
│  ┌────────────┐  ┌────────────┐                         │
│  │Experiment  │  │Bioinf      │                         │
│  │Client      │  │Client      │                         │
│  └────────────┘  └────────────┘                         │
└──────────────────────────────────────────────────────────┘
```

### Main Endpoint: POST /api/pipeline/run

**Input:**
```json
{
  "osd_id": "137",
  "factor_name": "Factor Value[Spaceflight]",
  "factor_values": ["Space Flight", "Ground Control"],
  "algorithms": ["random_forest", "xgboost", "svm"],
  "cv_step": 0.25,
  "min_features": 1000,
  "transformations": ["log", "standardize"]
}
```

**Workflow:**
1. Download raw data (Data Service)
2. Transform for ML (Data Service)
3. Train models (ML Service)
4. Compute feature importance (Feature Importance Service)
5. Create experiment record (Experiment Service)
6. Return results

**Response:**
```json
{
  "status": "success",
  "experiment_id": "exp_abc123",
  "raw_dataset_id": "uuid_raw",
  "transformed_dataset_id": "uuid_transformed",
  "models": [
    {
      "model_id": "model_rf_123",
      "algorithm": "random_forest",
      "accuracy": 0.95
    }
  ],
  "feature_importance_results": {
    "model_rf_123": {
      "top_features": [...]
    }
  }
}
```

### Error Handling

```python
try:
    # Call Data Service
    dataset_response = data_client.download_dataset(...)
    
    if not dataset_response['is_valid']:
        return {
            "status": "error",
            "stage": "data_download",
            "message": dataset_response['errors']
        }
    
    # Call ML Service
    train_response = ml_client.train_model(...)
    
except grpc.RpcError as e:
    return {
        "status": "error",
        "stage": "grpc_communication",
        "message": str(e),
        "service": "ml_service"
    }
```

**Benefits:**
- Clear error messages
- Stage identification
- Easy debugging

### Progress Tracking

```python
async def generate_progress():
    yield {"stage": "downloading", "progress": 0}
    # Download data
    yield {"stage": "downloading", "progress": 100}
    
    yield {"stage": "transforming", "progress": 0}
    # Transform data
    yield {"stage": "transforming", "progress": 100}
    
    yield {"stage": "training", "progress": 0}
    # Train models
    yield {"stage": "training", "progress": 100}
```

**Server-Sent Events (SSE):** Real-time updates to client

---

## 7. Experiment Service

### Core Responsibility
**Track all experiments, parameters, and results for reproducibility**

### Why Track Experiments?

1. **Reproducibility:** Re-run exact same configuration
2. **Comparison:** Compare different hyperparameters/algorithms
3. **Audit Trail:** Know what was tried and when
4. **Collaboration:** Share results with team

### Data Model

```python
{
  "experiment_id": "exp_abc123",
  "created_at": "2025-05-19T10:30:00",
  "status": "completed",
  
  "parameters": {
    "osd_id": "137",
    "factor_name": "Factor Value[Spaceflight]",
    "factor_values": ["Space Flight", "Ground Control"],
    "algorithms": ["random_forest", "xgboost"],
    "cv_step": 0.25,
    "min_features": 1000,
    "transformations": ["log", "standardize"]
  },
  
  "datasets": {
    "raw_dataset_id": "uuid_raw",
    "filtered_dataset_id": "uuid_filtered",
    "transformed_dataset_id": "uuid_transformed"
  },
  
  "models": [
    {
      "model_id": "model_rf_123",
      "algorithm": "random_forest",
      "metrics": {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1_score": 0.94
      }
    }
  ],
  
  "feature_importance": {
    "consensus_features": 127,
    "top_features": ["GENE_A", "GENE_B", ...]
  },
  
  "deseq2_results": {
    "significant_genes": 847,
    "overlap_with_ml": 43
  }
}
```

### Key RPCs

#### CreateExperiment
**Purpose:** Initialize a new experiment record

**Input:** Parameters for the run

**Output:** experiment_id

#### UpdateExperiment
**Purpose:** Add results as pipeline progresses

**Input:**
```python
{
  "experiment_id": "exp_abc123",
  "updates": {
    "models": [...],
    "feature_importance": {...}
  }
}
```

#### GetExperiment
**Purpose:** Retrieve complete experiment details

**Input:** experiment_id

**Output:** Full experiment record

#### ListExperiments
**Purpose:** Browse all experiments

**Filters:**
- By date range
- By algorithm
- By dataset (OSD ID)
- By status (running/completed/failed)

### Storage

**Format:** JSON files

**Location:** `/app/experiments/`

**File naming:** `{experiment_id}.json`

**Benefits:**
- Human-readable
- Easy to grep/search
- Version control friendly
- No database overhead for small scale

---

## 8. Bioinformatics Service

### Core Responsibility
**Run R-based bioinformatics analyses (DESeq2, KEGG enrichment)**

### Why Separate Service?

1. **Different Language:** R, not Python
2. **Specialized Libraries:** Bioconductor packages (DESeq2, clusterProfiler)
3. **Different Data Requirements:** Needs raw counts, not transformed
4. **Independent Analysis:** Can run without ML pipeline

### Technology Stack

- **Base Image:** rocker/r-ver:4.3 (R in Docker)
- **R Packages:** DESeq2, clusterProfiler, enrichplot, org.Mm.eg.db, org.Hs.eg.db
- **Python Bridge:** rpy2 (call R from Python gRPC service)
- **gRPC:** Python server wrapping R functions

### Architecture

```
┌───────────────────────────────────────────────────┐
│        Bioinformatics Service                     │
├───────────────────────────────────────────────────┤
│                                                   │
│  ┌────────────────────┐                          │
│  │  gRPC Service      │  (Python)                │
│  │  BioinformaticsImpl│                          │
│  └─────────┬──────────┘                          │
│            │                                      │
│            │ rpy2                                 │
│            ▼                                      │
│  ┌────────────────────┐                          │
│  │  R Scripts         │                          │
│  ├────────────────────┤                          │
│  │ deseq2_wrapper.R   │  DESeq2 analysis        │
│  │ kegg_enrichment.R  │  Pathway enrichment     │
│  └────────────────────┘                          │
│            │                                      │
│            ▼                                      │
│  ┌────────────────────┐                          │
│  │  Bioconductor      │                          │
│  │  Packages          │                          │
│  │  (DESeq2, etc)     │                          │
│  └────────────────────┘                          │
└───────────────────────────────────────────────────┘
```

### Key RPC: RunDESeq2

**Purpose:** Differential gene expression analysis

**Input:**
```python
{
    "dataset_id": "filtered_uuid",  # RAW COUNTS!
    "condition_column": "Factor Value[Spaceflight]",
    "control_group": "Ground Control",
    "treatment_group": "Space Flight",
    "padj_threshold": 0.05,
    "log2fc_threshold": 1.0
}
```

**Process:**
1. Fetch filtered dataset (RAW counts, not transformed)
2. Extract condition metadata
3. Transpose to genes × samples (DESeq2 format)
4. Call R DESeq2 wrapper
5. Parse results
6. Generate plots (volcano, MA)
7. Return significant genes

**DESeq2 Requirements:**
- **Raw counts:** Integer counts, not normalized
- **Genes as rows, samples as columns**
- **Condition metadata** mapped to samples
- **Replicates:** ≥2 per condition

**Output:**
```python
{
    "success": True,
    "analysis_id": "deseq2_filtered_uuid",
    "results": {
        "num_genes": 55536,
        "num_significant": 847,
        "num_upregulated": 423,
        "num_downregulated": 424,
        "differential_genes": [
            {
                "gene_id": "ENSMUSG00000001",
                "log2_fold_change": 3.2,
                "pvalue": 1e-15,
                "padj": 1e-15,
                "base_mean": 145.2,
                "rank": 1
            },
            ...
        ],
        "volcano_plot_path": "/app/results/volcano_plot.png",
        "ma_plot_path": "/app/results/ma_plot.png"
    }
}
```

### Why DESeq2?

**Statistical rigor:**
- Models count data with negative binomial distribution
- Accounts for library size differences
- Shrinks log fold changes for low-count genes
- Multiple testing correction (Benjamini-Hochberg)

**Gold standard** for RNA-seq differential expression

### Calling R from Python (rpy2)

```python
import rpy2.robjects as ro

# Source R script
ro.r('source("/app/r_scripts/deseq2_wrapper.R")')

# Call R function
r_code = f'''
result <- run_deseq2(
    "{count_matrix_path}",
    "{condition_path}",
    "{control_group}",
    "{treatment_group}",
    "{output_dir}",
    {padj_threshold},
    {log2fc_threshold}
)
c(result$num_genes, result$num_significant, ...)
'''

result_vec = ro.r(r_code)
num_genes = int(result_vec[0])
```

**Challenges:**
- Type conversion (R → Python)
- Error handling across languages
- Managing R environment

### KEGG Enrichment (Future)

**Purpose:** Identify biological pathways enriched in significant genes

**Input:** List of significant gene IDs

**Output:**
- Enriched KEGG pathways
- P-values
- Gene lists per pathway
- Visualizations (dotplot, barplot)

---

## 9. Strengths & Weaknesses

### Strengths

#### 1. Modularity & Maintainability
✅ **Independent Services:** Each can be developed/tested/deployed separately  
✅ **Clear Boundaries:** Data, ML, Bioinformatics cleanly separated  
✅ **Easy to Understand:** Each service has single responsibility  
✅ **Technology Flexibility:** Python for ML, R for bioinformatics  

#### 2. Reproducibility
✅ **Docker:** Same environment everywhere  
✅ **Experiment Tracking:** All parameters/results logged  
✅ **Data Caching:** Avoid re-downloading same datasets  
✅ **Version Control:** All code in git, docker images tagged  

#### 3. Scalability
✅ **Horizontal Scaling:** Can replicate services  
✅ **Resource Allocation:** Heavy services get more CPU/RAM  
✅ **Async Operations:** Services can run in parallel  
✅ **Streaming:** Handle large datasets efficiently  

#### 4. Ensemble Approach
✅ **Robust Feature Selection:** Multiple algorithms vote  
✅ **Reduces Overfitting:** Consensus less likely to be noise  
✅ **Algorithm Comparison:** See which performs best  
✅ **Confidence Scores:** Features selected by 5/5 models = high confidence  

#### 5. Integration of ML + Traditional Bioinformatics
✅ **Best of Both Worlds:** ML feature selection + DESeq2 statistics  
✅ **Cross-Validation:** ML features that overlap with DESeq2 = most robust  
✅ **Same Gene Filtering:** Fair comparison (both use CV-filtered genes)  

#### 6. NASA OSDR Integration
✅ **Open Science:** Public data, reproducible  
✅ **Rich Metadata:** Experimental conditions included  
✅ **Caching:** Fast repeated access  

### Weaknesses

#### 1. Complexity
❌ **Learning Curve:** gRPC, Docker, microservices all need to be understood  
❌ **Deployment:** More complex than monolithic app  
❌ **Debugging:** Errors can span multiple services  
❌ **Network Overhead:** gRPC calls between services add latency  

#### 2. Small Sample Size
❌ **18 samples → 14 train, 4 test:** Very small for ML  
❌ **Overfitting Risk:** Models may memorize training data  
❌ **Unstable Performance:** Small changes in data = large metric swings  
❌ **Limited Generalization:** May not work on new datasets  

**Mitigation:**
- Use cross-validation instead of single train/test split
- Regularization (L1/L2)
- Simpler models (fewer parameters)
- Focus on consensus features (less likely to be noise)

#### 3. Feature Order Bias (Current Bug)
❌ **Top features are alphabetically first genes:** Suggests position bias  
❌ **Not biologically meaningful:** ENSMUSG00000000001-10 unlikely to all be important  

**Likely Causes:**
- Random state issue in data splitting
- Algorithms picking features by position
- Correlation with condition by chance

**Solutions to Try:**
- Shuffle feature order before training
- Different random seeds
- Check if features are actually correlated with outcome
- Increase sample size (if possible)

#### 4. Limited Algorithm Diversity
❌ **All supervised classification:** No unsupervised, no deep learning  
❌ **No ensemble methods:** Bagging/boosting could improve  
❌ **No feature engineering:** Just using raw (transformed) counts  

#### 5. No Hyperparameter Tuning
❌ **Using defaults:** Not optimized for this specific data  
❌ **No grid search / random search:** Missing potential performance gains  

**Why not included:**
- Small sample size makes tuning unreliable
- Computational cost
- Complexity

#### 6. Single Point of Failure
❌ **If Data Service fails:** Entire pipeline stops  
❌ **No redundancy:** Each service has one instance  
❌ **No load balancing:** Can't distribute work  

**Enterprise Solution:**
- Service meshes (Istio, Linkerd)
- Load balancers
- Kubernetes for orchestration

#### 7. Limited Bioinformatics Features
❌ **Only DESeq2:** No other differential expression tools  
❌ **KEGG not fully integrated:** Pathway analysis incomplete  
❌ **No gene ontology:** Missing functional enrichment  
❌ **No network analysis:** Gene interaction networks not explored  

#### 8. No Web UI
❌ **Command line only:** Not user-friendly for non-programmers  
❌ **No visualizations in browser:** Plots saved as files  
❌ **No real-time progress bar:** Have to check logs  

---

## 10. Future Directions

### Short Term (1-3 months)

#### 1. Fix Feature Order Bias
**Priority: CRITICAL**
- Debug why top features are alphabetically first
- Implement feature shuffling
- Validate with different random seeds
- Compare results before/after fix

#### 2. Add Cross-Validation
**Priority: HIGH**
- Replace single train/test split with 5-fold or 10-fold CV
- More reliable performance estimates
- Better for small datasets
- Stratified to maintain class balance

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

#### 3. Implement Hyperparameter Tuning
**Priority: MEDIUM**
- Grid search or random search for each algorithm
- Use cross-validation for evaluation
- Save best hyperparameters per dataset

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

#### 4. Complete KEGG Enrichment Integration
**Priority: MEDIUM**
- Finish KEGG RPC implementation
- Visualize enriched pathways
- Add to ensemble pipeline
- Compare ML consensus features with KEGG pathways

#### 5. Add Gene Ontology Analysis
**Priority: MEDIUM**
- GO term enrichment for consensus features
- Biological process, molecular function, cellular component
- Integration with R (clusterProfiler)

### Medium Term (3-6 months)

#### 6. Web Dashboard
**Priority: HIGH**
- React/Vue frontend
- Real-time pipeline progress
- Interactive visualizations (D3.js, Plotly)
- Browse experiments
- Compare runs side-by-side

**Features:**
- Upload new datasets
- Start pipeline with form inputs
- View results in tables/plots
- Download models and plots
- Experiment comparison

#### 7. Deep Learning Models
**Priority: MEDIUM**
- Add neural network classifiers (PyTorch/TensorFlow)
- Autoencoders for feature extraction
- Transfer learning from pre-trained models
- Compare with traditional ML

#### 8. Feature Engineering
**Priority: MEDIUM**
- Gene-gene interactions (polynomial features)
- Pathway-level features (aggregate genes by pathway)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Feature selection algorithms (RFE, SelectKBest)

#### 9. More Bioinformatics Analyses
**Priority: MEDIUM**
- Gene set enrichment analysis (GSEA)
- Protein-protein interaction networks
- Co-expression network analysis (WGCNA)
- Integration with other databases (Reactome, GO, MSigDB)

#### 10. Performance Monitoring
**Priority: MEDIUM**
- Prometheus metrics
- Grafana dashboards
- Track API latency, success rates
- Resource usage (CPU, memory, disk)
- Alerting on failures

### Long Term (6-12 months)

#### 11. Multi-Omics Integration
**Priority: HIGH (if data available)**
- Integrate proteomics, metabolomics, genomics
- Multi-view learning
- Cross-omics feature selection
- Systems biology approach

#### 12. Production Deployment
**Priority: HIGH**
- Kubernetes orchestration
- Load balancing
- Auto-scaling based on demand
- CI/CD pipeline (GitHub Actions)
- Automated testing
- Blue-green deployments

#### 13. Advanced Ensemble Methods
**Priority: MEDIUM**
- Stacking (meta-learner on top of base models)
- Weighted voting based on model confidence
- Ensemble feature selection (stability selection)
- Bayesian model averaging

#### 14. Causal Inference
**Priority: RESEARCH**
- Move beyond correlation to causation
- Directed acyclic graphs (DAGs)
- Structural equation modeling
- Interventional predictions

#### 15. Federated Learning
**Priority: RESEARCH**
- Train models across multiple institutions
- Preserve data privacy
- Aggregate model updates, not data
- Useful for sensitive medical data

#### 16. Explainable AI (XAI)
**Priority: HIGH**
- SHAP values for model interpretability
- LIME for local explanations
- Feature interaction visualization
- Understand WHY models make predictions

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

#### 17. Automated Machine Learning (AutoML)
**Priority: MEDIUM**
- Auto-sklearn or TPOT for algorithm selection
- Automatic hyperparameter tuning
- Feature engineering automation
- Reduce manual configuration

#### 18. Real-Time Predictions API
**Priority: MEDIUM**
- REST endpoint for predictions
- Load model once, serve many requests
- Low-latency inference
- Useful for applications

```python
POST /api/predict
{
  "model_id": "model_abc123",
  "features": {
    "GENE_001": 123.4,
    "GENE_002": 567.8,
    ...
  }
}

Response:
{
  "prediction": "Space Flight",
  "probability": 0.89,
  "confidence": "high"
}
```

### Research Directions

#### 19. Transfer Learning from Large Datasets
- Pre-train on large public RNA-seq datasets
- Fine-tune on small spaceflight datasets
- Leverage knowledge from thousands of experiments

#### 20. Active Learning
- Intelligently select which samples to sequence next
- Maximize information gain per sample
- Reduce sequencing costs

#### 21. Synthetic Data Generation
- GANs to generate synthetic RNA-seq data
- Augment small datasets
- Balance classes
- Privacy-preserving data sharing

#### 22. Integration with Literature
- PubMed/bioRxiv search for relevant papers
- Extract gene mentions and associations
- Prior knowledge integration
- Hypothesis generation

---

## Summary

### What We Built
A **modular, microservices-based ML bioinformatics pipeline** that:
1. Downloads RNA-seq data from NASA OSDR
2. Applies CV filtering for feature selection
3. Trains ensemble of 5 ML algorithms
4. Computes consensus feature importance
5. Runs DESeq2 differential expression
6. Tracks all experiments

### Key Innovations
- **Dual Analysis:** ML consensus + DESeq2 statistics
- **Same Gene Filtering:** Fair comparison between methods
- **Ensemble Approach:** Robust feature selection
- **Modular Architecture:** Easy to extend and maintain
- **Docker + gRPC:** Reproducible, scalable, language-agnostic

### Current Limitations
- Small sample sizes (18 samples)
- Feature order bias bug
- No hyperparameter tuning
- Limited bioinformatics analyses
- No web interface

### Next Steps
1. **Fix feature order bias** (critical)
2. Add cross-validation
3. Build web dashboard
4. Expand bioinformatics features
5. Deploy to production (Kubernetes)

### Impact
- **Accelerates discovery:** Automated pipeline vs manual analysis
- **Increases robustness:** Ensemble > single algorithm
- **Enables reproducibility:** Docker + experiment tracking
- **Facilitates collaboration:** Shared interface, tracked results

---

## Questions?

