# causal_service: CRISP in the pipeline

CRISP v1.1 (NASA FDL 2020 Astronaut Health team, `crisp/`) wrapped as a gRPC service. Orchestration
calls it after DESeq2 when a configuration sets `"causal": {"enabled": True}`.

## What it computes

Each OSD study (`source_dataset`) is an *environment*. CRISP's invariance methods look for genes
whose relationship to the factor (Space Flight vs Ground Control) is the same in every study:

| code  | method                                   | runs on                         |
|-------|------------------------------------------|---------------------------------|
| ICP   | linear invariant causal prediction       | top `max_features` genes        |
| NLICP | non-linear invariant causal prediction   | top `max_features` genes        |
| LIRM  | linear invariant risk minimization       | top `max_features` genes        |
| IRM   | non-linear invariant risk minimization   | all genes                       |
| ERM, RF | non-causal baselines (optional)        | all genes; never ranked         |

The "top `max_features` genes" come from a feature-reduction IRM that CRISP runs first (`FRIRM` in
the results). Spaceflight is the treatment and genes respond to it, so the genes CRISP reports are
**robust responders** (same flight signal in every study), not genes that cause the flight label.

Input preparation (`src/prepare.py`): samples outside the two factor values are dropped, studies
without both groups are dropped (at least 2 must remain), counts become log2(CPM + 1) and are
z-scored per gene, target = 1 for the second factor value.

## Ranking (`ranking_method`)

- `consensus` (default): each causal method's |coefficients| are divided by its largest one (0 stays
  0, top gene = 1); each method votes for its `top_k` genes (default `max_features`). Genes are ranked
  by votes, then mean scaled |coefficient|. `consensus` = votes ≥ a majority of the methods that
  produced coefficients.
- `crisp_v1.1`: CRISP's own `get_ensemble_results` (MinMax scaling to [-1, 1]), kept for comparison.
  MinMax scaling moves zero: with all-positive coefficients the least important gene maps to −1 and
  ranks with the most important one.

`direction`, `log2fc` and `consistent` come from the data, not the coefficients (whose signs mean
different things across methods): the treatment − control difference in log2(CPM + 1), and whether
it has the same sign in every study.

## Changes to CRISP itself

Only one, in `crisp/main.py`: Linear IRM now also triggers the feature-reduction step (it uses the
reduced feature set, so `models: ["LIRM"]` alone used to raise a NameError). The Google Cloud helpers
are replaced by a stand-in at run time (`src/crisp_runner.py`); nothing else is modified.

## Resources and limits

- Each analysis runs in a child process (`src/crisp_runner.py`); `CRISP_MAX_CONCURRENT` (default 1)
  runs at a time; `CRISP_TIMEOUT_SECONDS` (default 6 h) or the request's `timeout_seconds` stops it.
- Tested: 46 samples × 300 genes, default models, about 1 minute on 2 CPUs.
- **Memory**: NLICP's defining-set step grows quickly when many gene subsets pass its invariance test.
  With `max_features: 50` it exceeded 8 GB on the same small dataset. Keep `max_features` at 20 unless
  you have memory to spare, and give the container a memory limit so an overrun only kills the CRISP
  child process (the service reports "killed, most likely out of memory").

## docker-compose.yml

```yaml
  causal_service:
    build: ./causal_service
    container_name: causal_service
    environment:
      - CAUSAL_PORT=50055
      - CRISP_MAX_CONCURRENT=1
    volumes:
      - ./causal_service/results:/app/results     # per-run CRISP inputs, logs, PDFs (optional)
    mem_limit: 12g                                # adjust to the machine
    restart: unless-stopped
```

and in the orchestration service: `CAUSAL_SERVICE_URL=causal_service:50055` (that is also the
default), plus `depends_on: [causal_service]` if you want it started first.

Orchestration needs the stubs like every other service:

```bash
python -m grpc_tools.protoc -I causal_service/proto --python_out=orchestration_service/src/generated \
    --grpc_python_out=orchestration_service/src/generated causal_service/proto/causal_service.proto
sed -i '' 's/^import causal_service_pb2 as/from src.generated import causal_service_pb2 as/' \
    orchestration_service/src/generated/causal_service_pb2_grpc.py      # macOS sed; Linux: sed -i
```

The service's own stubs are generated inside its image by the Dockerfile.
