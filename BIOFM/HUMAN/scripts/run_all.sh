#!/bin/bash

JOBID_0=$(sbatch --parsable -t 1-0 --gres=gpu:1 ./run_0.sh)
JOBID_1a=$(sbatch --parsable -t 1-0 --gres=gpu:1 --dependency=afterok:${JOBID_0} ./run_1a.sh)
JOBID_1b=$(sbatch --parsable -t 1-0 --gres=gpu:1 --dependency=afterok:${JOBID_1a} ./run_1b.sh)
JOBID_2a=$(sbatch --parsable -t 1-0 --gres=gpu:1 --dependency=afterok:${JOBID_1b} ./run_2a.sh)
JOBID_2b=$(sbatch --parsable -t 1-0 --gres=gpu:1 --dependency=afterok:${JOBID_2a} ./run_2b.sh)
JOBID_2c=$(sbatch --parsable -t 1-0 --gres=gpu:1 --dependency=afterok:${JOBID_2b} ./run_2c.sh)
JOBID_3=$(sbatch --parsable -t 1-0 --gres=gpu:1 --dependency=afterok:${JOBID_2c} ./run_3.sh)
JOBID_4=$(sbatch --parsable -t 3-0 --gres=gpu:2 ./run_4.sh --dependency=afterok:${JOB_3}) 
JOBID_5a=$(sbatch --parsable -t 1-0 --gres=gpu:1 --dependency=afterok:${JOBID_4} ./run_5a.sh)
JOBID_5b=$(sbatch --parsable -t 1-0 --gres=gpu:1 --dependency=afterok:${JOBID_5a} ./run_5b.sh)
