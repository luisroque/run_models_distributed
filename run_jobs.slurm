#!/bin/bash
#SBATCH --ntasks=32   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=8G   # memory per CPU core
#SBATCH --mail-user=luis_roque@live.com   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
python run_models.py -a gpf -d police -e original
