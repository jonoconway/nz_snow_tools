#!/bin/sh -e

#SBATCH --job-name=gen_met23
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=nesi_prepost
#SBATCH --account=niwa00004
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jono.conway@niwa.co.nz
#SBATCH --mem=5GB
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export COMPUTE_NODE_LAUNCH=srun

module load NIWA Miniforge3 
source $(conda info --base)/etc/profile.d/conda.sh

conda activate /nesi/project/niwa03150/jonoconway/conda_env/mf_nzst_312_clean2

srun --unbuffered python /nesi/project/niwa03150/jonoconway/python_packages/nz_snow_tools/nz_snow_tools/hpc_runs/interp_met_nzcsm.py  nzra_cragieburn_30_square.yaml
#srun --unbuffered python /nesi/project/niwa03150/jonoconway/python_packages/nz_snow_tools/nz_snow_tools/hpc_runs/interp_met_nzcsm.py  nzra_ahuriri_test_100_square.yaml
