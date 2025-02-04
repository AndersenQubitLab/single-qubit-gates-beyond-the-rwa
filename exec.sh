#!/bin/bash

#SBATCH --job-name="Generate time evolutions"
#SBATCH --time=30:00:10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=41
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=100MB
#SBATCH --account=research-qutech-qrd

module load 2023r1
module load python
module load py-pip
pip install --user -r requirements.txt

srun python main.py > pi.log
