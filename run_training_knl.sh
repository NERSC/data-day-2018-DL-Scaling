#!/bin/bash
#SBATCH -q regular
#SBATCH -C knl
#SBATCH -t 4:00:00
#SBATCH -J hep_training

#load environment
module load tensorflow/intel-1.9.0-py36

#run the code
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 --cpu_bind=cores python train_demo.py hparams/cnn.yaml demo_single_node
