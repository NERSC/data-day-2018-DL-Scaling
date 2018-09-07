#!/bin/bash
#SBATCH -q regular
#SBATCH -C knl
#SBATCH -t 00:20:00
#SBATCH -J hep_training

#load environment
module load tensorflow/intel-1.9.0-py36

#OpenMP
export OMP_NUM_THREADS=66
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the code
rm -rf logs
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 --cpu_bind=cores python train_horovod.py hparams/cnn.yaml demo_multi_node
