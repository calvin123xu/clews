#!/encs/bin/bash
#SBATCH --job-name=clews_train
#SBATCH --account=weiping
#SBATCH --output=clews_train-slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=x_qiaoyu@speed.encs.concordia.ca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G -p pt
#SBATCH --gpus=1 -p pg
#SBATCH --time=7-00:00:00

# 环境设置
cd /speed-scratch/qiaoyu/speed-hpc/project/clews

source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/qiaoyu/speed-hpc/env/clews

# 运行下载脚本
export OMP_NUM_THREADS=1
srun python train.py jobname=dvi-clews conf=config/dvi-clews.yaml fabric.nnodes=1 fabric.ngpus=1 data.nworkers=2 training.batchsize=4
