#!/bin/bash

# #SBATCH --nodes=1
# #SBATCH --gres=gpu:rtx8000:1
# #SBATCH --time=48:00:00
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=16GB
# ##SBATCH --job-name=imagenet100_b0_t5_replay
# #SBATCH --output=hpc_logs/%j.out
# #SBATCH --error=hpc_logs/%j.err

# module purge
# module load anaconda3/2020.07
# eval "$(conda shell.bash hook)"
# conda activate /scratch/sc10891/envs/pycil
# nvidia-smi

# cd /scratch/sc10891/research/PyCIL/

# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_replay.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 0 --config=./exps/replay_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_replay.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 1 --config=./exps/replay_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_replay.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 2 --config=./exps/replay_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_replay.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 3 --config=./exps/replay_imagenet50_1_search.json
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_replay.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 4 --config=./exps/replay_imagenet50_1_search.json 


# #SBATCH --nodes=1
# #SBATCH --gres=gpu:rtx8000:1
# #SBATCH --time=48:00:00
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=16GB
# #SBATCH --job-name=imagenet100_b0_t5_wa
# #SBATCH --output=hpc_logs/%j.out
# #SBATCH --error=hpc_logs/%j.err

# module purge
# module load anaconda3/2020.07
# eval "$(conda shell.bash hook)"
# conda activate /scratch/sc10891/envs/pycil
# nvidia-smi

# cd /scratch/sc10891/research/PyCIL/


# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_wa.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 0 --config=./exps/wa_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_wa.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 1 --config=./exps/wa_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_wa.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 2 --config=./exps/wa_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_wa.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 3 --config=./exps/wa_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_wa.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 4 --config=./exps/wa_imagenet50_1_search.json 


# #SBATCH --nodes=1
# #SBATCH --gres=gpu:rtx8000:1
# #SBATCH --time=48:00:00
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=16GB
# #SBATCH --job-name=imagenet100_b0_t5_icarl
# #SBATCH --output=hpc_logs/%j.out
# #SBATCH --error=hpc_logs/%j.err

# module purge
# module load anaconda3/2020.07
# eval "$(conda shell.bash hook)"
# conda activate /scratch/sc10891/envs/pycil
# nvidia-smi

# cd /scratch/sc10891/research/PyCIL/

# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_icarl.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 0 --config=./exps/icarl_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_icarl.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 1 --config=./exps/icarl_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_icarl.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 2 --config=./exps/icarl_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_icarl.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 3 --config=./exps/icarl_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_icarl.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 4 --config=./exps/icarl_imagenet50_1_search.json 


# #SBATCH --nodes=1
# #SBATCH --gres=gpu:rtx8000:1
# #SBATCH --time=48:00:00
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=2
# #SBATCH --mem=16GB
# #SBATCH --job-name=imagenet100_b0_t5_wa
# #SBATCH --output=hpc_logs/%j.out
# #SBATCH --error=hpc_logs/%j.err

# module purge
# module load anaconda3/2020.07
# eval "$(conda shell.bash hook)"
# conda activate /scratch/sc10891/envs/pycil
# nvidia-smi

# cd /scratch/sc10891/research/PyCIL/

# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_podnet.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 0 --config=./exps/podnet_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_podnet.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 1 --config=./exps/podnet_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_podnet.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 2 --config=./exps/podnet_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_podnet.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 3 --config=./exps/podnet_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_podnet.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 4 --config=./exps/podnet_imagenet50_1_search.json 


# #SBATCH --nodes=1
# #SBATCH --gres=gpu:rtx8000:1
# #SBATCH --time=48:00:00
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=16GB
# #SBATCH --job-name=imagenet100_b0_t5_bic
# #SBATCH --output=hpc_logs/%j.out
# #SBATCH --error=hpc_logs/%j.err

# module purge
# module load anaconda3/2020.07
# eval "$(conda shell.bash hook)"
# conda activate /scratch/sc10891/envs/pycil
# nvidia-smi

# cd /scratch/sc10891/research/PyCIL/

# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_bic.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 0 --config=./exps/bic_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_bic.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 1 --config=./exps/bic_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_bic.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 2 --config=./exps/bic_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_bic.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 3 --config=./exps/bic_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_bic.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 4 --config=./exps/bic_imagenet50_1_search.json 


# #SBATCH --nodes=1
# #SBATCH --gres=gpu:1
# #SBATCH --time=48:00:00
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=16GB
# #SBATCH --job-name=imagenet100_b0_t5_der
# #SBATCH --output=hpc_logs/%j.out
# #SBATCH --error=hpc_logs/%j.err

# module purge
# module load anaconda3/2020.07
# eval "$(conda shell.bash hook)"
# conda activate /scratch/sc10891/envs/pycil
# nvidia-smi

# cd /scratch/sc10891/research/PyCIL/

# # CUDA_VISIBLE_DEVICES=0 python3.8 main_search_der.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 0 --config=./exps/der_imagenet50_1_search.json 
# # CUDA_VISIBLE_DEVICES=0 python3.8 main_search_der.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 1 --config=./exps/der_imagenet50_1_search.json 
# # CUDA_VISIBLE_DEVICES=0 python3.8 main_search_der.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 2 --config=./exps/der_imagenet50_1_search.json 
# # CUDA_VISIBLE_DEVICES=0 python3.8 main_search_der.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 3 --config=./exps/der_imagenet50_1_search.json 
# # CUDA_VISIBLE_DEVICES=0 python3.8 main_search_der.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 4 --config=./exps/der_imagenet50_1_search.json 


# #SBATCH --nodes=1
# #SBATCH --gres=gpu:1
# #SBATCH --time=48:00:00
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=16GB
# #SBATCH --job-name=imagenet100_b0_t5_foster
# #SBATCH --output=hpc_logs/%j.out
# #SBATCH --error=hpc_logs/%j.err

# module purge
# module load anaconda3/2020.07
# eval "$(conda shell.bash hook)"
# conda activate /scratch/sc10891/envs/pycil
# nvidia-smi

# cd /scratch/sc10891/research/PyCIL/

# # CUDA_VISIBLE_DEVICES=0 python3.8 main_search_foster.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 0 --config=./exps/foster_imagenet50_1_search.json 
# # CUDA_VISIBLE_DEVICES=0 python3.8 main_search_foster.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 1 --config=./exps/foster_imagenet50_1_search.json 
# # CUDA_VISIBLE_DEVICES=0 python3.8 main_search_foster.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 2 --config=./exps/foster_imagenet50_1_search.json 
# # CUDA_VISIBLE_DEVICES=0 python3.8 main_search_foster.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 3 --config=./exps/foster_imagenet50_1_search.json 
# # CUDA_VISIBLE_DEVICES=0 python3.8 main_search_foster.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 4 --config=./exps/foster_imagenet50_1_search.json 


# #SBATCH --nodes=1
# #SBATCH --gres=gpu:1
# #SBATCH --time=48:00:00
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=16GB
# #SBATCH --job-name=imagenet100_b0_t5_beef
# #SBATCH --output=hpc_logs/%j.out
# #SBATCH --error=hpc_logs/%j.err

# module purge
# module load anaconda3/2020.07
# eval "$(conda shell.bash hook)"
# conda activate /scratch/sc10891/envs/pycil
# nvidia-smi

# cd /scratch/sc10891/research/PyCIL/

# # CUDA_VISIBLE_DEVICES=0 python3.8 main_search_beef.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 0 --config=./exps/beef_imagenet50_1_search.json 
# # CUDA_VISIBLE_DEVICES=0 python3.8 main_search_beef.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 1 --config=./exps/beef_imagenet50_1_search.json 
# # CUDA_VISIBLE_DEVICES=0 python3.8 main_search_beef.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 2 --config=./exps/beef_imagenet50_1_search.json
# # CUDA_VISIBLE_DEVICES=0 python3.8 main_search_beef.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 3 --config=./exps/beef_imagenet50_1_search.json
# # CUDA_VISIBLE_DEVICES=0 python3.8 main_search_beef.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 4 --config=./exps/beef_imagenet50_1_search.json 


# #SBATCH --nodes=1
# #SBATCH --gres=gpu:1
# #SBATCH --time=48:00:00
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=16GB
# #SBATCH --job-name=imagenet100_b0_t5_memo
# #SBATCH --output=hpc_logs/%j.out
# #SBATCH --error=hpc_logs/%j.err

# module purge
# module load anaconda3/2020.07
# eval "$(conda shell.bash hook)"
# conda activate /scratch/sc10891/envs/pycil
# nvidia-smi

# cd /scratch/sc10891/research/PyCIL/

# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_memo.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 0 --config=./exps/memo_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_memo.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 1 --config=./exps/memo_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_memo.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 2 --config=./exps/memo_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_memo.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 3 --config=./exps/memo_imagenet50_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python3.8 main_search_memo.py --rand_num=$SLURM_ARRAY_TASK_ID --rand_select_seed 4 --config=./exps/memo_imagenet50_1_search.json 
