#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main_test_l2p.py --config=./exps/l2p_imagenetr100_2_test.json 
CUDA_VISIBLE_DEVICES=1 python main_test_l2p.py --config=./exps/l2p_cub100_2_test.json 
CUDA_VISIBLE_DEVICES=1 python main_test_l2p.py --config=./exps/l2p_imageneta100_2_test.json 

# CUDA_VISIBLE_DEVICES=1 python main_test_dualprompt.py --config=./exps/dualprompt_imagenetr100_2_test.json 
# CUDA_VISIBLE_DEVICES=1 python main_test_dualprompt.py --config=./exps/dualprompt_cub100_2_test.json 
# CUDA_VISIBLE_DEVICES=1 python main_test_dualprompt.py --config=./exps/dualprompt_imageneta100_2_test.json 

# CUDA_VISIBLE_DEVICES=1 python main_test_adam_adapter.py --config=./exps/adam_adapter_imagenetr100_2_test.json 
# CUDA_VISIBLE_DEVICES=1 python main_test_adam_adapter.py --config=./exps/adam_adapter_cub100_2_test.json 
# CUDA_VISIBLE_DEVICES=1 python main_test_adam_adapter.py --config=./exps/adam_adapter_imageneta100_2_test.json 

# CUDA_VISIBLE_DEVICES=1 python main_test_coda_prompt.py --config=./exps/coda_prompt_imagenetr100_2_test.json 
# CUDA_VISIBLE_DEVICES=1 python main_test_coda_prompt.py --config=./exps/coda_prompt_cub100_2_test.json 
# CUDA_VISIBLE_DEVICES=1 python main_test_coda_prompt.py --config=./exps/coda_prompt_imageneta100_2_test.json 

# CUDA_VISIBLE_DEVICES=1 python main_test_ease.py --config=./exps/ease_imagenetr100_2_test.json 
# CUDA_VISIBLE_DEVICES=1 python main_test_ease.py --config=./exps/ease_cub100_2_test.json 
# CUDA_VISIBLE_DEVICES=1 python main_test_ease.py --config=./exps/ease_imageneta100_2_test.json 

# CUDA_VISIBLE_DEVICES=1 python main_test_ranpac.py --config=./exps/ranpac_imagenetr_search.json 
# CUDA_VISIBLE_DEVICES=1 python main_test_ranpac.py --config=./exps/ranpac_cub100_2_test.json 
# CUDA_VISIBLE_DEVICES=1 python main_test_ranpac.py --config=./exps/ranpac_imageneta100_2_test.json 
