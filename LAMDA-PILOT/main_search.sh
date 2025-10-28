#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./main_search/main_search_l2p.py --config=./exps/l2p_cub100_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python ./main_search/main_search_coda_prompt.py --config=./exps/coda_prompt_cub100_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python ./main_searchmain_search_dualprompt.py --config=./exps/dualprompt_cub100_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python ./main_searchmain_search_ease.py --config=./exps/ease_cub100_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python ./main_searchmain_search_ranpac.py --config=./exps/ranpac_cub100_1_search.json 
# CUDA_VISIBLE_DEVICES=0 python ./main_searchmain_search_adam_adapter.py --config=./exps/adam_adapter_cub100_1_search.json 
