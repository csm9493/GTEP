#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main_test_replay.py --config=./exps/replay_imagenet100_2.json --rand_select_seed 0
# CUDA_VISIBLE_DEVICES=0 python main_test_wa.py --config=./exps/wa_imagenet100_2.json --rand_select_seed 0
# CUDA_VISIBLE_DEVICES=0 python main_test_icarl.py --config=./exps/icarl_imagenet100_2.json --rand_select_seed 0
# CUDA_VISIBLE_DEVICES=0 python main_test_bic.py --config=./exps/bic_imagenet100_2.json --rand_select_seed 0
# CUDA_VISIBLE_DEVICES=0 python main_test_podnet.py --config=./exps/podnet_imagenet100_2.json --rand_select_seed 0
# CUDA_VISIBLE_DEVICES=0 python main_test_foster.py --config=./exps/foster_imagenet100_2.json --rand_select_seed 0
# CUDA_VISIBLE_DEVICES=0 python main_test_der.py --config=./exps/der_imagenet100_2.json --rand_select_seed 0
# CUDA_VISIBLE_DEVICES=0 python main_test_memo.py --config=./exps/memo_imagenet100_2.json --rand_select_seed 0
# CUDA_VISIBLE_DEVICES=0 python main_test_beef.py --config=./exps/beef_imagenet100_2.json --rand_select_seed 0

