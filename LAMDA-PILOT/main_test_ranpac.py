import json
import argparse
from trainer import train
import random

hp_data = 'imgr_t20'

ep_arr = [20]
milestone_arr = [2]

lr_arr = [0.05]
lr_decay_arr = [0.3]
batch_arr = [24]
w_decay_arr = [0.0005]
scheduler_arr = ['constant']
optimizer_arr = ['sgd']

ffn_num_arr = [16] 
M_arr=[15000] 
prompt_token_num_arr = [20]

seed_arr = [0,1,2,3,4]
total_rand_num = 1

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    
    for rand_num in range(0, total_rand_num):
        
        ep = random.choice(ep_arr)
        milestone_num = random.choice(milestone_arr)
        lr = random.choice(lr_arr)
        lr_decay = random.choice(lr_decay_arr)
        batch = random.choice(batch_arr)
        w_decay = random.choice(w_decay_arr)
        scheduler = random.choice(scheduler_arr)
        optimizer = random.choice(optimizer_arr)
        seed = random.choice(seed_arr)
        
        ffn_num = random.choice(ffn_num_arr)
        M = random.choice(M_arr)
        prompt_token_num = random.choice(prompt_token_num_arr)
        
        
        if milestone_num == 2:
            
            milestones = [int(ep*(2/5)), int(ep*(4/5))]
            
        elif milestone_num == 3:
            
            milestones = [int(ep*(2/7)), int(ep*(4/7)), int(ep*(6/7))]
            
        elif milestone_num == 4:
            
            milestones = [int(ep*(2/9)), int(ep*(4/9)), int(ep*(6/9)), int(ep*(8/9))]
            
        
        prefix = "{}_ep_{}_milestone_{}_lr_{}_lr_decay_{}_batch_{}_w_decay_{}_scheduler_{}_optimizer_{}_ffn_num_{}_M_{}_pt_num_{}".format(
                        hp_data,
                        ep,
                        milestone_num,
                        lr,
                        lr_decay,
                        batch,
                        w_decay,
                        scheduler,
                        optimizer,
                        ffn_num,
                        M,
                        prompt_token_num
                    )

        parameters = {
            "seed":seed_arr, 
            "prefix":prefix, 
            "epochs":ep,
            "lrate":lr,
            "milestones":milestones,
            "lrate_decay":lr_decay,
            "batch_size":batch,
            "weight_decay":w_decay,
            "scheduler":scheduler,
            "optimizer":optimizer,
            "ffn_num":ffn_num,
            "M":M,
            "prompt_token_num":prompt_token_num
        }
        args.update(parameters)  # Add parameters from json

        print (args)

        train(args)
                        
    


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')

    return parser


if __name__ == '__main__':
    main()
