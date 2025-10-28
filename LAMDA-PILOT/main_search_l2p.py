import json
import argparse
from trainer import train
import random

ep_arr = [3, 5, 10, 15, 20, 25]
milestone_arr = [2, 3, 4]

lr_arr = [0.000875, 0.001375, 0.001875, 0.002375, 0.0025]
lr_decay_arr = [0.1, 0.3, 0.5]
batch_arr = [8, 16, 24, 48, 64, 128]
w_decay_arr = [0, 0.0001, 0.0005]
scheduler_arr = ['steplr', 'cosine', 'constant']
optimizer_arr = ['sgd', 'adam', 'adamw']

size_arr = [10, 15, 20, 25, 30] # M
length_arr = [2, 4, 6, 8, 10] # L_p
top_k_arr = [2, 4, 6, 8, 10] # N
lamb_arr = [0.1, 0.3, 0.5]

seed_arr = [0,1,2,3,4]
total_rand_num = 30

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
        
        size = random.choice(size_arr)
        length = random.choice(length_arr)
        top_k = random.choice(top_k_arr)
        lamb = random.choice(lamb_arr)
        
        if milestone_num == 2:
            
            milestones = [int(ep*(2/5)), int(ep*(4/5))]
            
        elif milestone_num == 3:
            
            milestones = [int(ep*(2/7)), int(ep*(4/7)), int(ep*(6/7))]
            
        elif milestone_num == 4:
            
            milestones = [int(ep*(2/9)), int(ep*(4/9)), int(ep*(6/9)), int(ep*(8/9))]
            
        prefix = "rand_num_{}_ep_{}_milestone_{}_lr_{}_lr_decay_{}_batch_{}_w_decay_{}_scheduler_{}_optimizer_{}_size_{}_length_{}_top_k_{}_lamb_{}".format(
                        rand_num,
                        ep,
                        milestone_num,
                        lr,
                        lr_decay,
                        batch,
                        w_decay,
                        scheduler,
                        optimizer,
                        size,
                        length,
                        top_k,
                        lamb,
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
            "size":size,
            "length":length,
            "top_k":top_k,
            "lamb":lamb
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
