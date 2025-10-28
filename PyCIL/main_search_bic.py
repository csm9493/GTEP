import json
import argparse
from trainer import train
import random

ep_arr = [30, 70, 120, 160, 200]
milestone_arr = [2, 3, 4]

lr_arr = [0.05, 0.1, 0.15, 0.2, 0.3]
lr_decay_arr = [0.1, 0.3, 0.5]
batch_arr = [32, 64, 128, 256, 512]
w_decay_arr = [0.0001, 0.0005, 0.001, 0.005]
scheduler_arr = ['steplr', 'cosine']

T_arr = [0.5, 1, 1.5, 2, 2.5]
lambda_kd_arr = [0.5, 1, 1.5, 2, 3]
split_ratio_arr = [0.05, 0.1, 0.15, 0.2, 0.3]

def main():
    args = setup_parser().parse_args()
    total_rand_num = args.rand_num
    seed_arr = [args.rand_select_seed,]
    random.seed(total_rand_num)
    
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    print ('total rand num : ', total_rand_num)
    print ('rand select seed : ', seed_arr[0])
    
    for rand_num in range(total_rand_num, total_rand_num+1):
        
        seed = random.choice(seed_arr)
        ep = random.choice(ep_arr)
        milestone_num = random.choice(milestone_arr)
        lr = random.choice(lr_arr)
        lr_decay = random.choice(lr_decay_arr)
        batch = random.choice(batch_arr)
        w_decay = random.choice(w_decay_arr)
        scheduler = random.choice(scheduler_arr)
        
        T = random.choice(T_arr)
        lambda_kd = random.choice(lambda_kd_arr)
        split_ratio = random.choice(split_ratio_arr)
        
        
        if milestone_num == 2:
            
            milestones = [int(ep*(2/5)), int(ep*(4/5))]
            
        elif milestone_num == 3:
            
            milestones = [int(ep*(2/7)), int(ep*(4/7)), int(ep*(6/7))]
            
        elif milestone_num == 4:
            
            milestones = [int(ep*(2/9)), int(ep*(4/9)), int(ep*(6/9)), int(ep*(8/9))]
            
        prefix = "rand_num_{}_ep_{}_milestone_{}_lr_{}_lr_decay_{}_batch_{}_w_decay_{}_scheduler_{}_T_{}_lambda_kd_{}_split_ratio_{}".format(
                        rand_num,
                        ep,
                        milestone_num,
                        lr,
                        lr_decay,
                        batch,
                        w_decay,
                        scheduler,
                        T,
                        lambda_kd,
                        split_ratio,
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
            "T":T,
            "lambda_kd":lambda_kd,
            "split_ratio":split_ratio,
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
    parser.add_argument('--rand_num', type=int, default=0)
    parser.add_argument('--rand_select_seed', type=int, default=0)

    return parser


if __name__ == '__main__':
    main()
