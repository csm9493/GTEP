import json
import argparse
from trainer import train
import random

# IamgeNet100-1 Three seeds
# T=10
hp_data = 'imagenet100_1'
ep_arr = [30]
milestone_arr = [4]

lr_arr = [0.05]
lr_decay_arr = [0.1]
batch_arr = [64]
w_decay_arr = [0.0001]
scheduler_arr = ['steplr']

T_arr = [1.0]
lambda_kd_arr = [1.5]
lambda_fe_arr = [1.0]
beta1_arr = [0.93]
beta2_arr = [0.97]
comp_ep_arr = [160]

total_rand_num = 1

def main():
    args = setup_parser().parse_args()
    seed_arr = [args.rand_select_seed,]

    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    print ('seed_arr : ', seed_arr)
    
    for rand_num in range(total_rand_num):
        
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
        lambda_fe = random.choice(lambda_fe_arr)
        beta1 = random.choice(beta1_arr)
        beta2 = random.choice(beta2_arr)
        comp_ep = random.choice(comp_ep_arr)
        
        if milestone_num == 2:
            
            milestones = [int(ep*(2/5)), int(ep*(4/5))]
            comp_milestones = [int(comp_ep*(2/5)), int(comp_ep*(4/5))]
            
        elif milestone_num == 3:
            
            milestones = [int(ep*(2/7)), int(ep*(4/7)), int(ep*(6/7))]
            comp_milestones = [int(comp_ep*(2/7)), int(comp_ep*(4/7)), int(comp_ep*(6/7))]
            
        elif milestone_num == 4:
            
            milestones = [int(ep*(2/9)), int(ep*(4/9)), int(ep*(6/9)), int(ep*(8/9))]
            comp_milestones = [int(comp_ep*(2/9)), int(comp_ep*(4/9)), int(comp_ep*(6/9)), int(comp_ep*(8/9))]
            
        prefix = "{}_ep_{}_milestone_{}_lr_{}_lr_decay_{}_batch_{}_w_decay_{}_scheduler_{}_T_{}_lambda_kd_{}_fe_{}_beta_{}_{}_comp_ep_{}".format(
                        hp_data,
                        ep,
                        milestone_num,
                        lr,
                        lr_decay,
                        batch,
                        w_decay,
                        scheduler,
                        T,
                        lambda_kd,
                        lambda_fe,
                        beta1,
                        beta2,
                        comp_ep,
                    )
            
        parameters = {
            "seed":seed_arr, 
            "prefix":prefix, 
            "epochs":ep,
            "boosting_epochs":ep,
            "lrate":lr,
            "milestones":milestones,
            "lrate_decay":lr_decay,
            "batch_size":batch,
            "weight_decay":w_decay,
            "scheduler":scheduler,
            "T":T,
            "lambda_okd":lambda_kd,
            "lambda_fe":lambda_fe,
            "beta1":beta1,
            "beta2":beta2,
            "compression_epochs":comp_ep,
            "comp_milestones": comp_milestones
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
