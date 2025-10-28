import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
from scipy.io import savemat
import time
import numpy as np

global init
init = False

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    

    for seed in seed_list:
        args["seed"] = seed
        
        if init == False:
            args["device"] = device
            
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_seed_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    
    global init
    if init == False:
        _set_device(args)
        init = True
        
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    forgetting_dict = {"cnn_forgetting": [], "nme_forgetting": []}
    other_info = {"eval_time": [], "tr_time": [], "after_tr_time": [], "memory": [], "num_all_params": [], "num_trainable_params": []}
    
    for task in range(data_manager.nb_tasks):
        
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        
        other_info["num_all_params"].append(count_parameters(model._network))
        other_info["num_trainable_params"].append(count_parameters(model._network, True))
        
        tr_start_time = time.time()
        model.incremental_train(data_manager)
        tr_time = time.time() - tr_start_time
        other_info["tr_time"].append(tr_time)
        
        logging.info(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
        other_info["memory"].append(torch.cuda.max_memory_allocated(device=None))
        
        eval_start_time = time.time()
        cnn_accy, nme_accy = model.eval_task()
        eval_time = time.time() - eval_start_time
        other_info["eval_time"].append(eval_time)
        
        after_tr_start_time = time.time()
        model.after_task()
        after_tr_time = time.time() - after_tr_start_time
        other_info["after_tr_time"].append(after_tr_time)
        
        savemat(logfilename + "_other_info.mat", other_info)

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)
            
            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
            
#             matrix["cnn_matrix"] = cnn_matrix
#             matrix["nme_matrix"] = nme_matrix
            
#             print (matrix)
            
            savemat(logfilename + "_cnn_curve.mat", cnn_curve)
            savemat(logfilename + "_nme_curve.mat", nme_curve)
#             savemat(logfilename + "_matrix.mat", matrix)
            
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            
#             matrix["cnn_matrix"] = cnn_matrix
            
            savemat(logfilename + "_cnn_curve.mat", cnn_curve)
            savemat(logfilename + "_matrix.mat", matrix)
            
    
    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print('Accuracy Matrix (CNN):')
        forgetting_dict["cnn_forgetting"].append(forgetting)
        print(np_acctable)
        logging.info('Forgetting (CNN): {}'.format(forgetting))
    if len(nme_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(nme_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print('Accuracy Matrix (NME):')
        forgetting_dict["nme_forgetting"].append(forgetting)
        print(np_acctable)
        logging.info('Forgetting (NME): {}'.format(forgetting))
    
    
    savemat(logfilename + "_forgetting_dict.mat", forgetting_dict)
        
            
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
            

    
def _set_device(args):
    device_type = args["device"]
    gpus = []


    for device in device_type:

        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

        args["device"] = gpus


def _set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
