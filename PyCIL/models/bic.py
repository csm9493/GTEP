import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNetWithBias

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005

num_workers = 8

class BiC(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNetWithBias(
            args, False, bias_correction=True
        )
        self._class_means = None
        
        self.T = args["T"]
        self.lambda_kd = args["lambda_kd"]
        self.split_ratio = args["split_ratio"]

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task >= 1:
            train_dset, val_dset = data_manager.get_dataset_with_split(
                np.arange(self._known_classes, self._total_classes),
                source="train",
                mode="train",
                appendent=self._get_memory(),
                val_samples_per_class=int(
                    self.split_ratio * self._memory_size / self._known_classes
                ),
            )
            self.val_loader = DataLoader(
                val_dset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
            )
            logging.info(
                "Stage1 dset: {}, Stage2 dset: {}".format(
                    len(train_dset), len(val_dset)
                )
            )
            self.lamda = self._known_classes / self._total_classes
            logging.info("Lambda: {:.3f}".format(self.lamda))
        else:
            train_dset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train",
                mode="train",
                appendent=self._get_memory(),
            )
        test_dset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )

        self.train_loader = DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            test_dset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers
        )

        self._log_bias_params()
        
        if self._cur_task == 0:
            self._stage1_training_first(self.train_loader, self.test_loader)
        else:
            self._stage1_training(self.train_loader, self.test_loader)
            
        if self._cur_task >= 1:
            self._stage2_bias_correction(self.val_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._log_bias_params()

    def _run(self, train_loader, test_loader, optimizer, scheduler, epochs, stage):
        for epoch in range(1, epochs + 1):
            self._network.train()
            losses = 0.0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                if stage == "training":
                    clf_loss = F.cross_entropy(logits, targets)
                    if self._old_network is not None:
                        old_logits = self._old_network(inputs)["logits"].detach()
                        hat_pai_k = F.softmax(old_logits / self.T, dim=1)
                        log_pai_k = F.log_softmax(
                            logits[:, : self._known_classes] / self.T, dim=1
                        )
                        distill_loss = -torch.mean(
                            torch.sum(hat_pai_k * log_pai_k, dim=1)
                        )
                        
                        if self.lambda_kd == 0:
                        
                            loss = distill_loss * self.lamda + clf_loss * (1 - self.lamda)
                        else:
                            
                            loss = clf_loss + self.lambda_kd * distill_loss
                    else:
                        loss = clf_loss
                elif stage == "bias_correction":
                    loss = F.cross_entropy(torch.softmax(logits, dim=1), targets)
                else:
                    raise NotImplementedError()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            train_acc = self._compute_accuracy(self._network, train_loader)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "{} => Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}".format(
                stage,
                self._cur_task,
                epoch,
                epochs,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            logging.info(info)
            
    def _stage1_training_first(self, train_loader, test_loader):
        """
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        """

        ignored_params = list(map(id, self._network.bias_layers.parameters()))
        base_params = filter(
            lambda p: id(p) not in ignored_params, self._network.parameters()
        )
        network_params = [
            {"params": base_params, "lr": init_lr, "weight_decay": init_weight_decay},
            {
                "params": self._network.bias_layers.parameters(),
                "lr": 0,
                "weight_decay": 0,
            },
        ]
        optimizer = optim.SGD(
            network_params, lr=init_lr, momentum=0.9, weight_decay=init_weight_decay
        )
        
        if self.scheduler == 'cosine':

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=init_epoch
            )

        else:

            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        self._run(train_loader, test_loader, optimizer, scheduler, init_epoch, stage="training")

    def _stage1_training(self, train_loader, test_loader):
        """
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        """

        ignored_params = list(map(id, self._network.bias_layers.parameters()))
        base_params = filter(
            lambda p: id(p) not in ignored_params, self._network.parameters()
        )
        network_params = [
            {"params": base_params, "lr": self.lrate, "weight_decay": self.weight_decay},
            {
                "params": self._network.bias_layers.parameters(),
                "lr": 0,
                "weight_decay": 0,
            },
        ]
        optimizer = optim.SGD(
            network_params, lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay
        )
        if self.scheduler == 'cosine':

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.epochs
            )
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay
            )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        self._run(train_loader, test_loader, optimizer, scheduler, self.epochs, stage="training")

    def _stage2_bias_correction(self, val_loader, test_loader):
        if isinstance(self._network, nn.DataParallel):
            self._network = self._network.module
        network_params = [
            {
                "params": self._network.bias_layers[-1].parameters(),
                "lr": self.lrate,
                "weight_decay": self.weight_decay,
            }
        ]
        optimizer = optim.SGD(
            network_params, lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay
        )
        if self.scheduler == 'cosine':

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.epochs
            )
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay
            )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.to(self._device)

        self._run(
            val_loader, test_loader, optimizer, scheduler, self.epochs, stage="bias_correction"
        )

    def _log_bias_params(self):
        logging.info("Parameters of bias layer:")
        params = self._network.get_bias_params()
        for i, param in enumerate(params):
            logging.info("{} => {:.3f}, {:.3f}".format(i, param[0], param[1]))
