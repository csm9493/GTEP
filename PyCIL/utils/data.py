import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
        
class iCIFAR50_1(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data_temp, self.train_targets_temp = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data_temp, self.test_targets_temp = test_dataset.data, np.array(
            test_dataset.targets
        )
        
        tr_data_len = self.train_targets_temp.shape[0]
        te_data_len = self.test_targets_temp.shape[0]
        
        order = [i for i in range(len(np.unique(self.train_targets_temp)))][:50]
        np.random.shuffle(order)
        
        self.train_data = []
        self.train_targets = []
        
        for i in range(tr_data_len):
            
            if self.train_targets_temp[i] in order:
                self.train_data.append(self.train_data_temp[i])
                self.train_targets.append(self.train_targets_temp[i])
                
        self.train_targets = np.array(self.train_targets)
                
        
        self.test_data = []
        self.test_targets = []
        
        for i in range(te_data_len):
            
            if self.test_targets_temp[i] in order:
                self.test_data.append(self.test_data_temp[i])
                self.test_targets.append(self.test_targets_temp[i])
        
        self.test_targets = np.array(self.test_targets)
            
#         self.train_data = self.train_data[:tr_data_len]
#         self.train_targets = self.train_targets[:tr_data_len]
        
#         self.test_data = self.test_data[:te_data_len]
#         self.test_targets = self.test_targets[:te_data_len]
        
        print ('len(train_data) : ', len(self.train_data),' len(train_targets) : ',  self.train_targets.shape[0])
        print ('len(test_data) : ', len(self.test_data),' len(test_targets) : ',  self.test_targets.shape[0])
        
class iCIFAR50_2(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        
        self.train_data_temp, self.train_targets_temp = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data_temp, self.test_targets_temp = test_dataset.data, np.array(
            test_dataset.targets
        )
        
        tr_data_len = self.train_targets_temp.shape[0]
        te_data_len = self.test_targets_temp.shape[0]
        
        order = [i for i in range(len(np.unique(self.train_targets_temp)))][50:]
        np.random.shuffle(order)
        
        self.train_data = []
        self.train_targets = []
        
        for i in range(tr_data_len):
            
            if self.train_targets_temp[i] in order:
                self.train_data.append(self.train_data_temp[i])
                self.train_targets.append(self.train_targets_temp[i])
                
        self.train_targets = np.array(self.train_targets) % 50
                
        
        self.test_data = []
        self.test_targets = []
        
        for i in range(te_data_len):
            
            if self.test_targets_temp[i] in order:
                self.test_data.append(self.test_data_temp[i])
                self.test_targets.append(self.test_targets_temp[i])
        
        self.test_targets = np.array(self.test_targets) % 50
            
#         self.train_data = self.train_data[:tr_data_len]
#         self.train_targets = self.train_targets[:tr_data_len]
        
#         self.test_data = self.test_data[:te_data_len]
#         self.test_targets = self.test_targets[:te_data_len]
        
        print ('len(train_data) : ', len(self.train_data),' len(train_targets) : ',  self.train_targets.shape[0])
        print ('len(test_data) : ', len(self.test_data),' len(test_targets) : ',  self.test_targets.shape[0])
        

class iImageNet50_1(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    # class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data_temp, self.train_targets_temp = split_images_labels(train_dset.imgs)
        self.test_data_temp, self.test_targets_temp = split_images_labels(test_dset.imgs)
        
        tr_data_len = self.train_targets_temp.shape[0]
        te_data_len = self.test_targets_temp.shape[0]
        
        order = [i for i in range(len(np.unique(self.train_targets_temp)))][:50]
        np.random.shuffle(order)
        
        self.train_data = []
        self.train_targets = []
        
        for i in range(tr_data_len):
            
            if self.train_targets_temp[i] in order:
                self.train_data.append(self.train_data_temp[i])
                self.train_targets.append(self.train_targets_temp[i])
                
        self.train_targets = np.array(self.train_targets) % 50
                
        
        self.test_data = []
        self.test_targets = []
        
        for i in range(te_data_len):
            
            if self.test_targets_temp[i] in order:
                self.test_data.append(self.test_data_temp[i])
                self.test_targets.append(self.test_targets_temp[i])
        
        self.test_targets = np.array(self.test_targets) % 50
            
        print ('len(train_data) : ', len(self.train_data),' len(train_targets) : ',  self.train_targets.shape[0])
        print ('len(test_data) : ', len(self.test_data),' len(test_targets) : ',  self.test_targets.shape[0])
        


class iImageNet50_2(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    # class_order = np.arange(1000).tolist()

    def download_data(self):
#         assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data_temp, self.train_targets_temp = split_images_labels(train_dset.imgs)
        self.test_data_temp, self.test_targets_temp = split_images_labels(test_dset.imgs)
        
        tr_data_len = self.train_targets_temp.shape[0]
        te_data_len = self.test_targets_temp.shape[0]
        
        order = [i for i in range(len(np.unique(self.train_targets_temp)))][50:]
        np.random.shuffle(order)
        
        self.train_data = []
        self.train_targets = []
        
        for i in range(tr_data_len):
            
            if self.train_targets_temp[i] in order:
                self.train_data.append(self.train_data_temp[i])
                self.train_targets.append(self.train_targets_temp[i])
                
        self.train_targets = np.array(self.train_targets) % 50
                
        
        self.test_data = []
        self.test_targets = []
        
        for i in range(te_data_len):
            
            if self.test_targets_temp[i] in order:
                self.test_data.append(self.test_data_temp[i])
                self.test_targets.append(self.test_targets_temp[i])
        
        self.test_targets = np.array(self.test_targets) % 50
            
        print ('len(train_data) : ', len(self.train_data),' len(train_targets) : ',  self.train_targets.shape[0])
        print ('len(test_data) : ', len(self.test_data),' len(test_targets) : ',  self.test_targets.shape[0])
        

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
#         assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iImageNet100_2(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
#         assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
