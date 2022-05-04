import torch
from vtab.cifar import CIFAR10, CIFAR100
from vtab.flowers102 import Flowers102
from vtab.clevr import CLEVRClassification, CLEVRDistance

def cifar100_1k_datasets(data_path, train_transforms, val_transforms, download=True):
    train_set = CIFAR100(data_path, train=True, transform=train_transforms, download=download, type='train1000')
    val_set = CIFAR100(data_path, train=False, transform=val_transforms, download=download, type='test')
    num_classes = 100
    return train_set, val_set, num_classes

def cifar100_800_200_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CIFAR100(data_path, train=True, transform=train_transforms, download=download, type='train800')
    val_set = CIFAR100(data_path, train=True, transform=val_transforms, download=download, type='val200')
    num_classes = 100
    return train_set, val_set, num_classes
    # raise NotImplementedError

def cifar100_full_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CIFAR100(data_path, train=True, transform=train_transforms, download=download, type='train1000')
    val_set = CIFAR100(data_path, train=False, transform=val_transforms, download=download, type='test')
    num_classes = 100
    return train_set, val_set, num_classes
    raise NotImplementedError

def flowers102_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = Flowers102(data_path, split='train', transform=train_transforms, download=download, type='train1000')
    val_set = Flowers102(data_path, split='test', transform=val_transforms, download=download, type='test')
    num_classes = 102
    return train_set, val_set, num_classes

def flowers102_800_200_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = Flowers102(data_path, split='train', transform=train_transforms, download=download, type='train800')
    val_set = Flowers102(data_path, split='val', transform=val_transforms, download=download, type='val200')
    num_classes = 102
    return train_set, val_set, num_classes

def flowers102_full_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = Flowers102(data_path, split='train', transform=train_transforms, download=download, type='all')
    val_set = Flowers102(data_path, split='test', transform=val_transforms, download=download, type='all')
    num_classes = 102
    return train_set, val_set, num_classes

#TODO: caltech-101 datasets
def caltech101_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    raise NotImplementedError
    train_set = Flowers102(data_path, split='train', transform=train_transforms, download=download, type='train1000')
    val_set = Flowers102(data_path, split='test', transform=val_transforms, download=download, type='test')
    num_classes = 102
    return train_set, val_set, num_classes

def clevr_count_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRClassification(data_path, split='train', transform=train_transforms, download=download, type='train1000')
    val_set = CLEVRClassification(data_path, split='val', transform=val_transforms, download=download, type='val')
    num_classes = 8
    return train_set, val_set, num_classes

def clevr_count_800_200_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRClassification(data_path, split='train', transform=train_transforms, download=download, type='train800')
    val_set = CLEVRClassification(data_path, split='train', transform=val_transforms, download=download, type='val200')
    num_classes = 8
    return train_set, val_set, num_classes

def clevr_count_full_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRClassification(data_path, split='train', transform=train_transforms, download=download, type='all')
    val_set = CLEVRClassification(data_path, split='val', transform=val_transforms, download=download, type='all')
    num_classes = 8
    return train_set, val_set, num_classes

def clevr_distance_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRDistance(data_path, split='train', transform=train_transforms, download=download, type='train1000')
    val_set = CLEVRDistance(data_path, split='val', transform=val_transforms, download=download, type='val')
    num_classes = 6
    return train_set, val_set, num_classes

def clevr_distance_800_200_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRDistance(data_path, split='train', transform=train_transforms, download=download, type='train800')
    val_set = CLEVRDistance(data_path, split='train', transform=val_transforms, download=download, type='val200')
    num_classes = 6
    return train_set, val_set, num_classes

def clevr_distance_full_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRDistance(data_path, split='train', transform=train_transforms, download=download, type='all')
    val_set = CLEVRDistance(data_path, split='val', transform=val_transforms, download=download, type='all')
    num_classes = 6
    return train_set, val_set, num_classes

#TODO: Retinopathy datasets
def retinopathy_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    raise NotImplementedError
    train_set = Flowers102(data_path, split='train', transform=train_transforms, download=download, type='train1000')
    val_set = Flowers102(data_path, split='test', transform=val_transforms, download=download, type='test')
    num_classes = 102
    return train_set, val_set, num_classes

def create_datasets(data_path, train_transforms, val_transforms, name='cifar100', type='1000'):
    if type == '1000':
        if name == 'cifar100':
            return cifar100_1k_datasets(data_path, train_transforms, val_transforms)
    elif type == '800':
        if name == 'cifar100':
            return cifar100_800_200_datasets(data_path, train_transforms, val_transforms)
    elif type == 'full':
        if name == 'cifar100':
            return cifar100_full_datasets(data_path, train_transforms, val_transforms)