"""
SCFF Data Loading
Exact implementation matching original SCFF_CIFAR.py
"""
import torch
import torchvision
from torchvision.transforms import transforms, ToPILImage
from torch.utils.data import DataLoader, Subset


# Augmentation strength parameter
s = 0.5

# Dual augmentation transforms (for contrastive learning)
transform1 = transforms.Compose([
    transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform2 = transforms.Compose([
    transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Standard training transform
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Test transform (no augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class DualAugmentCIFAR10(torchvision.datasets.CIFAR10):
    """
    Custom CIFAR-10 dataset that applies dual augmentation techniques 
    for unsupervised SCFF. (default: no augmentation is used) 

    Args:
        root (str): Root directory where the dataset is stored.
        augment (str): Type of augmentation to apply. 
                       Options: 'no' (default), 'single', 'dual'.
        *args: Additional arguments for the CIFAR-10 dataset.

    Attributes:
        augment (str): Stores the selected augmentation mode.
    """
    
    def __init__(self, root, augment="No", *args, **kwargs):
        super(DualAugmentCIFAR10, self).__init__(root, *args, **kwargs)
        self.augment = augment
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img_pil = ToPILImage()(img)
        img_original = transform_train(img_pil)

        if self.augment == "single":
            img1 = transform1(img_pil)
            return img_original, img1, img_original, target
        elif self.augment == "dual":
            img1 = transform1(img_pil)
            img2 = transform2(img_pil)
            return img_original, img1, img2, target
        else:
            return img_original, target


class DualAugmentCIFAR10_test(torchvision.datasets.CIFAR10):
    """
    Custom CIFAR-10 dataset that applies augmentation techniques 
    for supervised evaluation of the trained model with SCFF.

    Args:
        aug (bool): Whether to apply data augmentation to test images.
        *args: Additional arguments for the CIFAR-10 dataset.

    Attributes:
        aug (bool): Stores whether augmentation is applied. True for train set, False for test set
    """
    
    def __init__(self, aug=False, *args, **kwargs):
        super(DualAugmentCIFAR10_test, self).__init__(*args, **kwargs)
        self.aug = aug
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = ToPILImage()(img)
        
        if self.aug:
            img = transform_train(img)
        else:
            img = transform_test(img)
        
        return img, target


def get_train(batchsize, augment, Factor, seed=1234):
    """
    Creates data loaders for CIFAR-10 training and validation.

    Args:
        batchsize (int): Batch size for training.
        augment (str): Data augmentation strategy (e.g., 'no', 'single', 'dual').
        Factor (float): Proportion of dataset to use for training.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, val_loader, test_loader, sup_train_loader)
    """
    torch.manual_seed(seed)
    
    trainset = DualAugmentCIFAR10(root='./data', train=True, download=True, augment=augment)
    sup_trainset = DualAugmentCIFAR10_test(root='./data', aug=True, train=True, download=True)
    
    # Create a DataLoader
    factor = Factor
    train_len = int(len(trainset) * factor)

    indices = torch.randperm(len(trainset)).tolist()
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    # Create subsets
    train_data = Subset(trainset, train_indices)
    sup_train_data = Subset(sup_trainset, train_indices)
    val_data = Subset(sup_trainset, val_indices)

    testset = DualAugmentCIFAR10_test(root='./data', aug=False, train=False, download=True)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

    trainloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2)

    if factor == 1:
        valloader = testloader
    else:
        valloader = DataLoader(val_data, batch_size=1000, shuffle=True, num_workers=2)

    sup_trainloader = DataLoader(sup_train_data, batch_size=64, shuffle=True)

    return trainloader, valloader, testloader, sup_trainloader
