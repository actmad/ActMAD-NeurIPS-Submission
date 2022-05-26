import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

te_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(*NORM)
                                    ])

tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*NORM)
                                    ])


common_corruptions = ['cifar_new', 'gaussian_noise', 'original', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


def prepare_test_data(args):
    tesize = 10000
    if args.corruption in common_corruptions:
        print('Test on %s level %d' %(args.corruption, args.level))
        teset_raw = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
        teset_raw = teset_raw[(args.level-1)*tesize: args.level*tesize]
        teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                            train=False, download=True, transform=te_transforms)
        teset_ = torchvision.datasets.CIFAR10(root=args.dataroot,
                                             train=False, download=True, transform=tr_transforms)
        teset.data = teset_raw
        teset_.data = teset_raw
    else:
        raise Exception('Corruption not found!')

    if not hasattr(args, 'workers'):
        args.workers = 1

    teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers)

    teloader_ = torch.utils.data.DataLoader(teset_, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers)

    return teloader_, teloader


def prepare_test_data_fraction(args):
    tesize = 10000
    if args.corruption in common_corruptions:
        print('Test on %s level %d' %(args.corruption, args.level))
        teset_raw = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
        teset_raw = teset_raw[(args.level-1)*tesize: args.level*tesize]
        lent = len(teset_raw) * args.d_size
        x = [random.randint(0, len(teset_raw) - 1) for _ in range(int(lent))]
        teset_raw = teset_raw[x]
        teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                             train=False, download=True, transform=te_transforms)
        teset.data = teset_raw
    else:
        raise Exception('Corruption not found!')

    if not hasattr(args, 'workers'):
        args.workers = 1

    teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers)

    return teset, teloader


def prepare_train_data(args):
    print('Preparing data...')
    trset = torchvision.datasets.CIFAR10(root=args.dataroot, transform=tr_transforms,
                                        train=True, download=True)
    if not hasattr(args, 'workers'):
        args.workers = 1

    trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers)
    return trset, trloader


def prepare_train_data_frac(args):
    print('Preparing data...')
    trset = torchvision.datasets.CIFAR10(root=args.dataroot, transform=tr_transforms,
                                        train=True, download=True)
    lent = int(len(trset) * args.d_size)
    dataset, _ = torch.utils.data.random_split(trset, [lent, len(trset) - lent])

    print(f'Size of Train Set (Fraction): {len(dataset)}')
    if not hasattr(args, 'workers'):
        args.workers = 1

    trloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers)
    return trset, trloader