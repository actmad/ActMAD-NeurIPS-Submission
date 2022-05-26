import os
import torch
import torchvision.transforms as transforms
import torch.utils.data
import torchvision

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tr_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize
                                    ])

te_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize
                                    ])

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


def prepare_train_data(args):
    print('Preparing train data...')
    dataset = torchvision.datasets.ImageNet(args.dataroot, split='train', transform=tr_transforms, download='False')
    trloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)
    return trloader


def prepare_train_data_frac(args):
    print('Preparing train data (Fraction)...')

    dataset = torchvision.datasets.ImageNet(args.dataroot, split='train', transform=tr_transforms)

    lent = int(len(dataset) * args.d_size)
    dataset, _ = torch.utils.data.random_split(dataset, [lent, len(dataset) - lent])
    print(f'Size of Train Set (Fraction): {len(dataset)}')

    trloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)
    return trloader


def prepare_test_data(args):
    if args.corruption in common_corruptions:
        print('Test on %s level %d' % (args.corruption, args.level))
        validdir = os.path.join(args.dataroot, args.corruption, str(args.level))
        imagenet_data = torchvision.datasets.ImageFolder(validdir, transform=te_transforms)
        print(f'Size of Test Set: {len(imagenet_data)}')
    else:
        raise Exception('Corruption not found!')

    if not hasattr(args, 'workers'):
        args.workers = 1

    # fix batchsize for testing
    teloader = torch.utils.data.DataLoader(imagenet_data, batch_size=250, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)
    return teloader


def prepare_test_data_frac(args):
    # always used for adaptation
    if args.corruption in common_corruptions:
        print('Test Adapt on %s level %d' % (args.corruption, args.level))
        validdir = os.path.join(args.dataroot, args.corruption, str(args.level))
        imagenet_data = torchvision.datasets.ImageFolder(validdir, transform=te_transforms)
        lent = int(len(imagenet_data) * args.d_size)
        imagenet_data, _ = torch.utils.data.random_split(imagenet_data, [lent, 50000 - lent])
        print(f'Size of Adapt Test Set: {len(imagenet_data)}')
    else:
        raise Exception('Corruption not found!')

    if not hasattr(args, 'workers'):
        args.workers = 1
    # batchsize used from main file
    teloader = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)

    return teloader
