from __future__ import print_function
from argparse import Namespace
import torch.optim as optim
import argparse
from utils.test_helpers import test
from utils.inc_dataloader import *
from torchvision import models
import numpy as np
from utils.get_methods import *
from utils.misc import *
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='PATH_TO_FOLDER')
parser.add_argument('--model', default='')
parser.add_argument('--models', default=list)
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--batch_size', default=250, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
########################################################################
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--save_embeddings', default=False, type=bool)
parser.add_argument('--adapt', default=False, type=bool)
parser.add_argument('--milestone_1', default=3, type=int)
parser.add_argument('--milestone_2', default=5, type=int)
parser.add_argument('--nepoch', default=5, type=int)
parser.add_argument('--d_size', default=0.025, type=float)
parser.add_argument('--workers', default=8, type=int)
########################################################################
parser.add_argument('--outf', default='.')
args: Namespace = parser.parse_args()
activation = {}

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness',
                      'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
severity = [5, 4, 3, 2, 1]
d_size_ = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
b_size_ = [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]


def load_np(i):
    act = np.load(f"stats/stats_{args.model}/mean_{args.d_size}/original_{i}.npy")
    return act


def load_np_var(i):
    act = np.load(f"stats/stats_{args.model}/var_{args.d_size}/original_{i}.npy")
    return act


default_lr = args.lr

for args.model in args.models:
    print(f'Starting Adaptation for: {args.model}')
    if args.model == 'deep_aug':
        state_dict = torch.load(
            'ckpt/deepaugment_and_augmix.pth.tar')
        state_dict = get_state_dict(state_dict)
        for args.dsize in d_size_:
            print(f'Dataset Size ::: {args.dsize * 100}%')
            for args.batch_size in b_size_:
                print(f'Batch Size ::: {args.batch_size}')
                act_clean_3 = torch.tensor(load_np(3)).cuda()
                act_clean_7 = torch.tensor(load_np(7)).cuda()
                act_clean_8 = torch.tensor(load_np(8)).cuda()
                act_clean_9 = torch.tensor(load_np(9)).cuda()
                act_clean_10 = torch.tensor(load_np(10)).cuda()
                act_clean_11 = torch.tensor(load_np(11)).cuda()
                act_clean_3_var = torch.tensor(load_np_var(3)).cuda()
                act_clean_7_var = torch.tensor(load_np_var(7)).cuda()
                act_clean_8_var = torch.tensor(load_np_var(8)).cuda()
                act_clean_9_var = torch.tensor(load_np_var(9)).cuda()
                act_clean_10_var = torch.tensor(load_np_var(10)).cuda()
                act_clean_11_var = torch.tensor(load_np_var(11)).cuda()

                for args.level in severity:
                    all_res = []
                    for args.corruption in common_corruptions:
                        net = models.resnet50(pretrained=False)
                        net = net.cuda()
                        net.load_state_dict(state_dict)
                        err_corr = []
                        parameters = list(net.parameters())

                        # LR scaling ( [Goyal et al.] https://arxiv.org/pdf/1706.02677.pdf)
                        args.lr = default_lr * args.batch_size / 250

                        optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
                        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                            optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)

                        te_loader = prepare_test_data(args)
                        te_loader_frac = prepare_test_data_frac(args)
                        err_cls = test(te_loader, net)
                        print('Error before adaptation: ', 100 - (err_cls * 100))
                        print('Epoch \t\t Loss \t\t Error(%)')
                        for epoch in range(1, args.nepoch + 1):
                            loss_arr = list()

                            net.train()

                            for name, param in net.named_parameters():
                                for m in net.modules():
                                    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                                    nn.BatchNorm3d):
                                        m.eval()

                            for idx, (inputs, labels) in enumerate(te_loader_frac):
                                optimizer.zero_grad()
                                save_output_3, save_output_7, save_output_8, save_output_9, save_output_10, save_output_11 \
                                    = SaveOutput(), SaveOutput(), \
                                      SaveOutput(), SaveOutput(), \
                                      SaveOutput(), SaveOutput()

                                hook_3 = net.layer2[3].bn3.register_forward_hook(save_output_3)
                                hook_7 = net.layer3[4].bn3.register_forward_hook(save_output_7)
                                hook_8 = net.layer3[5].bn3.register_forward_hook(save_output_8)
                                hook_9 = net.layer4[0].bn3.register_forward_hook(save_output_9)
                                hook_10 = net.layer4[1].bn3.register_forward_hook(save_output_10)
                                hook_11 = net.layer4[2].bn3.register_forward_hook(save_output_11)

                                inputs = inputs.cuda()
                                _ = net(inputs)

                                out_3 = get_out(save_output_3)
                                out_7 = get_out(save_output_7)
                                out_8 = get_out(save_output_8)
                                out_9 = get_out(save_output_9)
                                out_10 = get_out(save_output_10)
                                out_11 = get_out(save_output_11)

                                out_3_var = get_out_var(save_output_3)
                                out_7_var = get_out_var(save_output_7)
                                out_8_var = get_out_var(save_output_8)
                                out_9_var = get_out_var(save_output_9)
                                out_10_var = get_out_var(save_output_10)
                                out_11_var = get_out_var(save_output_11)

                                loss_mean = l1_loss(out_3, act_clean_3) + l1_loss(out_7, act_clean_7) + \
                                            l1_loss(out_8, act_clean_8) + l1_loss(out_9, act_clean_9) + \
                                            l1_loss(out_10, act_clean_10) + l1_loss(out_11, act_clean_11)

                                loss_var = l1_loss(out_3_var, act_clean_3_var) + l1_loss(out_7_var, act_clean_7_var) \
                                           + l1_loss(out_8_var, act_clean_8_var) + l1_loss(out_9_var, act_clean_9_var) \
                                           + l1_loss(out_10_var, act_clean_10_var) + l1_loss(out_11_var, act_clean_11_var)

                                loss = loss_mean + loss_var
                                loss_arr.append(loss)
                                loss.backward()
                                optimizer.step()

                                save_output_3.clear(), save_output_7.clear(), save_output_8.clear(), save_output_9.clear(), \
                                save_output_10.clear(), save_output_11.clear()

                                hook_3.remove(), hook_7.remove(), hook_8.remove(), hook_9.remove(), \
                                hook_10.remove(), hook_11.remove()
                            l_m = mean(loss_arr)
                            err_cls = 100 - (test(te_loader, net) * 100)
                            err_corr.append(err_cls)
                            print(f'{epoch} \t\t {l_m: .1f} \t\t {err_cls: .1f}')
                        min_err_corr = min(err_corr)
                        print(f'Minimum Error after adaptation - {args.corruption}_{args.level}: ', min_err_corr)
                        all_res.append(min_err_corr)
                    print(f'Mean Error Level_{args.level}: ', sum(all_res) / len(all_res))
                    # save results
                    Path(f'results/imagenet/{args.model}/').mkdir(exist_ok=True, parents=True)
                    np.save(f'results/imagenet/{args.model}/{args.level}_{args.batch_size}_{args.d_size}.npy', all_res)

    if args.model == 'res_50':
        for args.dsize in d_size_:
            print(f'Dataset Size ::: {args.dsize * 100}%')
            for args.batch_size in b_size_:
                print(f'Batch Size ::: {args.batch_size}')
                act_clean_3 = torch.tensor(load_np(3)).cuda()
                act_clean_7 = torch.tensor(load_np(7)).cuda()
                act_clean_8 = torch.tensor(load_np(8)).cuda()
                act_clean_9 = torch.tensor(load_np(9)).cuda()
                act_clean_10 = torch.tensor(load_np(10)).cuda()
                act_clean_11 = torch.tensor(load_np(11)).cuda()
                act_clean_3_var = torch.tensor(load_np_var(3)).cuda()
                act_clean_7_var = torch.tensor(load_np_var(7)).cuda()
                act_clean_8_var = torch.tensor(load_np_var(8)).cuda()
                act_clean_9_var = torch.tensor(load_np_var(9)).cuda()
                act_clean_10_var = torch.tensor(load_np_var(10)).cuda()
                act_clean_11_var = torch.tensor(load_np_var(11)).cuda()

                for args.level in severity:
                    all_res = []
                    for args.corruption in common_corruptions:
                        net = models.resnet50(pretrained=True)
                        net = net.cuda()
                        err_corr = []
                        parameters = list(net.parameters())

                        # LR scaling ( [Goyal et al.] https://arxiv.org/pdf/1706.02677.pdf)
                        args.lr = default_lr * args.batch_size / 250

                        optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
                        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                            optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)

                        te_loader = prepare_test_data(args)
                        te_loader_frac = prepare_test_data_frac(args)
                        err_cls = test(te_loader, net)
                        print('Error before adaptation: ', 100 - (err_cls * 100))
                        print('Epoch \t\t Loss \t\t Error(%)')
                        for epoch in range(1, args.nepoch + 1):
                            loss_arr = list()

                            net.train()

                            for name, param in net.named_parameters():
                                for m in net.modules():
                                    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                                    nn.BatchNorm3d):
                                        m.eval()

                            for idx, (inputs, labels) in enumerate(te_loader_frac):
                                optimizer.zero_grad()
                                save_output_3, save_output_7, save_output_8, save_output_9, save_output_10, save_output_11 \
                                    = SaveOutput(), SaveOutput(), \
                                      SaveOutput(), SaveOutput(), \
                                      SaveOutput(), SaveOutput()

                                hook_3 = net.layer2[3].bn3.register_forward_hook(save_output_3)
                                hook_7 = net.layer3[4].bn3.register_forward_hook(save_output_7)
                                hook_8 = net.layer3[5].bn3.register_forward_hook(save_output_8)
                                hook_9 = net.layer4[0].bn3.register_forward_hook(save_output_9)
                                hook_10 = net.layer4[1].bn3.register_forward_hook(save_output_10)
                                hook_11 = net.layer4[2].bn3.register_forward_hook(save_output_11)

                                inputs = inputs.cuda()
                                _ = net(inputs)

                                out_3 = get_out(save_output_3)
                                out_7 = get_out(save_output_7)
                                out_8 = get_out(save_output_8)
                                out_9 = get_out(save_output_9)
                                out_10 = get_out(save_output_10)
                                out_11 = get_out(save_output_11)

                                out_3_var = get_out_var(save_output_3)
                                out_7_var = get_out_var(save_output_7)
                                out_8_var = get_out_var(save_output_8)
                                out_9_var = get_out_var(save_output_9)
                                out_10_var = get_out_var(save_output_10)
                                out_11_var = get_out_var(save_output_11)

                                loss_mean = l1_loss(out_3, act_clean_3) + l1_loss(out_7, act_clean_7) + \
                                            l1_loss(out_8, act_clean_8) + l1_loss(out_9, act_clean_9) + \
                                            l1_loss(out_10, act_clean_10) + l1_loss(out_11, act_clean_11)

                                loss_var = l1_loss(out_3_var, act_clean_3_var) + l1_loss(out_7_var, act_clean_7_var) \
                                           + l1_loss(out_8_var, act_clean_8_var) + l1_loss(out_9_var, act_clean_9_var) \
                                           + l1_loss(out_10_var, act_clean_10_var) + l1_loss(out_11_var, act_clean_11_var)

                                loss = loss_mean + loss_var
                                loss_arr.append(loss)
                                loss.backward()
                                optimizer.step()

                                save_output_3.clear(), save_output_7.clear(), save_output_8.clear(), save_output_9.clear(), \
                                save_output_10.clear(), save_output_11.clear()

                                hook_3.remove(), hook_7.remove(), hook_8.remove(), hook_9.remove(), \
                                hook_10.remove(), hook_11.remove()
                            l_m = mean(loss_arr)
                            err_cls = 100 - (test(te_loader, net) * 100)
                            err_corr.append(err_cls)
                            print(f'{epoch} \t\t {l_m: .1f} \t\t {err_cls: .1f}')
                        min_err_corr = min(err_corr)
                        print(f'Minimum Error after adaptation - {args.corruption}_{args.level}: ', min_err_corr)
                        all_res.append(min_err_corr)
                    print(f'Mean Error Level_{args.level}: ', sum(all_res) / len(all_res))
                    # save results
                    Path(f'results/imagenet/{args.model}/').mkdir(exist_ok=True, parents=True)
                    np.save(f'results/imagenet/{args.model}/{args.level}_{args.batch_size}_{args.d_size}.npy', all_res)

    if args.model == 'res_18':
        for args.dsize in d_size_:
            print(f'Dataset Size ::: {args.dsize * 100}%')
            for args.batch_size in b_size_:
                print(f'Batch Size ::: {args.batch_size}')
                act_clean_1 = torch.tensor(load_np(1)).cuda()
                act_clean_2 = torch.tensor(load_np(2)).cuda()
                act_clean_3 = torch.tensor(load_np(3)).cuda()
                act_clean_4 = torch.tensor(load_np(4)).cuda()
                act_clean_5 = torch.tensor(load_np(5)).cuda()
                act_clean_6 = torch.tensor(load_np(6)).cuda()
                act_clean_7 = torch.tensor(load_np(7)).cuda()
                act_clean_8 = torch.tensor(load_np(8)).cuda()

                act_clean_1_var = torch.tensor(load_np_var(1)).cuda()
                act_clean_2_var = torch.tensor(load_np_var(2)).cuda()
                act_clean_3_var = torch.tensor(load_np_var(3)).cuda()
                act_clean_4_var = torch.tensor(load_np_var(4)).cuda()
                act_clean_5_var = torch.tensor(load_np_var(5)).cuda()
                act_clean_6_var = torch.tensor(load_np_var(6)).cuda()
                act_clean_7_var = torch.tensor(load_np_var(7)).cuda()
                act_clean_8_var = torch.tensor(load_np_var(8)).cuda()

                for args.level in severity:
                    all_res = []
                    for args.corruption in common_corruptions:
                        net = models.resnet18(pretrained=True)
                        net = net.cuda()
                        err_corr = []
                        parameters = list(net.parameters())

                        # LR scaling ( [Goyal et al.] https://arxiv.org/pdf/1706.02677.pdf)
                        args.lr = default_lr * args.batch_size / 250

                        optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
                        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                            optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)
                        te_loader_frac = prepare_test_data_frac(args)
                        te_loader = prepare_test_data(args)
                        err_cls = test(te_loader, net)
                        print('Error before adaptation: ', 100 - (err_cls * 100))
                        print('Epoch \t\t Loss \t\t Error(%)')
                        for epoch in range(1, args.nepoch + 1):
                            loss_arr = list()
                            net.train()

                            for name, param in net.named_parameters():
                                for m in net.modules():
                                    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                                    nn.BatchNorm3d):
                                        m.eval()

                            for idx, (inputs, labels) in enumerate(te_loader_frac):
                                optimizer.zero_grad()
                                save_output_1, save_output_2, save_output_3, save_output_4, save_output_5, \
                                save_output_6, save_output_7, save_output_8 \
                                    = SaveOutput(), SaveOutput(), \
                                      SaveOutput(), SaveOutput(), \
                                      SaveOutput(), SaveOutput(), SaveOutput(), SaveOutput()

                                hook_1 = net.layer1[0].bn2.register_forward_hook(save_output_1)
                                hook_2 = net.layer1[1].bn2.register_forward_hook(save_output_2)
                                hook_3 = net.layer2[0].bn2.register_forward_hook(save_output_3)
                                hook_4 = net.layer2[1].bn2.register_forward_hook(save_output_4)
                                hook_5 = net.layer3[0].bn2.register_forward_hook(save_output_5)
                                hook_6 = net.layer3[1].bn2.register_forward_hook(save_output_6)
                                hook_7 = net.layer4[0].bn2.register_forward_hook(save_output_7)
                                hook_8 = net.layer4[1].bn2.register_forward_hook(save_output_8)

                                inputs = inputs.cuda()
                                _ = net(inputs)

                                out_1 = get_out(save_output_1)
                                out_2 = get_out(save_output_2)
                                out_3 = get_out(save_output_3)
                                out_4 = get_out(save_output_4)
                                out_5 = get_out(save_output_5)
                                out_6 = get_out(save_output_6)
                                out_7 = get_out(save_output_7)
                                out_8 = get_out(save_output_8)

                                out_1_var = get_out_var(save_output_1)
                                out_2_var = get_out_var(save_output_2)
                                out_3_var = get_out_var(save_output_3)
                                out_4_var = get_out_var(save_output_4)
                                out_5_var = get_out_var(save_output_5)
                                out_6_var = get_out_var(save_output_6)
                                out_7_var = get_out_var(save_output_7)
                                out_8_var = get_out_var(save_output_8)

                                loss_mean = l1_loss(out_1, act_clean_1) + l1_loss(out_2, act_clean_2) \
                                            + l1_loss(out_3, act_clean_3) + l1_loss(out_4, act_clean_4) \
                                            + l1_loss(out_5, act_clean_5) + l1_loss(out_6, act_clean_6) + \
                                            l1_loss(out_7, act_clean_7) + l1_loss(out_8, act_clean_8)

                                loss_var = l1_loss(out_1_var, act_clean_1_var) + l1_loss(out_2_var, act_clean_2_var) + l1_loss(
                                    out_3_var, act_clean_3_var) + l1_loss(out_4_var, act_clean_4_var) + \
                                           l1_loss(out_5_var, act_clean_5_var) + l1_loss(out_6_var, act_clean_6_var) + l1_loss(
                                    out_7_var, act_clean_7_var) + l1_loss(out_8_var, act_clean_8_var)

                                loss = loss_mean + loss_var
                                loss_arr.append(loss)
                                loss.backward()
                                optimizer.step()

                                save_output_1.clear(), save_output_2.clear(), save_output_3.clear(), save_output_4.clear(), \
                                save_output_5.clear(), save_output_6.clear(), save_output_7.clear(), save_output_8.clear()

                                hook_1.remove(), hook_2.remove(), hook_3.remove(), hook_4.remove(), hook_5.remove(), \
                                hook_6.remove(), hook_7.remove(), hook_8.remove()

                            err_cls = 100 - (test(te_loader, net) * 100)
                            err_corr.append(err_cls)
                            print(f'{epoch} \t\t {mean(loss_arr): .1f} \t\t {err_cls: .1f}')

                        min_err_corr = min(err_corr)
                        print(f'Minimum Error after adaptation - {args.corruption}_{args.level}: ', min_err_corr)
                        all_res.append(min_err_corr)
                    print(f'Mean Error Level_{args.level}: ', sum(all_res) / len(all_res))
                    # save results
                    Path(f'results/imagenet/{args.model}/').mkdir(exist_ok=True, parents=True)
                    np.save(f'results/imagenet/{args.model}/{args.level}_{args.batch_size}_{args.d_size}.npy', all_res)

