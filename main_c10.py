from __future__ import print_function
import argparse
from argparse import Namespace
from utils.c10_dataloader import *
from pathlib import Path
import torch.optim as optim
from utils.test_helpers import *
import torch.nn as nn
from models.build import build_model
from utils.get_methods import *
from models.wide_resnet import WideResNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default='./test-time-training')
parser.add_argument('--model', default='res_26')
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--batch_size', default=250, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--adapt', default=False, type=bool)
parser.add_argument('--save_model', default=False, type=bool)
parser.add_argument('--milestone_1', default=3, type=int)
parser.add_argument('--milestone_2', default=6, type=int)
parser.add_argument('--nepoch', default=10, type=int)
parser.add_argument('--d_size', default=1.0, type=float)
########################################################################
parser.add_argument('--outf', default='.')
args: Namespace = parser.parse_args()

common_corruptions = ['gaussian_noise',
                      'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness',
                      'contrast',
                      'elastic_transform',
                      'pixelate',
                      'jpeg_compression'
                      ]

models_ = ['res_26', 'augmix']

severity = [5, 4, 3, 2, 1]

for args.model in models_:
    if args.model == 'res_26':
        print(f'Backbone ::: {args.model}')
        net = build_model(args)
        net = net.cuda()
        ckpt = torch.load('ckpt' + '/ckpt_c10_bn.pth')
        net.load_state_dict(ckpt['net'])

        all_res = []
        save_output_1, save_output_2, save_output_3, \
        save_output_4, save_output_5, save_output_6, save_output_7, \
        save_output_8, save_output_9, save_output_10, save_output_11, \
            = SaveOutput(), SaveOutput(), SaveOutput(), SaveOutput(), \
              SaveOutput(), SaveOutput(), SaveOutput(), SaveOutput(), \
              SaveOutput(), SaveOutput(), SaveOutput()

        clean_1_list, clean_2_list, clean_3_list, clean_4_list, \
        clean_5_list, clean_6_list, clean_7_list, clean_9_list, \
        clean_10_list, clean_8_list, clean_11_list = list(), list(), list(), \
                                                     list(), list(), list(), list(), list(), list(), list(), list()

        clean_1_list_var, clean_2_list_var, clean_3_list_var, \
        clean_4_list_var, clean_5_list_var, clean_6_list_var, \
        clean_7_list_var, clean_9_list_var, clean_10_list_var, \
        clean_8_list_var, clean_11_list_var = list(), list(), list(), list(), list(), list(), \
                                              list(), list(), list(), list(), list()

        _, tr_loader = prepare_train_data(args)
        for idx, (inputs, labels) in enumerate(tr_loader):
            hook_1 = net.layer1[1].bn2.register_forward_hook(save_output_1)
            hook_2 = net.layer1[2].bn2.register_forward_hook(save_output_2)
            hook_3 = net.layer1[3].bn2.register_forward_hook(save_output_3)

            hook_4 = net.layer2[1].bn2.register_forward_hook(save_output_4)
            hook_5 = net.layer2[2].bn2.register_forward_hook(save_output_5)
            hook_6 = net.layer2[3].bn2.register_forward_hook(save_output_6)

            hook_11 = net.layer3[0].bn2.register_forward_hook(save_output_11)
            hook_7 = net.layer3[1].bn2.register_forward_hook(save_output_7)
            hook_8 = net.layer3[2].bn2.register_forward_hook(save_output_8)
            hook_9 = net.layer3[3].bn2.register_forward_hook(save_output_9)

            hook_10 = net.bn.register_forward_hook(save_output_10)

            inputs = inputs.cuda()
            with torch.no_grad():
                output = net(inputs)

            clean_1_list_var.append(get_clean_out_var(save_output_1))
            clean_2_list_var.append(get_clean_out_var(save_output_2))
            clean_3_list_var.append(get_clean_out_var(save_output_3))
            clean_4_list_var.append(get_clean_out_var(save_output_4))
            clean_5_list_var.append(get_clean_out_var(save_output_5))
            clean_6_list_var.append(get_clean_out_var(save_output_6))
            clean_7_list_var.append(get_clean_out_var(save_output_7))
            clean_8_list_var.append(get_clean_out_var(save_output_8))
            clean_9_list_var.append(get_clean_out_var(save_output_9))
            clean_10_list_var.append(get_clean_out_var(save_output_10))
            clean_11_list_var.append(get_clean_out_var(save_output_11))

            clean_1_list.append(get_clean_out(save_output_1))
            clean_2_list.append(get_clean_out(save_output_2))
            clean_3_list.append(get_clean_out(save_output_3))
            clean_4_list.append(get_clean_out(save_output_4))
            clean_5_list.append(get_clean_out(save_output_5))
            clean_6_list.append(get_clean_out(save_output_6))
            clean_7_list.append(get_clean_out(save_output_7))
            clean_8_list.append(get_clean_out(save_output_8))
            clean_9_list.append(get_clean_out(save_output_9))
            clean_10_list.append(get_clean_out(save_output_10))
            clean_11_list.append(get_clean_out(save_output_11))

            save_output_1.clear(), save_output_2.clear(), save_output_3.clear(), save_output_4.clear(), save_output_5.clear(), \
            save_output_6.clear(), save_output_7.clear(), save_output_8.clear(), save_output_9.clear(), save_output_10.clear(), save_output_11.clear()

            hook_1.remove(), hook_2.remove(), hook_3.remove(), hook_4.remove(), hook_5.remove(), hook_6.remove(), \
            hook_7.remove(), hook_8.remove(), hook_9.remove(), hook_10.remove(), hook_11.remove()

        act_clean_1 = take_mean(torch.stack(clean_1_list))
        act_clean_2 = take_mean(torch.stack(clean_2_list))
        act_clean_3 = take_mean(torch.stack(clean_3_list))
        act_clean_4 = take_mean(torch.stack(clean_4_list))
        act_clean_5 = take_mean(torch.stack(clean_5_list))
        act_clean_6 = take_mean(torch.stack(clean_6_list))
        act_clean_7 = take_mean(torch.stack(clean_7_list))
        act_clean_8 = take_mean(torch.stack(clean_8_list))
        act_clean_9 = take_mean(torch.stack(clean_9_list))
        act_clean_10 = take_mean(torch.stack(clean_10_list))
        act_clean_11 = take_mean(torch.stack(clean_11_list))

        act_clean_1_var = take_mean(torch.stack(clean_1_list_var))
        act_clean_2_var = take_mean(torch.stack(clean_2_list_var))
        act_clean_3_var = take_mean(torch.stack(clean_3_list_var))
        act_clean_4_var = take_mean(torch.stack(clean_4_list_var))
        act_clean_5_var = take_mean(torch.stack(clean_5_list_var))
        act_clean_6_var = take_mean(torch.stack(clean_6_list_var))
        act_clean_7_var = take_mean(torch.stack(clean_7_list_var))
        act_clean_8_var = take_mean(torch.stack(clean_8_list_var))
        act_clean_9_var = take_mean(torch.stack(clean_9_list_var))
        act_clean_10_var = take_mean(torch.stack(clean_10_list_var))
        act_clean_11_var = take_mean(torch.stack(clean_11_list_var))
        for args.level in severity:
            all_res = list()
            print(f'Starting adaptation for Level {args.level}...')
            for args.corruption in common_corruptions:
                net.load_state_dict(ckpt['net'])
                err_corr = []
                param_to_opt = list(net.parameters())
                optimizer = optim.SGD(param_to_opt, lr=args.lr, momentum=0.9, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)

                te_loader_, te_loader = prepare_test_data(args)

                err_cls = 100 - (test(te_loader, net) * 100)
                print(f'Error before adaptation: {err_cls: .1f}')
                print('Epoch \t\t Loss \t\t Error(%)')
                for epoch in range(1, args.nepoch + 1):
                    loss_arr = list()
                    net.train()

                    for name, param in net.named_parameters():
                        for m in net.modules():
                            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                            nn.BatchNorm3d):
                                m.eval()

                    for idx, (inputs, labels) in enumerate(te_loader):
                        optimizer.zero_grad()
                        save_output_1, save_output_2, save_output_3, save_output_4, save_output_5, save_output_6, \
                        save_output_7, save_output_8, save_output_9, save_output_10, \
                        save_output_11 = SaveOutput(), SaveOutput(), \
                                         SaveOutput(), SaveOutput(), \
                                         SaveOutput(), SaveOutput(), SaveOutput(), \
                                         SaveOutput(), SaveOutput(), SaveOutput(), \
                                         SaveOutput()

                        hook_1 = net.layer1[1].bn2.register_forward_hook(save_output_1)
                        hook_2 = net.layer1[2].bn2.register_forward_hook(save_output_2)
                        hook_3 = net.layer1[3].bn2.register_forward_hook(save_output_3)

                        hook_4 = net.layer2[1].bn2.register_forward_hook(save_output_4)
                        hook_5 = net.layer2[2].bn2.register_forward_hook(save_output_5)
                        hook_6 = net.layer2[3].bn2.register_forward_hook(save_output_6)

                        hook_7 = net.layer3[1].bn2.register_forward_hook(save_output_7)
                        hook_8 = net.layer3[2].bn2.register_forward_hook(save_output_8)
                        hook_9 = net.layer3[3].bn2.register_forward_hook(save_output_9)

                        hook_11 = net.layer3[0].bn2.register_forward_hook(save_output_11)
                        hook_10 = net.bn.register_forward_hook(save_output_10)

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
                        out_9 = get_out(save_output_9)
                        out_10 = get_out(save_output_10)
                        out_11 = get_out(save_output_11)

                        out_1_var = get_out_var(save_output_1)
                        out_2_var = get_out_var(save_output_2)
                        out_3_var = get_out_var(save_output_3)
                        out_4_var = get_out_var(save_output_4)
                        out_5_var = get_out_var(save_output_5)
                        out_6_var = get_out_var(save_output_6)
                        out_7_var = get_out_var(save_output_7)
                        out_8_var = get_out_var(save_output_8)
                        out_9_var = get_out_var(save_output_9)
                        out_10_var = get_out_var(save_output_10)
                        out_11_var = get_out_var(save_output_11)

                        loss_mean = l1_loss(out_1, act_clean_1) + l1_loss(out_2, act_clean_2) + \
                                    l1_loss(out_3, act_clean_3) + l1_loss(out_4, act_clean_4) + \
                                    l1_loss(out_5, act_clean_5) + l1_loss(out_6, act_clean_6) + \
                                    l1_loss(out_7, act_clean_7) + l1_loss(out_8, act_clean_8) + \
                                    l1_loss(out_9, act_clean_9) + l1_loss(out_10, act_clean_10) + \
                                    l1_loss(out_11, act_clean_11)

                        loss_var = l1_loss(out_1_var, act_clean_1_var) + l1_loss(out_2_var, act_clean_2_var) \
                                   + l1_loss(out_3_var, act_clean_3_var) + l1_loss(out_4_var, act_clean_4_var) \
                                   + l1_loss(out_5_var, act_clean_5_var) + l1_loss(out_6_var, act_clean_6_var) \
                                   + l1_loss(out_7_var, act_clean_7_var) + l1_loss(out_8_var, act_clean_8_var) + \
                                   l1_loss(out_9_var, act_clean_9_var) + l1_loss(out_10_var, act_clean_10_var) + \
                                   l1_loss(out_11_var, act_clean_11_var)

                        loss = loss_mean * 0.5 + loss_var * 0.5

                        loss_arr.append(loss)
                        loss.backward()
                        optimizer.step()

                        save_output_1.clear(), save_output_2.clear(), save_output_3.clear(), save_output_4.clear(), \
                        save_output_5.clear(), save_output_6.clear(), save_output_7.clear(), \
                        save_output_8.clear(), save_output_9.clear(),
                        save_output_10.clear(), save_output_11.clear()

                        hook_1.remove(), hook_2.remove(), hook_3.remove(), hook_4.remove(), hook_5.remove(), hook_6.remove(), \
                        hook_7.remove(), hook_8.remove(), hook_9.remove(), hook_10.remove(), hook_11.remove()

                    l_m = mean(loss_arr)
                    err_cls = 100 - (test(te_loader, net) * 100)
                    err_corr.append(err_cls)
                    scheduler.step()
                    print(f'{epoch} \t\t {l_m:.3f} \t\t {err_cls: .1f}')

                    if args.save_model:
                        if err_cls <= min(err_corr):
                            state = {
                                'net': net.state_dict()
                            }
                            Path(f"c10_adapted_models/{args.corruption}/").mkdir(parents=True,
                                                                                 exist_ok=True)
                            torch.save(state, f"c10_adapted_models/" + str(args.corruption) + '.pth')

                min_err_corr = min(err_corr)
                print(f'Minimum Error after adaptation: {min_err_corr: .1f}')

                all_res.append(min_err_corr)
            print('Mean Error: ', sum(all_res) / len(all_res))
            Path(f'results/cifar10/{args.model}/').mkdir(exist_ok=True, parents=True)
            np.save(f'results/cifar10/{args.model}/{args.level}.npy', all_res)

    if args.model == 'augmix':
        print(f'Backbone ::: {args.model}')
        all_res = []
        net = WideResNet(widen_factor=2, num_classes=10, depth=40)

        net = net.cuda()
        ckpt = torch.load('ckpt/augmix_c10.pt')
        net.load_state_dict(ckpt)
        save_output_1, save_output_2, save_output_3, save_output_4, save_output_5, save_output_6, \
        save_output_7, save_output_8, save_output_9, save_output_10, save_output_11, save_output_12, save_output_13 = SaveOutput(), SaveOutput(), SaveOutput(), \
                                                                                                                      SaveOutput(), SaveOutput(), SaveOutput(), \
                                                                                                                      SaveOutput(), SaveOutput(), SaveOutput(), \
                                                                                                                      SaveOutput(), SaveOutput(), SaveOutput(), SaveOutput()

        clean_1_list, clean_2_list, clean_3_list, clean_4_list, \
        clean_5_list, clean_6_list, clean_7_list, clean_9_list, \
        clean_10_list, clean_8_list, clean_11_list = list(), list(), \
                                                     list(), list(), list(), list(), \
                                                     list(), list(), list(), list(), list()
        clean_1_list_var, clean_2_list_var, clean_3_list_var, \
        clean_4_list_var, clean_5_list_var, clean_6_list_var, \
        clean_7_list_var, clean_9_list_var, clean_10_list_var, clean_8_list_var, \
        clean_11_list_var = list(), list(), list(), \
                            list(), list(), list(), list(), list(), list(), list(), list()

        args.corruption = 'original'
        _, tr_loader = prepare_train_data(args)
        for idx, (inputs, labels) in enumerate(tr_loader):
            hook_1 = net.block1.layer[4].bn2.register_forward_hook(save_output_1)
            hook_2 = net.block1.layer[3].bn2.register_forward_hook(save_output_2)
            hook_3 = net.block1.layer[5].bn2.register_forward_hook(save_output_3)

            hook_4 = net.block2.layer[4].bn2.register_forward_hook(save_output_4)
            hook_5 = net.block2.layer[3].bn2.register_forward_hook(save_output_5)
            hook_6 = net.block2.layer[5].bn2.register_forward_hook(save_output_6)

            hook_7 = net.block3.layer[4].bn2.register_forward_hook(save_output_7)
            hook_8 = net.block3.layer[3].bn2.register_forward_hook(save_output_8)
            hook_9 = net.block3.layer[5].bn2.register_forward_hook(save_output_9)
            hook_11 = net.block3.layer[2].bn2.register_forward_hook(save_output_11)

            hook_10 = net.bn1.register_forward_hook(save_output_10)

            inputs = inputs.cuda()
            with torch.no_grad():
                output = net(inputs)

                act_clean_1 = get_clean_out(save_output_1)
                act_clean_2 = get_clean_out(save_output_2)
                act_clean_3 = get_clean_out(save_output_3)
                act_clean_4 = get_clean_out(save_output_4)
                act_clean_5 = get_clean_out(save_output_5)
                act_clean_6 = get_clean_out(save_output_6)
                act_clean_7 = get_clean_out(save_output_7)
                act_clean_8 = get_clean_out(save_output_8)
                act_clean_9 = get_clean_out(save_output_9)
                act_clean_10 = get_clean_out(save_output_10)
                act_clean_11 = get_clean_out(save_output_11)

                act_clean_1_var = get_clean_out_var(save_output_1)
                act_clean_2_var = get_clean_out_var(save_output_2)
                act_clean_3_var = get_clean_out_var(save_output_3)
                act_clean_4_var = get_clean_out_var(save_output_4)
                act_clean_5_var = get_clean_out_var(save_output_5)
                act_clean_6_var = get_clean_out_var(save_output_6)
                act_clean_7_var = get_clean_out_var(save_output_7)
                act_clean_8_var = get_clean_out_var(save_output_8)
                act_clean_9_var = get_clean_out_var(save_output_9)
                act_clean_10_var = get_clean_out_var(save_output_10)
                act_clean_11_var = get_clean_out_var(save_output_11)

                clean_3_list.append(act_clean_3), clean_5_list.append(act_clean_5), \
                clean_6_list.append(act_clean_6), \
                clean_9_list.append(act_clean_9), \
                clean_10_list.append(act_clean_10), clean_2_list.append(act_clean_2), clean_8_list.append(act_clean_8)
                clean_1_list.append(act_clean_1), clean_4_list.append(act_clean_4), clean_7_list.append(
                    act_clean_7), clean_11_list.append(act_clean_11)

                clean_3_list_var.append(act_clean_3_var), clean_5_list_var.append(act_clean_5_var), \
                clean_6_list_var.append(act_clean_6_var), \
                clean_9_list_var.append(act_clean_9_var), \
                clean_10_list_var.append(act_clean_10_var), clean_2_list_var.append(
                    act_clean_2_var), clean_8_list_var.append(act_clean_8_var)
                clean_1_list_var.append(act_clean_1_var), clean_4_list_var.append(
                    act_clean_4_var), clean_7_list_var.append(
                    act_clean_7_var), clean_11_list_var.append(act_clean_11_var)

                # del save_output
                save_output_1.clear(), save_output_2.clear(), save_output_3.clear(), save_output_4.clear(), save_output_5.clear(), \
                save_output_6.clear(), save_output_7.clear(), save_output_8.clear(), save_output_9.clear(), save_output_10.clear(), save_output_11.clear()
                # remove hooks
                hook_1.remove(), hook_2.remove(), hook_3.remove(), hook_4.remove(), hook_5.remove(), hook_6.remove(), \
                hook_7.remove(), hook_8.remove(), hook_9.remove(), hook_10.remove(), hook_11.remove()

        act_clean_1 = torch.mean(torch.stack(clean_1_list), dim=0)
        act_clean_2 = torch.mean(torch.stack(clean_2_list), dim=0)
        act_clean_3 = torch.mean(torch.stack(clean_3_list), dim=0)
        act_clean_4 = torch.mean(torch.stack(clean_4_list), dim=0)
        act_clean_5 = torch.mean(torch.stack(clean_5_list), dim=0)
        act_clean_6 = torch.mean(torch.stack(clean_6_list), dim=0)
        act_clean_7 = torch.mean(torch.stack(clean_7_list), dim=0)
        act_clean_8 = torch.mean(torch.stack(clean_8_list), dim=0)
        act_clean_9 = torch.mean(torch.stack(clean_9_list), dim=0)
        act_clean_10 = torch.mean(torch.stack(clean_10_list), dim=0)
        act_clean_11 = torch.mean(torch.stack(clean_11_list), dim=0)

        act_clean_1_var = torch.mean(torch.stack(clean_1_list_var), dim=0)
        act_clean_2_var = torch.mean(torch.stack(clean_2_list_var), dim=0)
        act_clean_3_var = torch.mean(torch.stack(clean_3_list_var), dim=0)
        act_clean_4_var = torch.mean(torch.stack(clean_4_list_var), dim=0)
        act_clean_5_var = torch.mean(torch.stack(clean_5_list_var), dim=0)
        act_clean_6_var = torch.mean(torch.stack(clean_6_list_var), dim=0)
        act_clean_7_var = torch.mean(torch.stack(clean_7_list_var), dim=0)
        act_clean_8_var = torch.mean(torch.stack(clean_8_list_var), dim=0)
        act_clean_9_var = torch.mean(torch.stack(clean_9_list_var), dim=0)
        act_clean_10_var = torch.mean(torch.stack(clean_10_list_var), dim=0)
        act_clean_11_var = torch.mean(torch.stack(clean_11_list_var), dim=0)

        for args.level in severity:
            all_res = list()
            for args.corruption in common_corruptions:
                para_to_opt = []
                err_corr = []
                net.load_state_dict(ckpt)
                parameters = list(para_to_opt)
                optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)

                te_loader_, te_loader = prepare_test_data(args)

                err_cls = 100 - (test(te_loader, net) * 100)
                print(f'Error before adaptation: {err_cls: .1f}')
                print('Epoch \t Error(%)')
                for epoch in range(1, args.nepoch + 1):
                    net.train()
                    for name, param in net.named_parameters():
                        for m in net.modules():
                            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                            nn.BatchNorm3d):
                                m.eval()
                    for idx, (inputs, labels) in enumerate(te_loader_):
                        optimizer.zero_grad()
                        save_output_1, save_output_2, save_output_3, save_output_4, save_output_5, save_output_6, \
                        save_output_7, save_output_8, save_output_9, \
                        save_output_10, save_output_11 = SaveOutput(), \
                                                         SaveOutput(), SaveOutput(), SaveOutput(), \
                                                         SaveOutput(), SaveOutput(), SaveOutput(), \
                                                         SaveOutput(), SaveOutput(), SaveOutput(), SaveOutput()

                        hook_1 = net.block1.layer[4].bn2.register_forward_hook(save_output_1)
                        hook_2 = net.block1.layer[3].bn2.register_forward_hook(save_output_2)
                        hook_3 = net.block1.layer[5].bn2.register_forward_hook(save_output_3)

                        hook_4 = net.block2.layer[4].bn2.register_forward_hook(save_output_4)
                        hook_5 = net.block2.layer[3].bn2.register_forward_hook(save_output_5)
                        hook_6 = net.block2.layer[5].bn2.register_forward_hook(save_output_6)

                        hook_7 = net.block3.layer[4].bn2.register_forward_hook(save_output_7)
                        hook_8 = net.block3.layer[3].bn2.register_forward_hook(save_output_8)
                        hook_9 = net.block3.layer[5].bn2.register_forward_hook(save_output_9)
                        hook_11 = net.block3.layer[2].bn2.register_forward_hook(save_output_11)

                        hook_10 = net.bn1.register_forward_hook(save_output_10)

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
                        out_9 = get_out(save_output_9)
                        out_10 = get_out(save_output_10)
                        out_11 = get_out(save_output_11)

                        out_1_var = get_out_var(save_output_1)
                        out_2_var = get_out_var(save_output_2)
                        out_3_var = get_out_var(save_output_3)
                        out_4_var = get_out_var(save_output_4)
                        out_5_var = get_out_var(save_output_5)
                        out_6_var = get_out_var(save_output_6)
                        out_7_var = get_out_var(save_output_7)
                        out_8_var = get_out_var(save_output_8)
                        out_9_var = get_out_var(save_output_9)
                        out_10_var = get_out_var(save_output_10)
                        out_11_var = get_out_var(save_output_11)

                        loss_1, loss_2, loss_3, loss_4, \
                        loss_5, loss_6, loss_7, loss_8, \
                        loss_9, loss_10, loss_11 = l1_loss(out_1, act_clean_1), l1_loss(out_2, act_clean_2), \
                                                   l1_loss(out_3, act_clean_3), l1_loss(out_4, act_clean_4), \
                                                   l1_loss(out_5, act_clean_5), l1_loss(out_6, act_clean_6), \
                                                   l1_loss(out_7, act_clean_7), l1_loss(out_8, act_clean_8), \
                                                   l1_loss(out_9, act_clean_9), l1_loss(out_10, act_clean_10), l1_loss(
                            out_11,
                            act_clean_11)

                        loss_var = l1_loss(out_1_var, act_clean_1_var) + l1_loss(out_2_var, act_clean_2_var) \
                                   + l1_loss(out_3_var, act_clean_3_var) + l1_loss(out_4_var, act_clean_4_var) \
                                   + l1_loss(out_5_var, act_clean_5_var) + l1_loss(out_6_var, act_clean_6_var) \
                                   + l1_loss(out_7_var, act_clean_7_var) + l1_loss(out_8_var, act_clean_8_var) + \
                                   l1_loss(out_9_var, act_clean_9_var) + l1_loss(out_10_var,
                                                                                 act_clean_10_var) + l1_loss(
                            out_11_var, act_clean_11_var)

                        loss = (loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 +
                                loss_7 + loss_8 + loss_9 + loss_10 + loss_11) / 2 + loss_var / 2

                        loss.backward()
                        optimizer.step()

                        save_output_1.clear(), save_output_2.clear(), save_output_3.clear(), save_output_4.clear(), \
                        save_output_5.clear(), save_output_6.clear(), save_output_7.clear(), \
                        save_output_8.clear(), save_output_9.clear(),
                        save_output_10.clear(), save_output_11.clear(),

                        hook_1.remove(), hook_2.remove(), hook_3.remove(), hook_4.remove(), hook_5.remove(), hook_6.remove(), \
                        hook_7.remove(), hook_8.remove(), hook_9.remove(), hook_10.remove(), hook_11.remove()

                    err_cls = 100 - (test(te_loader, net) * 100)
                    err_corr.append(err_cls)
                    scheduler.step()
                    print(f'{epoch} \t {err_cls: .1f}')
                    if args.save_model:
                        if err_cls <= min(err_corr):
                            state = {
                                'net': net.state_dict()
                            }
                            Path("aug_mix_c_10_ckpt/").mkdir(parents=True, exist_ok=True)
                            torch.save(state,
                                       'aug_mix_c_10_ckpt/' + str(args.corruption) + '.pth')
                min_err_corr = min(err_corr)
                print(f'Minimum Error after adaptation: {min_err_corr: .1f}')
                all_res.append(min_err_corr)
            print('Mean Error: ', sum(all_res) / len(all_res))
            Path(f'results/cifar10/{args.model}/').mkdir(exist_ok=True, parents=True)
            np.save(f'results/cifar10/{args.model}/{args.level}.npy', all_res)
