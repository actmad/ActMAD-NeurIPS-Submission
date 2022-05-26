import numpy as np
from utils.get_methods import *
from utils.inc_dataloader import *
from torchvision import models
from argparse import Namespace
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='PATH_TO_FOLDER')
parser.add_argument('--model', default='')
########################################################################
parser.add_argument('--batch_size', default=1000, type=int)
########################################################################
parser.add_argument('--save', default=False, type=bool)
parser.add_argument('--d_size', default=0.01, type=float)
parser.add_argument('--workers', default=8, type=int)
########################################################################
parser.add_argument('--outf', default='.')
args: Namespace = parser.parse_args()

models_ = ['res_50', 'deep_aug', 'res_18']


def save_np(arr, i):
    args.corruption = 'original'
    Path(f"stats/stats_{args.model}/mean_{args.d_size}/").mkdir(parents=True, exist_ok=True)
    np.save(f'stats/stats_{args.model}/mean_{args.d_size}/' + str(args.corruption) + '_' + str(i) + '.npy', arr)


def save_np_var(arr, i):
    args.corruption = 'original'
    Path(f"stats/stats_{args.model}/var_{args.d_size}/").mkdir(parents=True, exist_ok=True)
    np.save(f'stats/stats_{args.model}/var_{args.d_size}/' + str(args.corruption) + '_' + str(i) + '.npy', arr)


for args.model in models_:
    print(f'Saving statistics for {args.model}')

    if args.model == 'res_50':
        # for saving statistics for different train dataset sizes
        d_size_ = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
        for args.d_size in d_size_:
            net = models.resnet50(pretrained=True)
            net = net.cuda()
            save_output_3, save_output_7, save_output_8, save_output_9, \
            save_output_10, save_output_11, = SaveOutput(), SaveOutput(), SaveOutput(), \
                             SaveOutput(), SaveOutput(), SaveOutput()

            args.corruption = 'original'

            tr_loader = prepare_train_data_frac(args)
            clean_3_list, clean_7_list, clean_8_list, clean_9_list, clean_10_list, clean_11_list = list(),list(),list(),\
                                                                                                   list(),list(),list(),
            clean_3_list_var, clean_7_list_var, clean_8_list_var, \
            clean_9_list_var, clean_10_list_var, clean_11_list_var = list(), list(), list(), list(), list(), list()

            for idx, (inputs, labels) in enumerate(tr_loader):

                hook_3 = net.layer2[3].bn3.register_forward_hook(save_output_3)
                hook_7 = net.layer3[4].bn3.register_forward_hook(save_output_7)
                hook_8 = net.layer3[5].bn3.register_forward_hook(save_output_8)
                hook_9 = net.layer4[0].bn3.register_forward_hook(save_output_9)
                hook_10 = net.layer4[1].bn3.register_forward_hook(save_output_10)
                hook_11 = net.layer4[2].bn3.register_forward_hook(save_output_11)

                inputs = inputs.cuda()
                net = net.cuda()
                with torch.no_grad():
                    output = net(inputs)
                act_clean_3 = get_clean_out(save_output_3)
                act_clean_7 = get_clean_out(save_output_7)
                act_clean_8 = get_clean_out(save_output_8)
                act_clean_9 = get_clean_out(save_output_9)
                act_clean_10 = get_clean_out(save_output_10)
                act_clean_11 = get_clean_out(save_output_11)

                act_clean_3_var = get_clean_out_var(save_output_3)
                act_clean_7_var = get_clean_out_var(save_output_7)
                act_clean_8_var = get_clean_out_var(save_output_8)
                act_clean_9_var = get_clean_out_var(save_output_9)
                act_clean_10_var = get_clean_out_var(save_output_10)
                act_clean_11_var = get_clean_out_var(save_output_11)

                clean_3_list.append(act_clean_3.cpu().numpy())
                clean_7_list.append(act_clean_7.cpu().numpy()), clean_8_list.append(
                    act_clean_8.cpu().numpy()), clean_9_list.append(act_clean_9.cpu().numpy()), \
                clean_10_list.append(act_clean_10.cpu().numpy()), clean_11_list.append(act_clean_11.cpu().numpy())

                clean_3_list_var.append(act_clean_3_var.cpu().numpy())
                clean_7_list_var.append(act_clean_7_var.cpu().numpy()), clean_8_list_var.append(
                    act_clean_8_var.cpu().numpy()), \
                clean_9_list_var.append(act_clean_9_var.cpu().numpy()), clean_10_list_var.append(
                    act_clean_10_var.cpu().numpy()), \
                clean_11_list_var.append(act_clean_11_var.cpu().numpy())

                save_output_3.clear()
                save_output_7.clear(), save_output_8.clear(), \
                save_output_9.clear(), save_output_10.clear(), save_output_11.clear()

                hook_3.remove()
                hook_7.remove(), hook_8.remove(), \
                hook_9.remove(), hook_10.remove(), hook_11.remove()

            act_clean_3 = np.mean(np.stack(clean_3_list), axis=0)
            act_clean_7 = np.mean(np.stack(clean_7_list), axis=0)
            act_clean_8 = np.mean(np.stack(clean_8_list), axis=0)
            act_clean_9 = np.mean(np.stack(clean_9_list), axis=0)
            act_clean_10 = np.mean(np.stack(clean_10_list), axis=0)
            act_clean_11 = np.mean(np.stack(clean_11_list), axis=0)

            act_clean_3_var = np.mean(np.stack(clean_3_list_var), axis=0)
            act_clean_7_var = np.mean(np.stack(clean_7_list_var), axis=0)
            act_clean_8_var = np.mean(np.stack(clean_8_list_var), axis=0)
            act_clean_9_var = np.mean(np.stack(clean_9_list_var), axis=0)
            act_clean_10_var = np.mean(np.stack(clean_10_list_var), axis=0)
            act_clean_11_var = np.mean(np.stack(clean_11_list_var), axis=0)

            save_np(act_clean_3, 3)
            save_np(act_clean_7, 7), save_np(act_clean_8, 8)
            save_np(act_clean_9, 9), save_np(act_clean_10, 10)
            save_np(act_clean_11, 11)

            save_np_var(act_clean_3_var, 3)
            save_np_var(act_clean_7_var, 7), save_np_var(act_clean_8_var, 8)
            save_np_var(act_clean_9_var, 9), save_np_var(act_clean_10_var, 10)
            save_np_var(act_clean_11_var, 11)

    if args.model == 'deep_aug':

        state_dict = torch.load(
            'ckpt/deepaugment_and_augmix.pth.tar')

        state_dict = get_state_dict(state_dict)

        d_size_ = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
        for args.d_size in d_size_:
            net = models.resnet50(pretrained=False)
            net.load_state_dict(state_dict)
            net = net.cuda()
            save_output_3, save_output_7, save_output_8, save_output_9, \
            save_output_10, save_output_11, = SaveOutput(), SaveOutput(), SaveOutput(), \
                                              SaveOutput(), SaveOutput(), SaveOutput()

            args.corruption = 'original'

            tr_loader = prepare_train_data_frac(args)
            clean_3_list, clean_7_list, clean_8_list, clean_9_list, clean_10_list, clean_11_list = list(), list(), list(), \
                                                                                                   list(), list(), list(),
            clean_3_list_var, clean_7_list_var, clean_8_list_var, \
            clean_9_list_var, clean_10_list_var, clean_11_list_var = list(), list(), list(), list(), list(), list()

            for idx, (inputs, labels) in enumerate(tr_loader):
                hook_3 = net.layer2[3].bn3.register_forward_hook(save_output_3)
                hook_7 = net.layer3[4].bn3.register_forward_hook(save_output_7)
                hook_8 = net.layer3[5].bn3.register_forward_hook(save_output_8)
                hook_9 = net.layer4[0].bn3.register_forward_hook(save_output_9)
                hook_10 = net.layer4[1].bn3.register_forward_hook(save_output_10)
                hook_11 = net.layer4[2].bn3.register_forward_hook(save_output_11)

                inputs = inputs.cuda()
                net = net.cuda()
                with torch.no_grad():
                    output = net(inputs)
                act_clean_3 = get_clean_out(save_output_3)
                act_clean_7 = get_clean_out(save_output_7)
                act_clean_8 = get_clean_out(save_output_8)
                act_clean_9 = get_clean_out(save_output_9)
                act_clean_10 = get_clean_out(save_output_10)
                act_clean_11 = get_clean_out(save_output_11)

                act_clean_3_var = get_clean_out_var(save_output_3)
                act_clean_7_var = get_clean_out_var(save_output_7)
                act_clean_8_var = get_clean_out_var(save_output_8)
                act_clean_9_var = get_clean_out_var(save_output_9)
                act_clean_10_var = get_clean_out_var(save_output_10)
                act_clean_11_var = get_clean_out_var(save_output_11)

                clean_3_list.append(act_clean_3.cpu().numpy())
                clean_7_list.append(act_clean_7.cpu().numpy()), clean_8_list.append(
                    act_clean_8.cpu().numpy()), clean_9_list.append(act_clean_9.cpu().numpy()), \
                clean_10_list.append(act_clean_10.cpu().numpy()), clean_11_list.append(act_clean_11.cpu().numpy())

                clean_3_list_var.append(act_clean_3_var.cpu().numpy())
                clean_7_list_var.append(act_clean_7_var.cpu().numpy()), clean_8_list_var.append(
                    act_clean_8_var.cpu().numpy()), \
                clean_9_list_var.append(act_clean_9_var.cpu().numpy()), clean_10_list_var.append(
                    act_clean_10_var.cpu().numpy()), \
                clean_11_list_var.append(act_clean_11_var.cpu().numpy())

                save_output_3.clear()
                save_output_7.clear(), save_output_8.clear(), \
                save_output_9.clear(), save_output_10.clear(), save_output_11.clear()

                hook_3.remove()
                hook_7.remove(), hook_8.remove(), \
                hook_9.remove(), hook_10.remove(), hook_11.remove()

            act_clean_3 = np.mean(np.stack(clean_3_list), axis=0)
            act_clean_7 = np.mean(np.stack(clean_7_list), axis=0)
            act_clean_8 = np.mean(np.stack(clean_8_list), axis=0)
            act_clean_9 = np.mean(np.stack(clean_9_list), axis=0)
            act_clean_10 = np.mean(np.stack(clean_10_list), axis=0)
            act_clean_11 = np.mean(np.stack(clean_11_list), axis=0)

            act_clean_3_var = np.mean(np.stack(clean_3_list_var), axis=0)
            act_clean_7_var = np.mean(np.stack(clean_7_list_var), axis=0)
            act_clean_8_var = np.mean(np.stack(clean_8_list_var), axis=0)
            act_clean_9_var = np.mean(np.stack(clean_9_list_var), axis=0)
            act_clean_10_var = np.mean(np.stack(clean_10_list_var), axis=0)
            act_clean_11_var = np.mean(np.stack(clean_11_list_var), axis=0)

            save_np(act_clean_3, 3)
            save_np(act_clean_7, 7), save_np(act_clean_8, 8)
            save_np(act_clean_9, 9), save_np(act_clean_10, 10)
            save_np(act_clean_11, 11)

            save_np_var(act_clean_3_var, 3)
            save_np_var(act_clean_7_var, 7), save_np_var(act_clean_8_var, 8)
            save_np_var(act_clean_9_var, 9), save_np_var(act_clean_10_var, 10)
            save_np_var(act_clean_11_var, 11)

    if args.model == 'res_18':
        d_size_ = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
        for args.d_size in d_size_:
            print(f'Saving statistics for {args.d_size * 100}% train data...')
            net = models.resnet18(pretrained=True)
            all_res = []
            save_output_1, save_output_2, save_output_3, save_output_4, \
            save_output_5, save_output_6, save_output_7, save_output_8 = SaveOutput(), SaveOutput(), SaveOutput(), \
                                                                         SaveOutput(), SaveOutput(), SaveOutput(), \
                                                                         SaveOutput(), SaveOutput()

            args.corruption = 'original'

            tr_loader = prepare_train_data_frac(args)
            clean_1_list, clean_2_list, clean_3_list, clean_4_list, clean_5_list, clean_6_list, clean_7_list, clean_8_list = list(), list(), \
                                                                                                                             list(), list(), list(), list(), list(), list()

            clean_1_list_var, clean_2_list_var, clean_3_list_var, clean_4_list_var, clean_5_list_var, clean_6_list_var, clean_7_list_var, clean_8_list_var = list(), list(), \
                                                                                                                                                             list(), list(), list(), list(), list(), list()

            for idx, (inputs, labels) in enumerate(tr_loader):
                hook_1 = net.layer1[0].bn2.register_forward_hook(save_output_1)
                hook_2 = net.layer1[1].bn2.register_forward_hook(save_output_2)
                hook_3 = net.layer2[0].bn2.register_forward_hook(save_output_3)
                hook_4 = net.layer2[1].bn2.register_forward_hook(save_output_4)
                hook_5 = net.layer3[0].bn2.register_forward_hook(save_output_5)
                hook_6 = net.layer3[1].bn2.register_forward_hook(save_output_6)
                hook_7 = net.layer4[0].bn2.register_forward_hook(save_output_7)
                hook_8 = net.layer4[1].bn2.register_forward_hook(save_output_8)
                inputs = inputs.cuda()
                net = net.cuda()
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

                act_clean_1_var = get_clean_out_var(save_output_1)
                act_clean_2_var = get_clean_out_var(save_output_2)
                act_clean_3_var = get_clean_out_var(save_output_3)
                act_clean_4_var = get_clean_out_var(save_output_4)
                act_clean_5_var = get_clean_out_var(save_output_5)
                act_clean_6_var = get_clean_out_var(save_output_6)
                act_clean_7_var = get_clean_out_var(save_output_7)
                act_clean_8_var = get_clean_out_var(save_output_8)

                clean_1_list.append(act_clean_1), clean_2_list.append(act_clean_2), clean_3_list.append(act_clean_3), \
                clean_4_list.append(act_clean_4), clean_5_list.append(act_clean_5), clean_6_list.append(act_clean_6)
                clean_7_list.append(act_clean_7)
                clean_8_list.append(act_clean_8)

                clean_1_list_var.append(act_clean_1_var), clean_2_list_var.append(act_clean_2_var), clean_3_list_var.append(
                    act_clean_3_var), \
                clean_4_list_var.append(act_clean_4_var), clean_5_list_var.append(act_clean_5_var), clean_6_list_var.append(
                    act_clean_6_var)
                clean_7_list_var.append(act_clean_7_var)
                clean_8_list_var.append(act_clean_8_var)
                #
                save_output_1.clear(), save_output_2.clear(), save_output_3.clear(), save_output_4.clear(), \
                save_output_5.clear(), save_output_6.clear(), save_output_7.clear(), save_output_8.clear()

                hook_1.remove(), hook_2.remove(), hook_3.remove(), hook_4.remove(), \
                hook_5.remove(), hook_6.remove(), hook_7.remove(), hook_8.remove()

            act_clean_1 = torch.mean(torch.stack(clean_1_list), dim=0)
            act_clean_2 = torch.mean(torch.stack(clean_2_list), dim=0)
            act_clean_3 = torch.mean(torch.stack(clean_3_list), dim=0)
            act_clean_4 = torch.mean(torch.stack(clean_4_list), dim=0)
            act_clean_5 = torch.mean(torch.stack(clean_5_list), dim=0)
            act_clean_6 = torch.mean(torch.stack(clean_6_list), dim=0)
            act_clean_7 = torch.mean(torch.stack(clean_7_list), dim=0)
            act_clean_8 = torch.mean(torch.stack(clean_8_list), dim=0)

            act_clean_1_var = torch.mean(torch.stack(clean_1_list_var), dim=0)
            act_clean_2_var = torch.mean(torch.stack(clean_2_list_var), dim=0)
            act_clean_3_var = torch.mean(torch.stack(clean_3_list_var), dim=0)
            act_clean_4_var = torch.mean(torch.stack(clean_4_list_var), dim=0)
            act_clean_5_var = torch.mean(torch.stack(clean_5_list_var), dim=0)
            act_clean_6_var = torch.mean(torch.stack(clean_6_list_var), dim=0)
            act_clean_7_var = torch.mean(torch.stack(clean_7_list_var), dim=0)
            act_clean_8_var = torch.mean(torch.stack(clean_8_list_var), dim=0)
            #
            save_np(act_clean_1.cpu().numpy(), 1), save_np(act_clean_2.cpu().numpy(), 2), \
            save_np(act_clean_3.cpu().numpy(), 3), save_np(act_clean_4.cpu().numpy(), 4), \
            save_np(act_clean_5.cpu().numpy(), 5), save_np(act_clean_6.cpu().numpy(), 6)
            save_np(act_clean_7.cpu().numpy(), 7), save_np(act_clean_8.cpu().numpy(), 8)

            save_np_var(act_clean_1_var.cpu().numpy(), 1), save_np_var(act_clean_2_var.cpu().numpy(), 2), \
            save_np_var(act_clean_3_var.cpu().numpy(), 3), save_np_var(act_clean_4_var.cpu().numpy(), 4), \
            save_np_var(act_clean_5_var.cpu().numpy(), 5), save_np_var(act_clean_6_var.cpu().numpy(), 6)
            save_np_var(act_clean_7_var.cpu().numpy(), 7), save_np_var(act_clean_8_var.cpu().numpy(), 8)