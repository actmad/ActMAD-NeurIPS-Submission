import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
########################################################################
parser.add_argument('--table', default='2', type=str,  choices=['2', '3a', '3b', '4'], help='The number of table (corresponding to main paper)')
parser.add_argument('--plot', default=False, action='store_true', help='Flag for plotting the ablation figures')
parser.add_argument('--all', default=False, action='store_true', help='Flag for analyzing all results')
parser.add_argument('--abl_only', default=False, action='store_true', help='Flag for analyzing only the ablation studies')
########################################################################
args = parser.parse_args()

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"],
    'font.size': 13
        })

levels = [5, 4, 3, 2, 1]
runs = np.arange(0, 10, 1)
d_size = [1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.01]
bs = ['BS-250', 'BS-225', 'BS-200', 'BS-175', 'BS-150', 'BS-125', 'BS-100', 'BS-75', 'BS-50', 'BS-25',
              'BS-10']


def print_dsize():
    print(f'######################## Figure: 4a ########################')

    print('Dataset Size: ')
    for d in d_size:
        print(f'{d*100}%'.ljust(8), end='\t')


def print_bs():
    print(f'######################## Figure: 4b ########################')

    print('Batch Sizes: ')
    for b in bs:
        print(f'{b}'.ljust(8), end='\t')


def print_corr():
    corruptions = ['gauss', 'shot', 'impul', 'defcs', 'gls', 'mtn', 'zm', 'snw', 'frst',
                   'fg', 'brt', 'cnt', 'els', 'px', 'jpg', 'Mean']
    for corr in corruptions:
        print(f'{corr}', end='\t')


def print_kitti_classes():
    classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person-Sit', 'Cyclist', 'Tram', 'Misc', 'Mean']
    for cls in classes:
        print(f'{cls}'.ljust(8), end='\t')


def print_kitti_source(path, weather):
    print('\n')
    print('Source Mean Average Precision: ')
    if weather == 'fog':
        src = np.load(f'{path}/fog.npy') * 100
        src = np.append(src, src.mean())

        for s in src:
            print(f'{s:.1f}'.ljust(8), end='\t')

    if weather == 'rain':
        src = np.load(f'{path}/rain.npy') * 100
        src = np.append(src, src.mean())

        for s in src:
            print(f'{s:.1f}'.ljust(8), end='\t')

    if weather == 'snow':
        src = np.load(f'{path}/snow.npy') * 100
        src = np.append(src, src.mean())

        for s in src:
            print(f'{s:.1f}'.ljust(8), end='\t')


def print_corr_results(mean_ind_corr, std_ind_corr, table):
    if table == 4:
        print('\n')
        print('ActMAD Mean Results (10 runs):')
        for a in mean_ind_corr:
            print(f'{a:.1f}'.ljust(8), end='\t')
        print('\n')
        print('Standard Deviation:')

        for b in std_ind_corr:
            print(f'{b:.2f}'.ljust(8), end='\t')
        print('\n')

    else:
        print('\n')
        print('ActMAD Mean Results (10 runs):')
        for a in mean_ind_corr:
            print(f'{a:.1f}', end='\t')
        print('\n')
        print('Standard Deviation:')

        for b in std_ind_corr:
            print(f'{b:.2f}', end='\t')
        print('\n')


def print_resnet_18_src(level):
    src = np.load(f'results/source/imagenet/res_18/level_{level}.npy')
    src = np.append(src, src.mean())
    print('Source Error (%): ')
    for s in src:
        s = 100 - s*100
        print(f'{s:.1f}', end='\t')


def print_source(path, level):
    src = np.load(f'{path}/level_{level}.npy')
    src = np.append(src, src.mean())
    print('Source Error (%): ')
    for s in src:
        # s = 100 - s*100
        print(f'{s:.1f}', end='\t')


def print_source_augmix(path, level):
    src = np.load(f'{path}/level_{level}.npy')
    src = np.append(src, src.mean())
    print('Source Error (%): ')
    for s in src:
        s = s*100
        print(f'{s:.1f}', end='\t')


def print_all_runs(table):
    print(f'############################ Printing All Results for Table - {table} ############################')
    if table == '2':
        print('\n############################ ImageNet ResNet-18 ############################')
        path_imagenet_res_18 = './results/main/imagenet/'
        for level in levels:
            print(f'\n######################## Level-{level} ########################')
            print_corr()
            print('\n')
            print_resnet_18_src(level)
            # print('\n')
            all_corr = list()
            for run in runs:
                load_np = np.load(f'{path_imagenet_res_18}/run_{run}/level_{level}.npy')
                all_corr.append(load_np)
            stk_v = np.vstack(all_corr)
            mean_ind_corr = stk_v.mean(axis=0)
            # mean all runs
            mean_all_runs = stk_v.mean(axis=1)
            mean_all_corr = np.mean(mean_ind_corr)
            mean_ind_corr = np.append(mean_ind_corr, mean_all_corr)
            std_ind_corr = np.std(stk_v, axis=0)
            # std for all runs
            mean_std = np.std(mean_all_runs)
            # std over corruptions
            mean_std_corr = np.mean(std_ind_corr)
            std_ind_corr = np.append(std_ind_corr, mean_std_corr)
            print_corr_results(mean_ind_corr, std_ind_corr, table)
    elif table == '3a':
        print('\n############################ CIFAR10 ResNet-26 ############################')
        path_c10_res_26 = './results/main/cifar10/res_26/'
        path_c10_res_26_src = './results/source/cifar_10/res_26'
        for level in levels:
            print(f'\n######################## Level-{level} ########################')
            print_corr()
            print('\n')
            print_source(path_c10_res_26_src, level)

            all_corr = list()
            for run in runs:
                load_np = np.load(f'{path_c10_res_26}/run_{run}/level_{level}.npy')
                all_corr.append(load_np)
            stk_v = np.vstack(all_corr)
            mean_ind_corr = stk_v.mean(axis=0)
            # mean all runs
            mean_all_runs = stk_v.mean(axis=1)
            mean_all_corr = np.mean(mean_ind_corr)
            mean_ind_corr = np.append(mean_ind_corr, mean_all_corr)
            std_ind_corr = np.std(stk_v, axis=0)
            # std for all runs
            mean_std = np.std(mean_all_runs)
            # std over corruptions
            mean_std_corr = np.mean(std_ind_corr)
            std_ind_corr = np.append(std_ind_corr, mean_std_corr)
            print_corr_results(mean_ind_corr, std_ind_corr, table=table)

        path_c100_res_26 = './results/main/cifar100/res_26/'
        path_c100_res_26_src = './results/source/cifar_100/res_26'

        print('\n############################ CIFAR100 ResNet-26 ############################')
        for level in levels:
            print(f'\n######################## Level-{level} ########################')
            print_corr()
            print('\n')
            print_source(path_c100_res_26_src, level)
            all_corr = list()
            for run in runs:
                load_np = np.load(f'{path_c100_res_26}/run_{run}/level_{level}.npy')
                all_corr.append(load_np)
            stk_v = np.vstack(all_corr)
            mean_ind_corr = stk_v.mean(axis=0)
            # mean all runs
            mean_all_runs = stk_v.mean(axis=1)
            mean_all_corr = np.mean(mean_ind_corr)
            mean_ind_corr = np.append(mean_ind_corr, mean_all_corr)
            std_ind_corr = np.std(stk_v, axis=0)
            # std for all runs
            mean_std = np.std(mean_all_runs)
            # std over corruptions
            mean_std_corr = np.mean(std_ind_corr)
            std_ind_corr = np.append(std_ind_corr, mean_std_corr)
            print_corr_results(mean_ind_corr, std_ind_corr, table=table)

    elif table == '3b':
        path_c10_augmix = './results/main/cifar10/augmix/'
        path_c10_augmix_src = './results/source/cifar_10/augmix'

        print('\n############################ CIFAR10 WRN-40-2 ############################')
        for level in levels:
            print(f'\n######################## Level-{level} ########################')
            print_corr()
            print('\n')
            print_source_augmix(path_c10_augmix_src, level)
            all_corr = list()
            for run in runs:
                load_np = np.load(f'{path_c10_augmix}/run_{run}/level_{level}.npy')
                all_corr.append(load_np)
            stk_v = np.vstack(all_corr)
            mean_ind_corr = stk_v.mean(axis=0)
            # mean all runs
            mean_all_runs = stk_v.mean(axis=1)
            mean_all_corr = np.mean(mean_ind_corr)
            mean_ind_corr = np.append(mean_ind_corr, mean_all_corr)
            std_ind_corr = np.std(stk_v, axis=0)
            # std for all runs
            mean_std = np.std(mean_all_runs)
            # std over corruptions
            mean_std_corr = np.mean(std_ind_corr)
            std_ind_corr = np.append(std_ind_corr, mean_std_corr)
            print_corr_results(mean_ind_corr, std_ind_corr, table=table)

        path_c100_augmix = './results/main/cifar100/augmix/'
        path_c100_augmix_src = './results/source/cifar_100/augmix'

        print('\n############################ CIFAR100 WRN-40-2 ############################')
        for level in levels:
            print(f'\n######################## Level-{level} ########################')
            print_corr()
            print('\n')
            print_source_augmix(path_c100_augmix_src, level)
            all_corr = list()
            for run in runs:
                load_np = np.load(f'{path_c100_augmix}/run_{run}/level_{level}.npy')
                all_corr.append(load_np)
            stk_v = np.vstack(all_corr)
            mean_ind_corr = stk_v.mean(axis=0)
            # mean all runs
            mean_all_runs = stk_v.mean(axis=1)
            mean_all_corr = np.mean(mean_ind_corr)
            mean_ind_corr = np.append(mean_ind_corr, mean_all_corr)
            std_ind_corr = np.std(stk_v, axis=0)
            # std for all runs
            mean_std = np.std(mean_all_runs)
            # std over corruptions
            mean_std_corr = np.mean(std_ind_corr)
            std_ind_corr = np.append(std_ind_corr, mean_std_corr)
            print_corr_results(mean_ind_corr, std_ind_corr, table=table)

    elif table == '4':
        print('\n############################ KITTI_FOG ############################')
        print_kitti_classes()
        path_kitti_fog = './results/main/kitti/kitti_fog/'
        path_kitti_src = './results/source/kitti/'
        print_kitti_source(path_kitti_src, weather='fog')
        all_corr = list()
        for run in runs:
            load_np = np.load(f'{path_kitti_fog}/run_{run}.npy')
            all_corr.append(load_np)
        stk_v = np.vstack(all_corr)
        mean_ind_corr = stk_v.mean(axis=0)
        mean_all_corr = np.mean(mean_ind_corr)
        mean_ind_corr = np.append(mean_ind_corr, mean_all_corr)
        std_ind_corr = np.std(stk_v, axis=0)
        mean_std_corr = np.mean(std_ind_corr)
        std_ind_corr = np.append(std_ind_corr, mean_std_corr)
        print_corr_results(mean_ind_corr, std_ind_corr, table=4)
        print('\n############################ KITTI_RAIN ############################')
        print_kitti_classes()
        path_kitti_fog = './results/main/kitti/kitti_rain/'
        print_kitti_source(path_kitti_src, weather='rain')
        all_corr = list()
        for run in runs:
            load_np = np.load(f'{path_kitti_fog}/run_{run}.npy')
            all_corr.append(load_np)
        stk_v = np.vstack(all_corr)
        mean_ind_corr = stk_v.mean(axis=0)
        mean_all_corr = np.mean(mean_ind_corr)
        mean_ind_corr = np.append(mean_ind_corr, mean_all_corr)
        std_ind_corr = np.std(stk_v, axis=0)
        mean_std_corr = np.mean(std_ind_corr)
        std_ind_corr = np.append(std_ind_corr, mean_std_corr)
        print_corr_results(mean_ind_corr, std_ind_corr, table=4)
        print('\n############################ KITTI_SNOW ############################')
        print_kitti_classes()
        path_kitti_fog = './results/main/kitti/kitti_snow/'
        print_kitti_source(path_kitti_src, weather='snow')
        all_corr = list()
        for run in runs:
            load_np = np.load(f'{path_kitti_fog}/run_{run}.npy')
            all_corr.append(load_np)
        stk_v = np.vstack(all_corr)
        mean_ind_corr = stk_v.mean(axis=0)
        mean_all_corr = np.mean(mean_ind_corr)
        mean_ind_corr = np.append(mean_ind_corr, mean_all_corr)
        std_ind_corr = np.std(stk_v, axis=0)
        mean_std_corr = np.mean(std_ind_corr)
        std_ind_corr = np.append(std_ind_corr, mean_std_corr)
        print_corr_results(mean_ind_corr, std_ind_corr, table=4)

    else:
        raise Exception('Table Not Found! Only Correct Choices are 2, 3a, 3b, 4')


def fig_4a(plot=False):
    path_test = './results/ablations/data_fractions/deep_aug/test_frac'
    path_train = './results/ablations/data_fractions/deep_aug/train_frac'
    path_train_test = './results/ablations/data_fractions/deep_aug/train_test_frac'

    dt_, dtr_, dtt_ = list(), list(), list()
    for d in d_size:
        dt = np.load(os.path.join(path_test, f'level_5_frac_{d}.npy'))
        dtr = np.load(os.path.join(path_train, f'level_5_frac_{d}.npy'))
        dtt = np.load(os.path.join(path_train_test, f'level_5_frac_{d}.npy'))

        dt_.append(dt.mean())
        dtr_.append(dtr.mean())
        dtt_.append(dtt.mean())

    test_deep_aug = dt_
    train_deep_aug = dtr_
    train_test_deep_aug = dtt_

    print_dsize()
    print('\nTest Fraction Results (Mean Error %):')
    for a in test_deep_aug:
        print(f'{a:.1f}%'.ljust(8), end='\t')

    print('\nTrain Fraction Results (Mean Error %):')
    for b in train_deep_aug:
        print(f'{b:.1f}%'.ljust(8), end='\t')

    print('\nTrain-Test Fraction Results (Mean Error %):')
    for c in train_test_deep_aug:
        print(f'{c:.1f}%'.ljust(8), end='\t')
    print('\n')

    all_arr = []
    all_arr.append(train_deep_aug)
    all_arr.append(test_deep_aug)
    all_arr.append(train_test_deep_aug)

    v_stk = np.vstack(all_arr)

    labels = ['DS\n100\%', 'DS\n50\%', 'DS\n40\%', 'DS\n30\%', 'DS\n20\%', 'DS\n10\%', 'DS\n5\%', 'DS\n2.5\%',
              'DS\n1\%']
    r1 = np.arange(len(labels))  # the label locations

    width = 0.23  # the width of the bars
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]

    fig, ax = plt.subplots()

    ax.yaxis.grid(zorder=0)

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    v_stk = np.transpose(v_stk)

    v_stk = np.around(v_stk, 2)

    v_stk = np.transpose(v_stk)
    rects1 = ax.bar(r1, v_stk[0], width, zorder=3, label='Reduced Train Data (for Clean Statistics)',
                    color='tab:purple')
    rects2 = ax.bar(r2, v_stk[1], width, zorder=3, label='Reduced Test Data (for Adaptation)', color='tab:orange')
    rects3 = ax.bar(r3, v_stk[2], width, zorder=3, label='Reduced Train \& Test Data', color='tab:blue')
    ax.set_ylabel('Top-1 Classification Error (\%)')
    plt.xticks([r + 1.0 * width for r in range(len(labels))], labels)
    plt.ylim(42, 46)
    plt.yticks([42, 44, 46])
    fig.set_size_inches(5.5, 3.0)
    fig.tight_layout()
    plt.legend(prop={'size': 13})
    if plot:
        plt.show()


def fig_4b(plot=False):
    err_ = list()
    path = './results/ablations/batch_size/deep_aug/'
    bs = [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    bs = bs[::-1]
    for b in bs:
        err = np.load(f'{path}/level_5_bs_{b}.npy')
        err_.append(err.mean())
    bs_res_50 = err_

    print_bs()

    bs_res_50 = np.round(bs_res_50, 1)
    print('\nActMAD (Mean Error %):')
    for a in bs_res_50:
        print(f'{a:.1f}%'.ljust(8), end='\t')
    print('\n')
    labels = ['BS\n250', 'BS\n225', 'BS\n200', 'BS\n175', 'BS\n150', 'BS\n125', 'BS\n100', 'BS\n75', 'BS\n50', 'BS\n25',
              'BS\n10']
    r1 = np.arange(len(labels))  # the label locations
    width = 0.75  # the width of the bars
    fig, ax = plt.subplots()
    ax.yaxis.grid(zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    rects1 = ax.bar(r1, bs_res_50, width, zorder=3, color='tab:blue')
    ax.set_ylabel('Top-1 Classification Error (\%)')
    plt.xticks([r for r in range(len(labels))], labels)
    plt.ylim(30, 50)
    plt.yticks([30, 40, 50])
    for i, v in enumerate(bs_res_50):
        plt.text(i - 0.40, v + 0.3, str(v))
    fig.set_size_inches(5.5, 3.0)
    fig.tight_layout()
    if plot:
        plt.show()


def fig_5(plot=False):
    print(f'######################## Figure: 5 ########################')

    plt.rcParams['legend.handlelength'] = 1
    plt.rcParams['legend.handleheight'] = 1

    all_arr = []

    act = np.load('./results/ablations/gn--bn/res_50_groupnorm/level_5.npy')
    act_bn = np.load('./results/ablations/gn--bn/res_50_batchnorm/level_5.npy')

    src = np.load('./results/ablations/gn--bn/source/level_5_gn.npy')
    src_bn = np.load('./results/ablations/gn--bn/source/level_5_bn.npy')

    all_arr.append(src_bn)
    all_arr.append(act_bn)
    all_arr.append(src)
    all_arr.append(act)
    v_stk = np.vstack(all_arr)
    labels_ = ['gauss', 'shot', 'impul', 'defcs', 'glas', 'motn', 'zoom', 'snow', 'frst', 'fog', 'brit', 'cont', 'elas',
               'pix', 'jpeg']
    print_corr()
    src = np.append(src, src.mean())

    src_bn = np.append(src_bn, src_bn.mean())

    act = np.append(act, act.mean())
    act_bn = np.append(act_bn, act_bn.mean())

    print('\nGroup Normalization Source: ')
    for a in src:
        print(f'{a:.1f}', end='\t')
    print('\n')
    print('Group Normalization ActMAD: ')
    for b in act:
        print(f'{b:.1f}', end='\t')
    print('\n')

    print('Batch Normalization Source: ')
    for c in src_bn:
        print(f'{c:.1f}', end='\t')
    print('\n')
    print('Batch Normalization ActMAD: ')
    for d in act_bn:
        print(f'{d:.1f}', end='\t')
    print('\n')

    labels = ['Source (BN)', 'ActMAD (BN)', 'Source (GN)', 'ActMAD (GN)']
    r1 = np.arange(len(labels_))  # the label locations
    width = 0.20  # the width of the bars
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]
    fig, ax = plt.subplots()
    ax.yaxis.grid(zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    v_stk = np.transpose(v_stk)
    v_stk = np.around(v_stk, 2)
    v_stk = np.transpose(v_stk)
    rects1 = ax.bar(r1, v_stk[0], width, zorder=3, label=labels[0], color='maroon')
    rects2 = ax.bar(r2, v_stk[1], width, zorder=3, label=labels[1], color='tab:orange')
    rects3 = ax.bar(r3, v_stk[2], width, zorder=3, label=labels[2], color='tab:green')
    rects4 = ax.bar(r4, v_stk[3], width, zorder=3, label=labels[3], color='tab:blue')
    ax.set_ylabel('Top-1 Classification Error (\%)')
    plt.xticks([r + 1.50 * width for r in range(len(labels_))], labels_)
    plt.ylim(30, 105)
    fig.set_size_inches(5.5, 3.0)
    plt.xticks(rotation=70)
    plt.legend(prop={'size': 8}, ncol=4)
    fig.tight_layout()
    if plot:
        plt.show()


if __name__ == "__main__":
    if args.all:
        print_all_runs(table='2')
        print_all_runs(table='3a')
        print_all_runs(table='3b')
        print_all_runs(table='4')
        print('\n############################ ABLATION STUDIES ############################\n')
        fig_4a(args.plot)
        fig_4b(args.plot)
        fig_5(args.plot)

    elif args.abl_only:
        fig_4a(args.plot)
        fig_4b(args.plot)
        fig_5(args.plot)

    else:
        print_all_runs(table=args.table)


