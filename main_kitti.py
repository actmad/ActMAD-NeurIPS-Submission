import argparse
import json
import os
from pathlib import Path
from threading import Thread
import torch.optim as optim
import numpy as np
import torch
import yaml
from tqdm import tqdm
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
import torch.nn as nn
import torchvision.transforms as transforms
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel

adapt_tr = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((224, 672)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),

])

adapt_tr_ = transforms.Compose([transforms.ToTensor(), ])


def test_adapt(data,
               weights=None,
               batch_size=32,
               imgsz=640,
               conf_thres=0.001,
               iou_thres=0.6,  # for NMS
               save_json=False,
               single_cls=False,
               augment=False,
               verbose=True,
               model=None,
               dataloader=None,
               save_dir=Path(''),  # for saving images
               save_txt=False,  # for auto-labelling
               save_hybrid=False,  # for hybrid auto-labelling
               save_conf=False,  # save auto-label confidences
               plots=True,
               wandb_logger=None,
               compute_loss=None,
               half_precision=False,
               is_coco=False,
               opt=None):
    training = model
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
        model = attempt_load(weights, map_location=device)  # load FP32 model

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        # print(model)
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    task = 'train'
    dataloader_train = create_dataloader(data[task], imgsz, opt.batch_size, gs, opt, pad=0.5, rect=True,
                                         prefix=colorstr(f'{task}: '))[0]

    task = 'val'
    dataloader_val = create_dataloader(data[task], imgsz, opt.batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    l1_loss = nn.L1Loss(reduction='mean')
    print('###########Evaluating before adaption##########')

    for k, v in model.named_parameters():
        v.requires_grad = True

    opt.verbose = True

    test_(opt.data,
          opt.weights,
          opt.batch_size,
          opt.img_size,
          opt.conf_thres,
          opt.iou_thres,
          opt.save_json,
          opt.single_cls,
          opt.augment,
          opt.verbose,
          model=model,
          dataloader=dataloader_val,
          save_txt=opt.save_txt | opt.save_hybrid,
          save_hybrid=opt.save_hybrid,
          save_conf=opt.save_conf,
          opt=opt
          )

    save_out_1 = SaveOutput()
    save_out_2 = SaveOutput()
    save_out_3 = SaveOutput()
    save_out_4 = SaveOutput()
    save_out_5 = SaveOutput()

    save_out_6 = SaveOutput()
    save_out_7 = SaveOutput()
    save_out_8 = SaveOutput()
    save_out_9 = SaveOutput()
    save_out_10 = SaveOutput()
    save_out_11 = SaveOutput()
    save_out_12 = SaveOutput()
    save_out_13 = SaveOutput()
    save_out_14 = SaveOutput()
    save_out_15 = SaveOutput()

    save_out_16 = SaveOutput()
    save_out_17 = SaveOutput()
    save_out_18 = SaveOutput()

    save_out_19 = SaveOutput()
    save_out_20 = SaveOutput()

    clean_1_list, clean_2_list, clean_3_list, \
    clean_4_list, clean_5_list, clean_6_list, clean_7_list, \
    clean_8_list, clean_9_list, clean_10_list, clean_11_list, \
    clean_12_list, clean_13_list, clean_14_list, clean_15_list, clean_16_list, \
    clean_17_list, clean_18_list, clean_19_list, clean_20_list = list(), list(), \
                                                                 list(), list(), list(), list(), list(), \
                                                                 list(), list(), list(), list(), list(), list(), list(), \
                                                                 list(), list(), list(), list(), list(), list()

    clean_1_list_var, clean_2_list_var, clean_3_list_var, \
    clean_4_list_var, clean_5_list_var, clean_6_list_var, clean_7_list_var, \
    clean_8_list_var, clean_9_list_var, clean_10_list_var, clean_11_list_var, \
    clean_12_list_var, clean_13_list_var, clean_14_list_var, clean_15_list_var, clean_16_list_var, \
    clean_17_list_var, clean_18_list_var, clean_19_list_var, clean_20_list_var = list(), list(), \
                                                                 list(), list(), list(), list(), list(), \
                                                                 list(), list(), list(), list(), list(), list(), list(), \
                                                                 list(), list(), list(), list(), list(), list()

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader_train)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        model.eval()
        img = img.cuda()

        hook_1 = model.model[27][1].cv2.bn.register_forward_hook(save_out_1)
        hook_2 = model.model[26].cv2.bn.register_forward_hook(save_out_2)
        hook_3 = model.model[27][0].cv2.bn.register_forward_hook(save_out_3)
        hook_4 = model.model[23].bn.register_forward_hook(save_out_4)
        hook_5 = model.model[21].bn.register_forward_hook(save_out_5)
        hook_6 = model.model[27][1].cv1.bn.register_forward_hook(save_out_6)
        hook_7 = model.model[27][0].cv1.bn.register_forward_hook(save_out_7)
        hook_8 = model.model[22].bn.register_forward_hook(save_out_8)
        hook_9 = model.model[26].cv1.bn.register_forward_hook(save_out_9)
        hook_10 = model.model[20].cv1.bn.register_forward_hook(save_out_10)
        hook_11 = model.model[16].bn.register_forward_hook(save_out_11)

        hook_12 = model.model[19].cv1.bn.register_forward_hook(save_out_12)

        hook_13 = model.model[13].bn.register_forward_hook(save_out_13)
        hook_14 = model.model[11].cv2.bn.register_forward_hook(save_out_14)
        hook_15 = model.model[10][0].cv2.bn.register_forward_hook(save_out_15)

        hook_16 = model.model[8][7].cv2.bn.register_forward_hook(save_out_16)
        hook_17 = model.model[8][4].cv2.bn.register_forward_hook(save_out_17)
        hook_18 = model.model[5].bn.register_forward_hook(save_out_18)

        hook_19 = model.model[10][3].cv2.bn.register_forward_hook(save_out_19)
        hook_20 = model.model[6][7].cv2.bn.register_forward_hook(save_out_20)
        with torch.no_grad():
            _, _ = model(img)  # inference and training outputs

        clean_1_list.append(get_clean_out(save_out_1))
        clean_2_list.append(get_clean_out(save_out_2))
        clean_3_list.append(get_clean_out(save_out_3))
        clean_4_list.append(get_clean_out(save_out_4))
        clean_5_list.append(get_clean_out(save_out_5))

        clean_6_list.append(get_clean_out(save_out_6))
        clean_7_list.append(get_clean_out(save_out_7))
        clean_8_list.append(get_clean_out(save_out_8))
        clean_9_list.append(get_clean_out(save_out_9))
        clean_10_list.append(get_clean_out(save_out_10))
        clean_11_list.append(get_clean_out(save_out_11))

        clean_12_list.append(get_clean_out(save_out_12))
        clean_13_list.append(get_clean_out(save_out_13))
        clean_14_list.append(get_clean_out(save_out_14))
        clean_15_list.append(get_clean_out(save_out_15))

        clean_16_list.append(get_clean_out(save_out_16))
        clean_17_list.append(get_clean_out(save_out_17))
        clean_18_list.append(get_clean_out(save_out_18))

        clean_19_list.append(get_clean_out(save_out_19))
        clean_20_list.append(get_clean_out(save_out_20))

        clean_1_list_var.append(get_out_var(save_out_1))
        clean_2_list_var.append(get_out_var(save_out_2))
        clean_3_list_var.append(get_out_var(save_out_3))
        clean_4_list_var.append(get_out_var(save_out_4))
        clean_5_list_var.append(get_out_var(save_out_5))

        clean_6_list_var.append(get_out_var(save_out_6))
        clean_7_list_var.append(get_out_var(save_out_7))
        clean_8_list_var.append(get_out_var(save_out_8))
        clean_9_list_var.append(get_out_var(save_out_9))
        clean_10_list_var.append(get_out_var(save_out_10))
        clean_11_list_var.append(get_out_var(save_out_11))

        clean_12_list_var.append(get_out_var(save_out_12))
        clean_13_list_var.append(get_out_var(save_out_13))
        clean_14_list_var.append(get_out_var(save_out_14))
        clean_15_list_var.append(get_out_var(save_out_15))

        clean_16_list_var.append(get_out_var(save_out_16))
        clean_17_list_var.append(get_out_var(save_out_17))
        clean_18_list_var.append(get_out_var(save_out_18))

        clean_19_list_var.append(get_out_var(save_out_19))
        clean_20_list_var.append(get_out_var(save_out_20))

        save_out_1.clear()
        save_out_2.clear()
        save_out_3.clear()
        save_out_4.clear()
        save_out_5.clear()

        save_out_6.clear()
        save_out_7.clear()
        save_out_8.clear()
        save_out_9.clear()
        save_out_10.clear()
        save_out_11.clear()
        save_out_12.clear()
        save_out_13.clear()
        save_out_14.clear()
        save_out_15.clear()

        save_out_16.clear()
        save_out_17.clear()
        save_out_18.clear()

        save_out_19.clear()
        save_out_20.clear()

        hook_1.remove()
        hook_2.remove()
        hook_3.remove()
        hook_4.remove()
        hook_5.remove()

        hook_6.remove()
        hook_7.remove()
        hook_8.remove()
        hook_9.remove()

        hook_10.remove()
        hook_11.remove()
        hook_12.remove()
        hook_13.remove()
        hook_14.remove()
        hook_15.remove()

        hook_16.remove()
        hook_17.remove()
        hook_18.remove()

        hook_19.remove()
        hook_20.remove()

    # print(clean_1_list_var)
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
    act_clean_12 = take_mean(torch.stack(clean_12_list))
    act_clean_13 = take_mean(torch.stack(clean_13_list))
    act_clean_14 = take_mean(torch.stack(clean_14_list))
    act_clean_15 = take_mean(torch.stack(clean_15_list))

    act_clean_16 = take_mean(torch.stack(clean_16_list))
    act_clean_17 = take_mean(torch.stack(clean_17_list))
    act_clean_18 = take_mean(torch.stack(clean_18_list))

    act_clean_19 = take_mean(torch.stack(clean_19_list))
    act_clean_20 = take_mean(torch.stack(clean_20_list))

    act_clean_1_var = take_mean(torch.stack(clean_1_list))
    act_clean_2_var = take_mean(torch.stack(clean_2_list))
    act_clean_3_var = take_mean(torch.stack(clean_3_list))
    act_clean_4_var = take_mean(torch.stack(clean_4_list))
    act_clean_5_var = take_mean(torch.stack(clean_5_list))

    act_clean_6_var = take_mean(torch.stack(clean_6_list))
    act_clean_7_var = take_mean(torch.stack(clean_7_list))
    act_clean_8_var = take_mean(torch.stack(clean_8_list))
    act_clean_9_var = take_mean(torch.stack(clean_9_list))
    act_clean_10_var = take_mean(torch.stack(clean_10_list))
    act_clean_11_var = take_mean(torch.stack(clean_11_list))
    act_clean_12_var = take_mean(torch.stack(clean_12_list))
    act_clean_13_var = take_mean(torch.stack(clean_13_list))
    act_clean_14_var = take_mean(torch.stack(clean_14_list))
    act_clean_15_var = take_mean(torch.stack(clean_15_list))

    act_clean_16_var = take_mean(torch.stack(clean_16_list))
    act_clean_17_var = take_mean(torch.stack(clean_17_list))
    act_clean_18_var = take_mean(torch.stack(clean_18_list))

    act_clean_19_var = take_mean(torch.stack(clean_19_list))
    act_clean_20_var = take_mean(torch.stack(clean_20_list))

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [opt.milestone_1, opt.milestone_2], gamma=0.5, last_epoch=-1)

    print('Starting TEST TIME ADAPTATION WITH ActMAD...')
    for epoch in range(1, opt.nepoch + 1):
        model.train()
        for name, param in model.named_parameters():
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                nn.BatchNorm3d):
                    m.eval()

        for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader_val)):
            optimizer.zero_grad()

            save_out_1 = SaveOutput()
            save_out_2 = SaveOutput()
            save_out_3 = SaveOutput()
            save_out_4 = SaveOutput()
            save_out_5 = SaveOutput()

            save_out_6 = SaveOutput()
            save_out_7 = SaveOutput()
            save_out_8 = SaveOutput()
            save_out_9 = SaveOutput()
            save_out_10 = SaveOutput()
            save_out_11 = SaveOutput()
            save_out_12 = SaveOutput()
            save_out_13 = SaveOutput()
            save_out_14 = SaveOutput()
            save_out_15 = SaveOutput()

            save_out_16 = SaveOutput()
            save_out_17 = SaveOutput()
            save_out_18 = SaveOutput()

            save_out_19 = SaveOutput()
            save_out_20 = SaveOutput()

            hook_1 = model.model[27][1].cv2.bn.register_forward_hook(save_out_1)
            hook_2 = model.model[26].cv2.bn.register_forward_hook(save_out_2)
            hook_3 = model.model[27][0].cv2.bn.register_forward_hook(save_out_3)
            hook_4 = model.model[23].bn.register_forward_hook(save_out_4)
            hook_5 = model.model[21].bn.register_forward_hook(save_out_5)
            hook_6 = model.model[27][1].cv1.bn.register_forward_hook(save_out_6)
            hook_7 = model.model[27][0].cv1.bn.register_forward_hook(save_out_7)
            hook_8 = model.model[22].bn.register_forward_hook(save_out_8)
            hook_9 = model.model[26].cv1.bn.register_forward_hook(save_out_9)
            hook_10 = model.model[20].cv1.bn.register_forward_hook(save_out_10)
            hook_11 = model.model[16].bn.register_forward_hook(save_out_11)

            hook_12 = model.model[19].cv1.bn.register_forward_hook(save_out_12)

            hook_13 = model.model[13].bn.register_forward_hook(save_out_13)
            hook_14 = model.model[11].cv2.bn.register_forward_hook(save_out_14)
            hook_15 = model.model[10][0].cv2.bn.register_forward_hook(save_out_15)

            hook_16 = model.model[8][7].cv2.bn.register_forward_hook(save_out_16)
            hook_17 = model.model[8][4].cv2.bn.register_forward_hook(save_out_17)
            hook_18 = model.model[5].bn.register_forward_hook(save_out_18)

            hook_19 = model.model[10][3].cv2.bn.register_forward_hook(save_out_19)
            hook_20 = model.model[6][7].cv2.bn.register_forward_hook(save_out_20)

            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            img = img.cuda()
            _ = model(img)
            out_1 = get_out(save_out_1)
            out_2 = get_out(save_out_2)
            out_3 = get_out(save_out_3)
            out_4 = get_out(save_out_4)
            out_5 = get_out(save_out_5)
            out_6 = get_out(save_out_6)
            out_7 = get_out(save_out_7)
            out_8 = get_out(save_out_8)
            out_9 = get_out(save_out_9)
            out_10 = get_out(save_out_10)
            out_11 = get_out(save_out_11)
            out_12 = get_out(save_out_12)
            out_13 = get_out(save_out_13)
            out_14 = get_out(save_out_14)
            out_15 = get_out(save_out_15)

            out_16 = get_out(save_out_16)
            out_17 = get_out(save_out_17)
            out_18 = get_out(save_out_18)

            out_19 = get_out(save_out_19)
            out_20 = get_out(save_out_20)

            out_1_var = get_out_var(save_out_1)
            out_2_var = get_out_var(save_out_2)
            out_3_var = get_out_var(save_out_3)
            out_4_var = get_out_var(save_out_4)
            out_5_var = get_out_var(save_out_5)
            out_6_var = get_out_var(save_out_6)
            out_7_var = get_out_var(save_out_7)
            out_8_var = get_out_var(save_out_8)
            out_9_var = get_out_var(save_out_9)
            out_10_var = get_out_var(save_out_10)
            out_11_var = get_out_var(save_out_11)
            out_12_var = get_out_var(save_out_12)
            out_13_var = get_out_var(save_out_13)
            out_14_var = get_out_var(save_out_14)
            out_15_var = get_out_var(save_out_15)

            out_16_var = get_out_var(save_out_16)
            out_17_var = get_out_var(save_out_17)
            out_18_var = get_out_var(save_out_18)

            out_19_var = get_out(save_out_19)
            out_20_var = get_out(save_out_20)

            loss_1, loss_2, loss_3, loss_4, \
            loss_5, loss_6, loss_7, loss_8, \
            loss_9, loss_10, loss_11, loss_12, \
            loss_13, loss_14, loss_15, loss_16, \
            loss_17, loss_18, loss_19, loss_20 = \
                l1_loss(out_1, act_clean_1), \
                l1_loss(out_2, act_clean_2), \
                l1_loss(out_3, act_clean_3), \
                l1_loss(out_4, act_clean_4), l1_loss(out_5, act_clean_5), \
                l1_loss(out_6, act_clean_6), l1_loss(out_7, act_clean_7), \
                l1_loss(out_8, act_clean_8), l1_loss(out_9, act_clean_9), \
                l1_loss(out_10, act_clean_10), l1_loss(out_11, act_clean_11), \
                l1_loss(out_12, act_clean_12), l1_loss(out_13, act_clean_13), \
                l1_loss(out_14, act_clean_14), l1_loss(out_15, act_clean_15), \
                l1_loss(out_16, act_clean_16), l1_loss(out_17, act_clean_17), \
                l1_loss(out_18, act_clean_18), l1_loss(out_19, act_clean_19), l1_loss(out_20, act_clean_20),

            loss_1_var, loss_2_var, loss_3_var, loss_4_var, \
            loss_5_var, loss_6_var, loss_7_var, loss_8_var, \
            loss_9_var, loss_10_var, loss_11_var, loss_12_var, \
            loss_13_var, loss_14_var, loss_15_var, loss_16_var, \
            loss_17_var, loss_18_var, loss_19_var, loss_20_var = \
                l1_loss(out_1_var, act_clean_1_var), \
                l1_loss(out_2_var, act_clean_2_var), \
                l1_loss(out_3_var, act_clean_3_var), \
                l1_loss(out_4_var, act_clean_4_var), l1_loss(out_5_var, act_clean_5_var), \
                l1_loss(out_6_var, act_clean_6_var), l1_loss(out_7_var, act_clean_7_var), \
                l1_loss(out_8_var, act_clean_8_var), l1_loss(out_9_var, act_clean_9_var), \
                l1_loss(out_10_var, act_clean_10_var), l1_loss(out_11_var, act_clean_11_var), \
                l1_loss(out_12_var, act_clean_12_var), l1_loss(out_13_var, act_clean_13_var), \
                l1_loss(out_14_var, act_clean_14_var), l1_loss(out_15_var, act_clean_15_var), \
                l1_loss(out_16_var, act_clean_16_var), l1_loss(out_17_var, act_clean_17_var), \
                l1_loss(out_18_var, act_clean_18_var), l1_loss(out_19_var, act_clean_19_var), l1_loss(out_20_var, act_clean_20_var),

            loss = (loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7 + loss_8
                    + loss_9 + loss_10 + loss_11 + loss_12 + loss_13 + loss_14
                    + loss_15 + loss_16 + loss_17 + loss_18 + loss_19 + loss_20)

            loss_var = (loss_1_var + loss_2_var + loss_3_var + loss_4_var + loss_5_var + loss_6_var + loss_7_var + loss_8_var
                    + loss_9_var + loss_10_var + loss_11_var + loss_12_var + loss_13_var + loss_14_var
                    + loss_15_var + loss_16_var + loss_17_var + loss_18_var + loss_19_var + loss_20_var)
            loss = loss_var + loss

            loss.backward()
            optimizer.step()

            save_out_1.clear()
            save_out_2.clear()
            save_out_3.clear()
            save_out_4.clear()
            save_out_5.clear()

            save_out_6.clear()
            save_out_7.clear()
            save_out_8.clear()
            save_out_9.clear()
            save_out_10.clear()
            save_out_11.clear()
            save_out_12.clear()
            save_out_13.clear()
            save_out_14.clear()
            save_out_15.clear()

            save_out_16.clear()
            save_out_17.clear()
            save_out_18.clear()

            save_out_19.clear()
            save_out_20.clear()

            hook_1.remove()
            hook_2.remove()
            hook_3.remove()
            hook_4.remove()
            hook_5.remove()

            hook_6.remove()
            hook_7.remove()
            hook_8.remove()
            hook_9.remove()

            hook_10.remove()
            hook_11.remove()
            hook_12.remove()
            hook_13.remove()
            hook_14.remove()
            hook_15.remove()

            hook_16.remove()
            hook_17.remove()
            hook_18.remove()

            hook_19.remove()
            hook_20.remove()

        # scheduler.step()

        print(f'Starting Testing after Epoch {epoch}..')
        map = test_(opt.data,
                    opt.weights,
                    opt.batch_size,
                    opt.img_size,
                    opt.conf_thres,
                    opt.iou_thres,
                    opt.save_json,
                    opt.single_cls,
                    opt.augment,
                    opt.verbose,
                    dataloader=dataloader_val,
                    model=model,
                    save_txt=opt.save_txt | opt.save_hybrid,
                    save_hybrid=opt.save_hybrid,
                    save_conf=opt.save_conf,
                    opt=opt
                    )[0][2]


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def get_out(output_holder):
    out = torch.vstack(output_holder.outputs)
    out = torch.mean(out, dim=0)
    return out


def get_clean_out(output_holder):
    out = torch.vstack(output_holder.outputs)
    out = torch.mean(out, dim=0)
    return out


def get_out_var(output_holder):
    out = torch.vstack(output_holder.outputs)
    out = torch.var(out, dim=0)
    return out


def get_clean_out_var(out_holder):
    out = torch.vstack(out_holder.outputs)
    out = torch.var(out, dim=0)
    return out


def take_mean(input_ten):
    input_ten = torch.mean(input_ten, dim=0)

    return input_ten


def test(data,
         weights=None,
         batch_size=32,
         imgsz=1216,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=True,
         model=None,
         dataloader=None,
         save_dir=Path('images'),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=False,
         wandb_logger=None,
         compute_loss=None,
         half_precision=False,
         is_coco=False,
         opt=None):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        # print(model)
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        # print('HEREEEEEEEEEEEEEEEEEEEEEEEE', data[task])
        dataloader = create_dataloader(data[task], imgsz, opt.batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

        dataloader_1 = create_dataloader(data[task], imgsz, 1, gs, opt, pad=0.5, rect=True,
                                         prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):

        img_1 = img.to(device, non_blocking=True)
        img_1 = img.half() if half else img.float()  # uint8 to fp16/32
        img_1 /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # Run model
        t = time_synchronized()
        model.eval()
        img_1 = img_1.cuda()
        out, train_out = model(img_1, augment=augment)  # inference and training outputs
        t0 += time_synchronized() - t

        # Compute loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time_synchronized()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        # if plots and batch_i < 3:
        if plots:
            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            # print(image_id)
            f = save_dir / f'{str(batch_i).zfill(6)}.png'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            # f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            # Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


@torch.no_grad()
def test_(data,
          weights=None,
          batch_size=32,
          imgsz=1216,
          conf_thres=0.001,
          iou_thres=0.6,  # for NMS
          save_json=False,
          single_cls=False,
          augment=False,
          verbose=True,
          model=None,
          dataloader=None,
          save_dir=Path('images'),  # for saving images
          save_txt=False,  # for auto-labelling
          save_hybrid=False,  # for hybrid auto-labelling
          save_conf=False,  # save auto-label confidences
          plots=False,
          wandb_logger=None,
          compute_loss=None,
          half_precision=False,
          is_coco=False,
          opt=None):
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        # print(model)
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        # print('HEREEEEEEEEEEEEEEEEEEEEEEEE', data[task])
        dataloader = create_dataloader(data[task], imgsz, opt.batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

        dataloader_1 = create_dataloader(data[task], imgsz, 1, gs, opt, pad=0.5, rect=True,
                                         prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):

        img_1 = img.to(device, non_blocking=True)
        img_1 = img.half() if half else img.float()  # uint8 to fp16/32
        img_1 /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # Run model
        t = time_synchronized()
        model.eval()
        img_1 = img_1.cuda()
        out, train_out = model(img_1, augment=augment)  # inference and training outputs
        t0 += time_synchronized() - t

        # Compute loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time_synchronized()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        # if plots and batch_i < 3:
        if plots:
            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            # print(image_id)
            f = save_dir / f'{str(batch_i).zfill(6)}.png'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            # f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            # Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov3.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/kitti.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=30, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--adapt', type=bool, default=False, help='for running test_time adaptation')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--milestone_1', default=3, type=int)
    parser.add_argument('--milestone_2', default=6, type=int)
    parser.add_argument('--nepoch', default=1, type=int)
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.task in ('train', 'val', 'test') and not opt.adapt:
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             opt=opt
             )

    if opt.adapt:
        test_adapt(opt.data,
                   opt.weights,
                   opt.batch_size,
                   opt.img_size,
                   opt.conf_thres,
                   opt.iou_thres,
                   opt.save_json,
                   opt.single_cls,
                   opt.augment,
                   opt.verbose,
                   save_txt=opt.save_txt | opt.save_hybrid,
                   save_hybrid=opt.save_hybrid,
                   save_conf=opt.save_conf,
                   opt=opt
                   )


    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, opt=opt)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov3.pt yolov3-spp.pt yolov3-tiny.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, opt=opt)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
