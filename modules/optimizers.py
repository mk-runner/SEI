import math
import torch
from torch import optim


def build_optimizer(args, model):
    if args['optim'] == 'AdamW':
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'],
                               amsgrad=args['amsgrad'])
    elif args['optim'] == 'RAdam':
        optimizer = optim.RAdam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    return optimizer


def build_lr_scheduler(args, optimizer):
    # if args['epochs'] <= 4:
    #     T_max = args['epochs']
    # else:
    #     T_max = args['epochs'] // 2
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=args['lr'] / 100)
    if args['lr_scheduler'] == 'StepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')

    return lr_scheduler
