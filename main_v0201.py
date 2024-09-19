import os
import json
import torch
import random
import argparse
import numpy as np

import yaml
from modules.tokenizers_new import build_my_tokenizer
from modules.dataloaders import PretrainLoader, FinetuneLoaderHaveIndication, FinetuneLoaderNotIndication, \
    PretrainInferenceLoader
from modules.metrics.metrics import compute_all_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer_finetune_iu import PTrainer, FTrainer, PretrainTester, Tester
from modules.utils import PretrainTestAnalysis, setup_arguments, setup_seed
from models.model_pretrain_region_knowledge import Pretrain
from models.model_pretrain_region_knowledge_local import LocalPretrain
from models.model_pretrain_region_knowledge_global import GlobalPretrain
from models.model_pretrain_region_knowledge_inference import PretrainInference
from models.model_finetune_region_knowledge_v1121 import FineTune

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import wandb

# os.environ["WANDB_API_KEY"] = '*********'
os.environ["WANDB_MODE"] = "offline"
# wandb.login(key='************')


def main():
    # -------------------------------
    # load hyper-param
    # -------------------------------
    args, logger = setup_arguments()
    # -------------------------------
    # init wandb
    runner = wandb.init(
        project=f'rrg_{args["data_name"]}_{args["task"]}_{args["text_decoder"]}_{args["sk_topk"]}',
        config=args,
    )
    # -------------------------------
    # fix random seeds
    # -------------------------------
    setup_seed(args["seed"])
    # -------------------------------
    logger.info('start load data...')
    # -------------------------------
    # create tokenizer
    # -------------------------------
    print("load tokenizer...")
    tokenizer = build_my_tokenizer(tokenizer_dir=args['tokenizer_dir'], model=args['tokenizer_model'],
                                   data_name=args['data_name'], ann_path=args['ann_path'],
                                   tokenizer_type=args['tokenizer_type'], is_same_tokenizer=True)
    args['vocab_size'] = tokenizer.get_vocab_size()
    args['suppress_UNK'] = tokenizer.token_to_id('[UNK]')  # used for the CMN or r2gen text decoder
    # -------------------------------
    # save the config
    params = ''
    for key, value in args.items():
        params += f'{key}:\t{value}\n'
    logger.info(params)
    print(params)
    # -------------------------------
    # create data loader
    # -------------------------------
    if args['task'] == 'pretrain':
        train_dataloader = PretrainLoader(args, tokenizer, split='train', shuffle=False, drop_last=False)
        val_dataloader = PretrainLoader(args, tokenizer, split='val', shuffle=False, drop_last=False)
        test_dataloader = PretrainLoader(args, tokenizer, split='test', shuffle=False, drop_last=False)
    elif args['task'] == 'pretrain_inference':
        # mimic_train_loader = PretrainInferenceLoaderMIMICOne(args, split='train', shuffle=False, drop_last=False)
        train_dataloader = PretrainInferenceLoader(args, split='train', shuffle=False, drop_last=False)
        val_dataloader = PretrainInferenceLoader(args, split='val', shuffle=False, drop_last=False)
        test_dataloader = PretrainInferenceLoader(args, split='test', shuffle=False, drop_last=False)
    elif args['task'] == 'finetune':
        # has similar historical cases and indications
        train_loader_inc, val_loader_inc, test_loader_inc = None, None, None
        if args['is_add_indication']:
            train_loader_inc = FinetuneLoaderHaveIndication(args, tokenizer, split='train', shuffle=False, drop_last=False)
            val_loader_inc = FinetuneLoaderHaveIndication(args, tokenizer, split='val', shuffle=False, drop_last=False)
            test_loader_inc = FinetuneLoaderHaveIndication(args, tokenizer, split='test', shuffle=False, drop_last=False)

        # has similar historical cases and not has indication
        train_loader_not_inc = FinetuneLoaderNotIndication(args, tokenizer, split='train', shuffle=False,
                                                           drop_last=False)
        val_loader_not_inc = FinetuneLoaderNotIndication(args, tokenizer, split='val', shuffle=False,
                                                         drop_last=False)
        test_loader_not_inc = FinetuneLoaderNotIndication(args, tokenizer, split='test', shuffle=False,
                                                          drop_last=False)
    else:  # test
        train_loader_inc, train_loader_not_inc = None, None
        val_loader_inc, val_loader_not_inc = None, None
        test_loader_inc = None
        if args['is_add_indication']:
            test_loader_inc = FinetuneLoaderHaveIndication(args, tokenizer, split='test', shuffle=False, drop_last=False)
        test_loader_not_inc = FinetuneLoaderNotIndication(args, tokenizer, split='test', shuffle=False,
                                                          drop_last=False)

    # -------------------------------
    # record statistic of dataloader
    # -------------------------------
    if args['task'] in ['pretrain', 'pretrain_inference']:
        print(f"train_data is {len(train_dataloader.dataset) if train_dataloader is not None else 'None'}, "
              f"val_data is {len(val_dataloader.dataset) if val_dataloader is not None else 'None'}, "
              f"test_data is {len(test_dataloader.dataset)}")
        logger.info(f"train_data is {len(train_dataloader.dataset) if train_dataloader is not None else 'None'}, "
                    f"val_data is {len(val_dataloader.dataset) if val_dataloader is not None else 'None'}, "
                    f"test_data is {len(test_dataloader.dataset)}")
        runner.config.update({
            'vocab_size': tokenizer.get_vocab_size(),
            'suppress_UNK': args['suppress_UNK'],
            'train_len': len(train_dataloader.dataset) if train_dataloader is not None else 'None',
            'val_len': len(val_dataloader.dataset) if val_dataloader is not None else "None",
            'test_len': len(test_dataloader.dataset)
        }, allow_val_change=True)
    else:
        num_train_inc = len(train_loader_inc.dataset) if train_loader_inc is not None else 'None'
        num_train_not_inc = len(train_loader_not_inc.dataset) if train_loader_not_inc is not None else 'None'
        num_val_inc = len(val_loader_inc.dataset) if val_loader_inc is not None else 'None'
        num_val_not_inc = len(val_loader_not_inc.dataset) if val_loader_not_inc is not None else 'None'
        num_test_inc = len(test_loader_inc.dataset) if test_loader_inc is not None else 'None'
        num_test_not_inc = len(test_loader_not_inc.dataset) if test_loader_not_inc is not None else 'None'
        print(f"the number of train_data (indication-not_indication): {num_train_inc}-{num_train_not_inc}, "
              f"valid_data (indication-not_indication): {num_val_inc}-{num_val_not_inc}, "
              f"test_data (indication-not_indication): {num_test_inc}-{num_test_not_inc}, ")
        logger.info(f"the number of train_data (indication-not_indication): {num_train_inc}-{num_train_not_inc}, "
                    f"valid_data (indication-not_indication): {num_val_inc}-{num_val_not_inc}, "
                    f"test_data (indication-not_indication): {num_test_inc}-{num_test_not_inc}, ")

        runner.config.update({
            'vocab_size': tokenizer.get_vocab_size(),
            'suppress_UNK': args['suppress_UNK'],
            'train_inc_len': num_train_inc,
            'train_not_inc_len': num_train_not_inc,
            'val_inc_len': num_val_inc,
            'val_not_inc_len': num_val_not_inc,
            'test_inc_len': num_test_inc,
            'test_not_inc_len': num_test_not_inc,
        }, allow_val_change=True)
    # -------------------------------
    # build model architecture
    # -------------------------------
    if args['task'] == 'pretrain':
        if args['align_loss'] == 'multi-level':
            model = Pretrain(args, tokenizer, args['data_name'])
        elif args['align_loss'] == 'local':
            model = LocalPretrain(args, tokenizer, args['data_name'])
        else:  # global
            model = GlobalPretrain(args, tokenizer, args['data_name'])
    elif args['task'] == 'pretrain_inference':
        model = PretrainInference(args, data_name=args['data_name'])
    else:  # finetune or test
        model = FineTune(args, tokenizer, args['data_name'])
    model = model.to(args['device'])
    # runner.watch(model, log='all')
    # -------------------------------
    print(f'finish instantiate model!, Trainable parameters:{str(model).split("Trainable parameters:")[1]}M')
    logger.info(f'finish instantiate model!, Trainable parameters:{str(model).split("Trainable parameters:")[1]}M')
    # get function handles of loss and metrics
    # -------------------------------
    metrics = compute_all_scores
    # -------------------------------
    # build optimizer, learning rate scheduler
    # -------------------------------
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # -------------------------------
    # build trainer and start to train
    logger.info(f'start {args["task"]}!')
    print(f'start {args["task"]}!')
    # -------------------------------
    if args['task'] in ['pretrain', 'pretrain_inference']:
        kwarg = {"model": model, "metric_ftns": metrics, "optimizer": optimizer, "args": args,
                 "lr_scheduler": lr_scheduler, "train_dataloader": train_dataloader, "val_dataloader": val_dataloader,
                 "test_dataloader": test_dataloader, "logger": logger, "task": args['task'], 'runner': runner,
                 'is_save_checkpoint': args['is_save_checkpoint']}
    else:  # finetune or test
        kwarg = {"model": model, "metric_ftns": metrics, "optimizer": optimizer, "args": args,
                 "lr_scheduler": lr_scheduler, "train_loader_inc": train_loader_inc,
                 "train_loader_not_inc": train_loader_not_inc, "val_loader_inc": val_loader_inc,
                 "val_loader_not_inc": val_loader_not_inc, "test_loader_inc": test_loader_inc,
                 "test_loader_not_inc": test_loader_not_inc, "logger": logger, "task": args['task'], 'runner': runner,
                 'is_save_checkpoint': args['is_save_checkpoint']}

    if args['task'] == 'pretrain':
        trainer = PTrainer(**kwarg)
        trainer.train()
    elif args['task'] == 'pretrain_inference':
        tester = PretrainTester(**kwarg)
        specific_knowledge_data = tester.predict_mimic_cxr()
        save_file_name = args['ann_path'].split('.json')[0] + f'{args["sk_file_name"]}{args["sk_topk"]}.json'
        tester.get_specific_knowledge_mimic_cxr(specific_knowledge_data, save_file_name=save_file_name)
    elif args["task"] == 'finetune':
        trainer = FTrainer(**kwarg)
        trainer.train()
    else:  # inference
        trainer = Tester(**kwarg)
        trainer.test()
    runner.finish()


if __name__ == '__main__':
    main()
