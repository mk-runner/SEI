import copy
import json
import os
import math
import time

import cv2
import faiss
import numpy as np
import torch
import pandas as pd
from PIL import Image
from numpy import inf
from torchvision.transforms import transforms
from tqdm import tqdm
from abc import abstractmethod, ABC

from .loss import compute_lm_loss
from .metrics.metrics import compute_all_scores
from modules.utils import generate_heatmap


class BaseTrainer(object):
    def __init__(self, model, metric_ftns, optimizer, args, task, is_save_checkpoint, logger):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args['n_gpu'])
        self.model = model.to(self.device)
        self.task = task
        self.is_save_checkpoint = is_save_checkpoint
        self.logger = logger
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args['epochs']
        self.save_period = self.args['save_period']

        self.mnt_mode = args['monitor_mode']
        self.mnt_metric = 'val_' + args['monitor_metric']
        self.mnt_metric_test = 'test_' + args['monitor_metric']
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = os.path.join(args['result_dir'], 'checkpoint')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if args['resume'] is not None and args['resume'] != '':
            self._resume_checkpoint(args['resume'])

        if args['load'] is not None and args['load'] != '':
            self._load_checkpoint(args['load'])

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.model.train()
            result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            log.update(result)
            # the best results
            if self.mnt_metric in log.keys():
                pass
            else:
                if self.task == 'finetune':
                    if self.args['monitor_metric'] == 'RC':  # radgraph-partial + chexbert-all
                        log[self.mnt_metric] = log['val_F1-Radgraph-partial'] + log['val_chexbert_all_micro_f1']
                        log[self.mnt_metric_test] = log['test_F1-Radgraph-partial'] + log['test_chexbert_all_micro_f1']
                    elif self.args['monitor_metric'] == 'RB':  # radgraph-partial + B4
                        log[self.mnt_metric] = log['val_F1-Radgraph-partial'] + log['val_BLEU_4']
                        log[self.mnt_metric_test] = log['test_F1-Radgraph-partial'] + log['test_BLEU_4']
                    elif self.args['monitor_metric'] == 'RCB':  # radgraph-partial + chexbert-all + B4
                        log[self.mnt_metric] = log['val_F1-Radgraph-partial'] + log['val_chexbert_all_micro_f1'] + log['val_BLEU_4']
                        log[self.mnt_metric_test] = log['test_F1-Radgraph-partial'] + log['test_chexbert_all_micro_f1'] + log['test_BLEU_4']
                    else:
                        raise ValueError(f"{self.args['monitor_metric']} not implemented!")
            self._record_best(log)

            # print logged information to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0 and self.is_save_checkpoint:
                self._save_checkpoint(epoch, save_best=best)
                # self._save_checkpoint(epoch, save_best=False)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args['seed']
        self.best_recorder['test']['seed'] = self.args['seed']
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        record_path = os.path.join(self.args['result_dir'], f'{self.args["data_name"]}_{self.task}_results_record.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_val = pd.DataFrame.from_dict(self.best_recorder['val'], orient='index').T
        record_tes = pd.DataFrame.from_dict(self.best_recorder['test'], orient='index').T
        record_table = pd.concat([record_table, record_val], axis=0, ignore_index=True)
        record_table = pd.concat([record_table, record_tes], axis=0, ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        # filename = os.path.join(self.checkpoint_dir, f'checkpoint_{epoch}.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print(f"Saving current best {epoch}: model_best.pth ...")
            self.logger.info(f"Saving current best {epoch}: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _load_checkpoint(self, load_path):

        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)['state_dict']

        our_model_state_dict = self.model.state_dict()
        for key in our_model_state_dict.keys():
            if key in checkpoint.keys():
                our_model_state_dict[key] = checkpoint[key]
            else:
                if 'text_encoder' in key:
                    tail_key = '.'.join(key.split('.')[1:])
                    pretrain_key = f'text_encoder.{tail_key}'
                    our_model_state_dict[key] = checkpoint[pretrain_key]
                elif 'text_global' in key:
                    tail_key = '.'.join(key.split('.')[1:])
                    pretrain_key = f'text_global.{tail_key}'
                    our_model_state_dict[key] = checkpoint[pretrain_key]
                elif 'text_local' in key:
                    tail_key = '.'.join(key.split('.')[1:])
                    pretrain_key = f'text_local.{tail_key}'
                    our_model_state_dict[key] = checkpoint[pretrain_key]
                # else:
                #     print("not pretrained model!", key)

        self.model.load_state_dict(our_model_state_dict, strict=False)

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)
        if self.mnt_metric_test in log:
            improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
                self.mnt_metric_test]) or \
                            (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                                self.mnt_metric_test])
            if improved_test:
                self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args["monitor_metric"]))
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args["monitor_metric"]))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args["monitor_metric"]))
        self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args["monitor_metric"]))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))
            self.logger.info('\t{:15s}: {}'.format(str(key), value))


class PTrainer(BaseTrainer):
    def __init__(self, model, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader, logger, task, runner, is_save_checkpoint):
        super(PTrainer, self).__init__(model, metric_ftns, optimizer, args, task, is_save_checkpoint, logger)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.runner = runner

    def _train_epoch(self, epoch):

        # train_loss, train_region_loss, train_instance_loss = 0.0, 0.0, 0.0
        train_ret = {
            'train_all_loss': 0.0,
            'train_region_loss': 0.0,
            'train_instance_loss': 0.0
        }
        # training dataset
        self.model.train()
        for batch_idx, (image_ids, images, radgraph_ids, radgraph_masks) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            radgraph_ids, radgraph_masks = radgraph_ids.to(self.device), radgraph_masks.to(self.device)
            self.optimizer.zero_grad()
            ret = self.model(images, radgraph_ids, radgraph_masks)
            ret['all_loss'].backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            # train_ret['train_all_loss'] += ret['']
            train_ret['train_all_loss'] += ret['all_loss'].cpu().detach().item()
            train_ret['train_region_loss'] += ret['itc_region'].cpu().detach().item()
            train_ret['train_instance_loss'] += ret['itc_instance'].cpu().detach().item()
            if batch_idx % 1000 == 0 or batch_idx + 1 == len(self.train_dataloader):
                print(f"Epoch {epoch}, step {batch_idx}/{len(self.train_dataloader)}, "
                      f"itc instance loss: {ret['itc_instance'].cpu().detach().item():0.4}, "
                      f"itc region loss: {ret['itc_region'].cpu().detach().item():0.4}, "
                      f"all_loss: {ret['all_loss'].cpu().detach().item():0.4}, "
                      f"lr: {self.optimizer.param_groups[0]['lr']}")
                self.logger.info(f"Epoch {epoch}, step {batch_idx}/{len(self.train_dataloader)}, "
                                 f"itc instance loss: {ret['itc_instance'].cpu().detach().item():0.4}, "
                                 f"itc region loss: {ret['itc_region'].cpu().detach().item():0.4}, "
                                 f"all_loss: {ret['all_loss'].cpu().detach().item():0.4}, "
                                 f"lr: {self.optimizer.param_groups[0]['lr']}")

            del ret, images, radgraph_ids, radgraph_masks
            torch.cuda.empty_cache()
        train_ret = {k: v / len(self.train_dataloader) for k, v in train_ret.items()}
        log = {'epoch': epoch, **train_ret}
        self.logger.info(
            f'Epoch {epoch}, all train_loss: {log["train_all_loss"]}, lr: {self.optimizer.param_groups[0]["lr"]}')
        print(f'Epoch {epoch}, all train_loss: {log["train_all_loss"]}, lr: {self.optimizer.param_groups[0]["lr"]}')

        # validation dataset
        self.model.eval()
        with torch.no_grad():
            val_ret = {
                'val_region_loss': 0.0,
                'val_instance_loss': 0.0,
                'val_all_loss': 0.0
            }
            for batch_idx, (images_id, images, radgraph_ids, radgraph_masks) in enumerate(self.val_dataloader):
                images = images.to(self.device)
                radgraph_ids, radgraph_masks = radgraph_ids.to(self.device), radgraph_masks.to(self.device)
                ret = self.model(images, radgraph_ids, radgraph_masks)
                val_ret['val_all_loss'] += ret['all_loss'].cpu().detach().item()
                val_ret['val_region_loss'] += ret['itc_region'].cpu().detach().item()
                val_ret['val_instance_loss'] += ret['itc_instance'].cpu().detach().item()
            val_ret = {k: v / len(self.val_dataloader) for k, v in val_ret.items()}
            log.update(val_ret)
            # log.update(**{'val_' + k: v for k, v in val_ret.items()})
            logg_info = ''
            for k, v in val_ret.items():
                logg_info += f"{k}: {v}; "
            self.logger.info(f"Epoch {epoch}, val metrics: {logg_info}")
        self.logger.info(
            f'Epoch {epoch}, all validation loss: {val_ret["val_all_loss"]}, lr: {self.optimizer.param_groups[0]["lr"]}')
        print(
            f'Epoch {epoch}, all validation loss: {val_ret["val_all_loss"]}, lr: {self.optimizer.param_groups[0]["lr"]}')

        if epoch % 5 == 0 or epoch == self.epochs:
            # test dataset
            self.model.eval()
            with torch.no_grad():
                test_ret = {
                    'test_region_loss': 0.0,
                    'test_instance_loss': 0.0,
                    'test_all_loss': 0.0
                }
                for batch_idx, (images_id, images, radgraph_ids, radgraph_masks) in enumerate(self.test_dataloader):
                    images = images.to(self.device)
                    radgraph_ids, radgraph_masks = radgraph_ids.to(self.device), radgraph_masks.to(self.device)
                    ret = self.model(images, radgraph_ids, radgraph_masks)
                    test_ret['test_all_loss'] += ret['all_loss'].cpu().detach().item()
                    test_ret['test_region_loss'] += ret['itc_region'].cpu().detach().item()
                    test_ret['test_instance_loss'] += ret['itc_instance'].cpu().detach().item()
                test_ret = {k: v / len(self.test_dataloader) for k, v in test_ret.items()}
                # log.update(**{'test_' + k: v for k, v in test_ret.items()})
                log.update(test_ret)
                logg_info = ''
                for k, v in test_ret.items():
                    logg_info += f"{k}: {v}; "
                self.logger.info(f"Epoch {epoch}, test metrics: {logg_info}")
            self.logger.info(
                f'Epoch {epoch}, all test loss: {test_ret["test_all_loss"]}, lr: {self.optimizer.param_groups[0]["lr"]}')
            print(
                f'Epoch {epoch}, all test loss: {test_ret["test_all_loss"]}, lr: {self.optimizer.param_groups[0]["lr"]}')
        if self.args['lr_scheduler'] == 'ReduceLROnPlateau':
            self.lr_scheduler.step(train_ret['train_all_loss'])
        else:
            self.lr_scheduler.step()
        print(log)
        self.runner.log(log)  # wandb
        self.logger.info("#############################################################")
        return log


class PretrainTester(object):
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, logger, args, mimic_train_loader):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.mimic_train_loader = mimic_train_loader
        self.model = model
        self.logger = logger
        self.args = args
        assert args['load'] is not None
        checkpoint = torch.load(args['load'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("_________________________________________")
        for i, (name, para) in enumerate(self.model.visual_extractor.named_parameters()):
            if para.ndim == 1 and i % 10 == 0:
                print(name, para[:3].cpu().detach())
        print("_________________________________________")

        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("_________________________________________")
        for i, (name, para) in enumerate(self.model.visual_extractor.named_parameters()):
            if para.ndim == 1 and i % 10 == 0:
                print(name, para[:3].cpu().detach())
        print("_________________________________________")

    def predict(self):
        # ================= build train index =================#
        if self.args['data_name'] == 'mimic_cxr':
            d, nlist = self.args['output_dim'] * 50, 100
        else:
            d, nlist = self.args['output_dim'] * 99, 40
        quantizer = faiss.IndexFlatIP(d)
        train_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        self.model.eval()
        train_ids = []
        self.logger.info('building the index using the train data.')
        ori_data_name = self.args['data_name']
        self.args['data_nanme'] = ''
        with torch.no_grad():
            all_ret = []
            for batch_idx, (images_id, images) in tqdm(enumerate(self.mimic_train_loader, start=1)):
                images = images.to(self.device)
                ret = self.model(images)
                ret = ret.reshape(images.shape[0], -1)
                all_ret.append(ret.cpu())
                if len(self.train_dataloader) / batch_idx in [1, 2]:   # half and all features feed faiss
                    ret = torch.cat(all_ret, dim=0)
                    train_index.train(ret)
                    train_index.add(ret)
                    all_ret = []
                train_ids.extend(images_id)
                if batch_idx % 100 == 0:
                    self.logger.info(f'building index progress bar {batch_idx}/{len(self.train_dataloader)}')

        # ================= obtain the topk image_id =================#
        # ann_data = json.load(open(self.ann_path))
        new_ann_data = {}
        del all_ret
        self.logger.info('obtain the topk specific knowledge!')

        self.model.eval()
        with torch.no_grad():
            print(f"Retrieve other image IDs that are most similar to the current sample in the train set "
                  f"using image feature")
            self.logger.info(f"Retrieve other image IDs that are most similar to the current sample in the train set "
                             f"using image feature")
            for batch_idx, (images_id, images) in tqdm(enumerate(self.train_dataloader)):
                images = images.to(self.device)
                ret = self.model(images)
                ret = ret.reshape(images.shape[0], -1).cpu().numpy()
                _, I = train_index.search(ret, (self.args['sk_topk'] + 10))
                # delete the corresponding report
                for item_idx, image_id in zip(I, images_id):
                    topk_image_ids = []
                    if self.args['data_name'] == 'mimic_cxr':
                        subject_id, study_id, dicom_id = image_id.split('_')
                        cur_image_id = f'{subject_id}_{study_id}'
                        for i in item_idx:
                            if len(topk_image_ids) == self.args['sk_topk']:
                                break
                            _subject_id, _study_id, _dicom_id = train_ids[i].split('_')
                            if f'{_subject_id}_{_study_id}' != cur_image_id:
                                topk_image_ids.append(train_ids[i])
                    else:   # iu_xray
                        cur_image_id = image_id
                        for i in item_idx:
                            if len(topk_image_ids) == self.args['sk_topk']:
                                break
                            if train_ids[i] != cur_image_id:
                                topk_image_ids.append(train_ids[i])
                    assert len(topk_image_ids) == self.args['sk_topk']
                    new_ann_data[image_id] = topk_image_ids

                if batch_idx % 100 == 0:
                    self.logger.info(f"train phase progress bar {batch_idx}/{len(self.train_dataloader)}")
            del self.train_dataloader

            print(f"Retrieve other image IDs that are most similar to the current sample in the val set "
                  f"using image feature")
            self.logger.info(f"Retrieve other image IDs that are most similar to the current sample in the val set "
                             f"using image feature")
            for batch_idx, (images_id, images) in tqdm(enumerate(self.val_dataloader)):
                images = images.to(self.device)
                ret = self.model(images)
                ret = ret.reshape(images.shape[0], -1).cpu().numpy()
                _, I = train_index.search(ret, self.args['sk_topk'])
                # delete the corresponding report
                for item_idx, image_id in zip(I, images_id):
                    topk_image_ids = [train_ids[i] for i in item_idx]
                    new_ann_data[image_id] = topk_image_ids
                if batch_idx % 100 == 0:
                    self.logger.info(f"val phase progress bar {batch_idx}/{len(self.val_dataloader)}")
            del self.val_dataloader

            print(f"Retrieve other image IDs that are most similar to the current sample in the test set "
                  f"using image feature")
            self.logger.info(f"Retrieve other image IDs that are most similar to the current sample in the test set "
                             f"using image feature")
            for batch_idx, (images_id, images) in tqdm(enumerate(self.test_dataloader)):
                images = images.to(self.device)
                ret = self.model(images)
                ret = ret.reshape(images.shape[0], -1).cpu().numpy()
                _, I = train_index.search(ret, self.args['sk_topk'])
                # delete the corresponding report
                for item_idx, image_id in zip(I, images_id):
                    topk_image_ids = [train_ids[i] for i in item_idx]
                    new_ann_data[image_id] = topk_image_ids
                if batch_idx % 100 == 0:
                    self.logger.info(f"val phase progress bar {batch_idx}/{len(self.test_dataloader)}")
            del self.test_dataloader

        del train_ids, train_index
        # new_ann_path = self.ann_path.split(".json")[0] + '_topk_images.json'
        # with open(self.topk_ann_path, 'w') as f:
        #     json.dump(new_ann_data, f, indent=2)
        return new_ann_data

    def predict_iu_xray(self):
        # ================= build train index =================#
        # if self.args['data_name'] == 'mimic_cxr':
        #     d, nlist = self.args['output_dim'] * 50, 100
        # else:
        #     d, nlist = self.args['output_dim'] * 99, 40
        d, nlist = self.args['output_dim'] * 50, 100
        quantizer = faiss.IndexFlatIP(d)
        train_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        self.model.eval()
        train_ids = []
        self.logger.info('building the index using the train data.')
        # ori_data_name = self.args['data_name']
        with torch.no_grad():
            all_ret = []
            for batch_idx, (images_id, images) in tqdm(enumerate(self.mimic_train_loader, start=1)):
                images = images.to(self.device)
                ret = self.model(images, data_name='mimic_cxr')
                ret = ret.reshape(images.shape[0], -1)
                all_ret.append(ret.cpu())
                if len(self.train_dataloader) / batch_idx in [1, 2]:   # half and all features feed faiss
                    ret = torch.cat(all_ret, dim=0)
                    train_index.train(ret)
                    train_index.add(ret)
                    all_ret = []
                train_ids.extend(images_id)
                if batch_idx % 100 == 0:
                    self.logger.info(f'building index progress bar {batch_idx}/{len(self.train_dataloader)}')
        # self.args['data_name'] = ori_data_name
        # ================= obtain the topk image_id =================#
        # ann_data = json.load(open(self.ann_path))
        del self.mimic_train_loader
        new_ann_data = {}
        del all_ret
        self.logger.info('obtain the topk specific knowledge!')

        self.model.eval()
        with torch.no_grad():
            print(f"Retrieve other image IDs that are most similar to the current sample in the train set "
                  f"using image feature")
            self.logger.info(f"Retrieve other image IDs that are most similar to the current sample in the train set "
                             f"using image feature")
            for batch_idx, (images_id, images) in tqdm(enumerate(self.train_dataloader)):
                images = images.to(self.device)
                ret = self.model(images, data_name='iu_xray')
                ret = ret.reshape(images.shape[0], -1).cpu().numpy()
                _, I = train_index.search(ret, (self.args['sk_topk'] + 10))
                # delete the corresponding report
                for item_idx, image_id in zip(I, images_id):
                    topk_image_ids = []
                    if self.args['data_name'] == 'mimic_cxr':
                        subject_id, study_id, dicom_id = image_id.split('_')
                        cur_image_id = f'{subject_id}_{study_id}'
                        for i in item_idx:
                            if len(topk_image_ids) == self.args['sk_topk']:
                                break
                            _subject_id, _study_id, _dicom_id = train_ids[i].split('_')
                            if f'{_subject_id}_{_study_id}' != cur_image_id:
                                topk_image_ids.append(train_ids[i])
                    else:   # iu_xray
                        cur_image_id = image_id
                        for i in item_idx:
                            if len(topk_image_ids) == self.args['sk_topk']:
                                break
                            if train_ids[i] != cur_image_id:
                                topk_image_ids.append(train_ids[i])
                    assert len(topk_image_ids) == self.args['sk_topk']
                    new_ann_data[image_id] = topk_image_ids

                if batch_idx % 100 == 0:
                    self.logger.info(f"train phase progress bar {batch_idx}/{len(self.train_dataloader)}")
            del self.train_dataloader

            print(f"Retrieve other image IDs that are most similar to the current sample in the val set "
                  f"using image feature")
            self.logger.info(f"Retrieve other image IDs that are most similar to the current sample in the val set "
                             f"using image feature")
            for batch_idx, (images_id, images) in tqdm(enumerate(self.val_dataloader)):
                images = images.to(self.device)
                ret = self.model(images, data_name='iu_xray')
                ret = ret.reshape(images.shape[0], -1).cpu().numpy()
                _, I = train_index.search(ret, self.args['sk_topk'])
                # delete the corresponding report
                for item_idx, image_id in zip(I, images_id):
                    topk_image_ids = [train_ids[i] for i in item_idx]
                    new_ann_data[image_id] = topk_image_ids
                if batch_idx % 100 == 0:
                    self.logger.info(f"val phase progress bar {batch_idx}/{len(self.val_dataloader)}")
            del self.val_dataloader

            print(f"Retrieve other image IDs that are most similar to the current sample in the test set "
                  f"using image feature")
            self.logger.info(f"Retrieve other image IDs that are most similar to the current sample in the test set "
                             f"using image feature")
            for batch_idx, (images_id, images) in tqdm(enumerate(self.test_dataloader)):
                images = images.to(self.device)
                ret = self.model(images, data_name='iu_xray')
                ret = ret.reshape(images.shape[0], -1).cpu().numpy()
                _, I = train_index.search(ret, self.args['sk_topk'])
                # delete the corresponding report
                for item_idx, image_id in zip(I, images_id):
                    topk_image_ids = [train_ids[i] for i in item_idx]
                    new_ann_data[image_id] = topk_image_ids
                if batch_idx % 100 == 0:
                    self.logger.info(f"val phase progress bar {batch_idx}/{len(self.test_dataloader)}")
            del self.test_dataloader

        del train_ids, train_index
        # new_ann_path = self.ann_path.split(".json")[0] + '_topk_images.json'
        # with open(self.topk_ann_path, 'w') as f:
        #     json.dump(new_ann_data, f, indent=2)
        return new_ann_data

    def get_specific_knowledge(self, id2image, save_file_name):
        # get id2report dict
        id2report = {}
        ann_data = json.load(open(self.args['ann_path']))
        for split, value in ann_data.items():
            for idx, item in tqdm(enumerate(value)):
                if self.args['data_name'] == 'mimic_cxr':
                    cur_idx = '_'.join([str(item['subject_id']), str(item['study_id']), item['id']])
                else:
                    cur_idx = item['id']
                id2report[cur_idx] = [item['report'], item['core_findings']]

        # get specific knowledge
        new_ann_data = {}
        for split, value in ann_data.items():
            new_ann_data[split] = []
            for idx, item in tqdm(enumerate(value)):
                if self.args['data_name'] == 'mimic_cxr':
                    cur_idx = '_'.join([str(item['subject_id']), str(item['study_id']), item['id']])
                else:
                    cur_idx = item['id']
                try:
                    topk_images_id = id2image[cur_idx][:self.args['sk_topk']]
                    sk_reports = [id2report[i][0] for i in topk_images_id]
                    sk_keywords = [id2report[i][1] for i in topk_images_id]

                    specific_knowledge = {'sk_ids': topk_images_id, 'reports': sk_reports, 'sk_keywords': sk_keywords}
                except:
                    specific_knowledge = {'sk_ids': [], 'reports': [], 'keywords': []}
                new_item = {
                    **item,
                    'specific_knowledge': specific_knowledge
                }
                new_ann_data[split].append(new_item)

        json.dump(new_ann_data, open(save_file_name, 'w'), indent=2)
        return id2report

    def get_specific_knowledge_iu_xray(self, id2image, save_file_name):
        # get id2report dict from mimic-cxr
        id2report = {}
        mimic_cxr_ann_data = json.load(open(self.args['mimic_cxr_ann_path']))
        for split, value in mimic_cxr_ann_data.items():
            if split != 'train':
                continue
            for idx, item in tqdm(enumerate(value)):
                cur_idx = '_'.join([str(item['subject_id']), str(item['study_id']), item['id']])
                id2report[cur_idx] = [item['report'], item['core_findings']]

        del mimic_cxr_ann_data
        iu_xray_ann_data = json.load(open(self.args['iu_xray_ann_path']))
        # get specific knowledge
        new_ann_data = {}
        for split, value in iu_xray_ann_data.items():
            new_ann_data[split] = []
            for idx, item in tqdm(enumerate(value)):
                if self.args['data_name'] == 'mimic_cxr':
                    cur_idx = '_'.join([str(item['subject_id']), str(item['study_id']), item['id']])
                else:
                    cur_idx = item['id']
                try:
                    topk_images_id = id2image[cur_idx][:self.args['sk_topk']]
                    sk_reports = [id2report[i][0] for i in topk_images_id]
                    sk_keywords = [id2report[i][1] for i in topk_images_id]

                    specific_knowledge = {'sk_ids': topk_images_id, 'reports': sk_reports, 'sk_keywords': sk_keywords}
                except:
                    specific_knowledge = {'sk_ids': [], 'reports': [], 'keywords': []}
                new_item = {
                    **item,
                    'specific_knowledge': specific_knowledge
                }
                new_ann_data[split].append(new_item)

        json.dump(new_ann_data, open(save_file_name, 'w'), indent=2)
        return id2report


class FTrainer(BaseTrainer):
    def __init__(self, model, metric_ftns, optimizer, args, lr_scheduler, train_loader_inc, train_loader_not_inc,
                 val_loader_inc, val_loader_not_inc, test_loader_inc, test_loader_not_inc, logger, task,
                 runner, is_save_checkpoint):
        super(FTrainer, self).__init__(model, metric_ftns, optimizer, args, task, is_save_checkpoint, logger)
        self.lr_scheduler = lr_scheduler
        self.train_loader_not_inc = train_loader_not_inc
        self.train_loader_inc = train_loader_inc
        self.val_loader_not_inc = val_loader_not_inc
        self.val_loader_inc = val_loader_inc
        self.test_loader_not_inc = test_loader_not_inc
        self.test_loader_inc = test_loader_inc
        self.runner = runner

    def _train_epoch(self, epoch):

        train_loss, batch_num = 0, 0
        self.model.train()
        # train dataset with indication section
        print(f"Epoch {epoch}, training dataset with indication section++++++++++++++++++++++++++++++++++++++")
        self.logger.info(f"Epoch {epoch}, training dataset with indication section++++++++++++++++++++++++++++++++++++++")
        if self.train_loader_inc is not None:
            for batch_idx, (image_ids, images, report_ids, report_masks, sk_ids, sk_masks, inc_ids, inc_masks) in enumerate(
                    self.train_loader_inc):
                images = images.to(self.device, non_blocking=True)
                report_ids, report_masks = report_ids.to(self.device), report_masks.to(self.device)
                self.optimizer.zero_grad()
                ret = self.model(images, report_ids, report_masks, sk_ids, sk_masks, inc_ids, inc_masks,
                                 mode='train', is_contain_indication=True)
                ret['all_loss'].backward()
                train_loss += ret['all_loss'].cpu().detach().item()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                if batch_idx % 2000 == 0 or batch_idx + 1 == len(self.train_loader_inc):
                    print(f"Epoch {epoch}, step {batch_idx}/{len(self.train_loader_inc)}, "
                          f"all_loss: {ret['all_loss'].cpu().detach().item():0.4}, "
                          f"lr: {self.optimizer.param_groups[0]['lr']}")
                    self.logger.info(f"Epoch {epoch}, step {batch_idx}/{len(self.train_loader_inc)}, "
                                     f"all_loss: {ret['all_loss'].cpu().detach().item():0.4}, "
                                     f"lr: {self.optimizer.param_groups[0]['lr']}")
            batch_num += len(self.train_loader_inc)
        print(f"Epoch {epoch}, training dataset only with similar historical cases++++++++++++++++++++++++++++++++++")
        self.logger.info(
            f"Epoch {epoch}, training dataset only with similar historical cases+++++++++++++++++++++++++++++++++++")
        for batch_idx, (image_ids, images, report_ids, report_masks, sk_ids, sk_masks) in enumerate(
                self.train_loader_not_inc):
            images = images.to(self.device, non_blocking=True)
            report_ids, report_masks = report_ids.to(self.device), report_masks.to(self.device)
            self.optimizer.zero_grad()
            ret = self.model(images, report_ids, report_masks, sk_ids, sk_masks,
                             mode='train', is_contain_indication=False)
            ret['all_loss'].backward()
            train_loss += ret['all_loss'].cpu().detach().item()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            if batch_idx % 2000 == 0 or batch_idx + 1 == len(self.train_loader_not_inc):
                print(f"Epoch {epoch}, step {batch_idx}/{len(self.train_loader_not_inc)}, "
                      f"all_loss: {ret['all_loss'].cpu().detach().item():0.4}, "
                      f"lr: {self.optimizer.param_groups[0]['lr']}")
                self.logger.info(f"Epoch {epoch}, step {batch_idx}/{len(self.train_loader_not_inc)}, "
                                 f"all_loss: {ret['all_loss'].cpu().detach().item():0.4}, "
                                 f"lr: {self.optimizer.param_groups[0]['lr']}")

        batch_num += len(self.train_loader_not_inc)
        log = {'train_loss': train_loss / batch_num, 'epoch': epoch}
        self.logger.info(f'Epoch {epoch}, training is over, train_loss: {log["train_loss"]}, '
                         f'lr: {self.optimizer.param_groups[0]["lr"]}')
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res, val_images_ids = [], [], []
            if self.val_loader_inc is not None:
                print(f"Epoch {epoch}, validation dataset with indication section+++++++++++++++++++")
                self.logger.info(f"Epoch {epoch}, validation dataset with indication section+++++++++++++++++++")
                for _, (images_id, images, reports_ids, reports_masks, sk_ids, sk_masks, inc_ids, inc_masks) in enumerate(
                        self.val_loader_inc):
                    images = images.to(self.device)
                    reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)
                    gen_texts, gt_texts = self.model(images, reports_ids, reports_masks, sk_ids, sk_masks,
                                                     inc_ids, inc_masks, mode='sample', is_contain_indication=True)
                    val_res.extend(gen_texts)
                    val_gts.extend(gt_texts)
                    val_images_ids.extend(images_id)

            print(f"Epoch {epoch}, validation dataset only with similar historical cases+++++++++++++++++++")
            self.logger.info(f"Epoch {epoch}, validation dataset only with similar historical cases+++++++++++++++++++")
            for _, (images_id, images, reports_ids, reports_masks, sk_ids, sk_masks) in enumerate(
                    self.val_loader_not_inc):
                images = images.to(self.device)
                reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)
                gen_texts, gt_texts = self.model(images, reports_ids, reports_masks, sk_ids, sk_masks,
                                                 mode='sample', is_contain_indication=False)
                val_res.extend(gen_texts)
                val_gts.extend(gt_texts)
                val_images_ids.extend(images_id)

            val_met = self.metric_ftns(gts=val_gts, res=val_res, args=self.args)
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            logg_info = ''
            for k, v in val_met.items():
                logg_info += f"{k}: {v}; "
            self.logger.info(f"Epoch {epoch}, val metrics: {logg_info}")

            # save the metrics and the predict results
            temp_ids, temp_val_gts, temp_val_res = list(val_met.keys()), [None]*len(val_met), list(val_met.values())
            temp_ids.extend(val_images_ids)
            temp_val_gts.extend(val_gts)
            temp_val_res.extend(val_res)
            cur_val_ret = pd.DataFrame({'images_id': temp_ids, 'ground_truth': temp_val_gts,
                                        f'pred_report_{epoch}': temp_val_res})
            val_pred_path = os.path.join(self.args['result_dir'], 'val_prediction.csv')
            if os.path.exists(val_pred_path):
                val_pred_df = pd.read_csv(val_pred_path)
                val_pred_df = pd.merge(val_pred_df, cur_val_ret[['images_id', f'pred_report_{epoch}']], on='images_id')
                val_pred_df.to_csv(val_pred_path, index=False)
            else:
                cur_val_ret.to_csv(val_pred_path, index=False)
        self.logger.info(f'Epoch {epoch}, validation is over...')
        self.model.eval()
        with torch.no_grad():
            if self.test_loader_inc is not None:
                test_gts, test_res, test_images_ids = [], [], []
                print(f"Epoch {epoch}, test dataset with indication section+++++++++++++++++++")
                self.logger.info(f"Epoch {epoch}, test dataset with indication section+++++++++++++++++++")
                for _, (images_id, images, reports_ids, reports_masks, sk_ids, sk_masks, inc_ids, inc_masks) in enumerate(
                        self.test_loader_inc):
                    images = images.to(self.device)
                    reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)
                    gen_texts, gt_texts = self.model(images, reports_ids, reports_masks, sk_ids, sk_masks,
                                                     inc_ids, inc_masks, mode='sample', is_contain_indication=True)
                    test_res.extend(gen_texts)
                    test_gts.extend(gt_texts)
                    test_images_ids.extend(images_id)

            print(f"Epoch {epoch}, test dataset only with similar historical cases+++++++++++++++++++")
            self.logger.info(f"Epoch {epoch}, test dataset only with similar historical cases+++++++++++++++++++")
            for _, (images_id, images, reports_ids, reports_masks, sk_ids, sk_masks) in enumerate(
                    self.test_loader_not_inc):
                images = images.to(self.device)
                reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)
                gen_texts, gt_texts = self.model(images, reports_ids, reports_masks, sk_ids, sk_masks, mode='sample')
                test_res.extend(gen_texts)
                test_gts.extend(gt_texts)
                test_images_ids.extend(images_id)

            test_met = self.metric_ftns(gts=test_gts, res=test_res, args=self.args)
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            logg_info = ''
            for k, v in test_met.items():
                logg_info += f"{k}: {v}; "
            self.logger.info(f"Epoch {epoch}, test metrics: {logg_info}")

            # save the metrics and the predict results
            temp_ids, temp_test_gts, temp_test_res = list(test_met.keys()), [None] * len(test_met), list(test_met.values())
            temp_ids.extend(test_images_ids)
            temp_test_gts.extend(test_gts)
            temp_test_res.extend(test_res)
            cur_test_ret = pd.DataFrame({'images_id': temp_ids, 'ground_truth': temp_test_gts,
                                        f'pred_report_{epoch}': temp_test_res})
            test_pred_path = os.path.join(self.args['result_dir'], 'test_prediction.csv')
            if os.path.exists(test_pred_path):
                test_pred_df = pd.read_csv(test_pred_path)
                test_pred_df = pd.merge(test_pred_df, cur_test_ret[['images_id', f'pred_report_{epoch}']], on='images_id')
                test_pred_df.to_csv(test_pred_path, index=False)
            else:
                cur_test_ret.to_csv(test_pred_path, index=False)
        if self.args['lr_scheduler'] == 'StepLR':
            self.lr_scheduler.step()
        else:
            self.lr_scheduler.step(val_met[self.args['lr_monitor_metric']])
        # print(log)
        self.runner.log(log)
        self.logger.info("#############################################################")
        return log


class Tester(BaseTrainer):
    def __init__(self, model, metric_ftns, optimizer, args, lr_scheduler, train_loader_inc, train_loader_not_inc,
                 val_loader_inc, val_loader_not_inc, test_loader_inc, test_loader_not_inc, logger, task,
                 runner, is_save_checkpoint):
        super(Tester, self).__init__(model, metric_ftns, optimizer, args, task, is_save_checkpoint, logger)
        self.lr_scheduler = lr_scheduler
        self.train_loader_not_inc = train_loader_not_inc
        self.train_loader_inc = train_loader_inc
        self.val_loader_not_inc = val_loader_not_inc
        self.val_loader_inc = val_loader_inc
        self.test_loader_not_inc = test_loader_not_inc
        self.test_loader_inc = test_loader_inc
        self.runner = runner

    def test(self):
        log = {}
        self.model.eval()
        with torch.no_grad():
            if self.test_loader_inc is not None:
                test_gts, test_res, test_images_ids = [], [], []
                print(f"test dataset with indication section+++++++++++++++++++")
                self.logger.info(f"test dataset with indication section+++++++++++++++++++")
                for _, (images_id, images, reports_ids, reports_masks, sk_ids, sk_masks, inc_ids, inc_masks) in enumerate(
                        self.test_loader_inc):
                    images = images.to(self.device)
                    reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)
                    gen_texts, gt_texts = self.model(images, reports_ids, reports_masks, sk_ids, sk_masks,
                                                     inc_ids, inc_masks, mode='sample', is_contain_indication=True)
                    test_res.extend(gen_texts)
                    test_gts.extend(gt_texts)
                    test_images_ids.extend(images_id)

            print(f"test dataset only with similar historical cases+++++++++++++++++++")
            self.logger.info(f"test dataset only with similar historical cases+++++++++++++++++++")
            for _, (images_id, images, reports_ids, reports_masks, sk_ids, sk_masks) in enumerate(
                    self.test_loader_not_inc):
                images = images.to(self.device)
                reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)
                gen_texts, gt_texts = self.model(images, reports_ids, reports_masks, sk_ids, sk_masks, mode='sample')
                test_res.extend(gen_texts)
                test_gts.extend(gt_texts)
                test_images_ids.extend(images_id)

            test_met = self.metric_ftns(gts=test_gts, res=test_res, args=self.args)
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            logg_info = ''
            for k, v in test_met.items():
                logg_info += f"{k}: {v}; "
            self.logger.info(f"test metrics: {logg_info}")

            # save the metrics and the predict results
            temp_ids, temp_test_gts, temp_test_res = list(test_met.keys()), [None] * len(test_met), list(test_met.values())
            temp_ids.extend(test_images_ids)
            temp_test_gts.extend(test_gts)
            temp_test_res.extend(test_res)
            cur_test_ret = pd.DataFrame({'images_id': temp_ids, 'ground_truth': temp_test_gts,
                                        f'pred_report': temp_test_res})
            test_pred_path = os.path.join(self.args['result_dir'], 'test_prediction.csv')
            if os.path.exists(test_pred_path):
                test_pred_df = pd.read_csv(test_pred_path)
                test_pred_df = pd.merge(test_pred_df, cur_test_ret[['images_id', f'pred_report']], on='images_id')
                test_pred_df.to_csv(test_pred_path, index=False)
            else:
                cur_test_ret.to_csv(test_pred_path, index=False)
        self.runner.log(log)
        self.logger.info("#############################################################")
        return log

    def test_for_each_sample_with_indication(self):
        log = {}
        self.model.eval()
        all_metrics = ['F1-Radgraph-partial',"chexbert_5_micro_f1", 'chexbert_5_macro_f1',
                       "chexbert_all_micro_f1", 'chexbert_all_macro_f1', 'BLEU_1', 'BLEU_2', 'BLEU_3',
                       'BLEU_4', 'METEOR', 'ROUGE_L', "CIDer"]
        with torch.no_grad():
            if self.test_loader_inc is not None:
                sample_results = []
                print(f"test dataset with indication section+++++++++++++++++++")
                self.logger.info(f"test dataset with indication section+++++++++++++++++++")
                for _, (images_id, images, reports_ids, reports_masks, sk_ids, sk_masks, inc_ids, inc_masks) in enumerate(
                        self.test_loader_inc):
                    images = images.to(self.device)
                    reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)
                    gen_texts, gt_texts = self.model(images, reports_ids, reports_masks, sk_ids, sk_masks,
                                                     inc_ids, inc_masks, mode='sample', is_contain_indication=True)
                    scores = compute_all_scores(gt_texts, gen_texts, self.args)
                    if scores['F1-Radgraph-partial'] < 0.6:
                        continue
                    print(f"step: {_}/{len(self.test_loader_inc)}", images_id[0], scores)
                    print('*****************************************')
                    item = [images_id[0].split('_')[-1], gt_texts[0], gen_texts[0], *list(scores.values()), len(gt_texts[0].split(' ')), len(gen_texts[0].split(' '))]
                    sample_results.append(item)

                keys = ['image_id', 'gt_report', 'gen_report', *all_metrics, 'gt_text_len', 'gen_text_len']
                sample_results = pd.DataFrame(sample_results, columns=keys)
                sample_results.to_csv("sample_results_for samples with indication.csv", index=False)

    def plot(self):
        root = '/home/miao/data/Code/SEI-Results/mimic-cxr/'
        pred_path = root + '/test_prediction_with_factual_serialization.csv'
        pred_df = pd.read_csv(pred_path)
        assert self.args['batch_size'] == 1
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(root, "attentions_entities"), exist_ok=True)
        # data_path = os.path.join(root, 'FSE-plot_cases.xlsx')
        # data = pd.read_excel(data_path)
        # image_id_list = data['images_id'].tolist()
        # image_id_list = ['c6d9dcd8-49e961d7-227e2c94-92994086-9831113b', 'b529320a-394d7b79-a3e8c3da-c28c6b94-7ec08b51',
        #                  '37f7e3ca-93ef1bc3-81e615c8-a061addd-3a3b6dbf', 'f2b4864c-c60e842d-258889c6-61e08bca-a7990195']
        image_id_list = pred_df['image_id'].tolist()
        all_metrics = ['F1-Radgraph-partial', 'chexbert_5_micro_f1',
                       'chexbert_all_micro_f1', 'BLEU_1', 'BLEU_2', 'BLEU_3',
                       'BLEU_4', 'METEOR', 'ROUGE_L']
        ann_data = json.load(open(self.args['ann_path']))
        del ann_data['train']
        del ann_data['val']
        id2image_path = {}
        new_image_id_list = []
        for value in ann_data['test']:
            if value['id'] not in image_id_list or len(value['core_findings']) == 0 or len(value['indication_core_findings']) == 0:
                continue

            id2image_path[value['id']] = {
                'image_path': value['image_path'][0],
                'gt_fs': value['core_findings'],
                'gt_report': value['report'],
                'similar_historical_cases': value['specific_knowledge']['sk_keywords'][:5],
                'indication': value['indication'],
                'indication_core_findings': value['indication_core_findings']
            }
            new_image_id_list.append(value['id'])
        del ann_data

        self.model.eval()
        final_analysis_cases = {}
        with torch.no_grad():
            for _, (image_id, images, reports_ids, reports_masks, sk_ids, sk_masks, inc_ids, inc_masks) in enumerate(
                    self.test_loader_inc):
                image_id = image_id[0].split('_')[-1]
                if image_id not in new_image_id_list:
                    continue
                images = images.to(self.device)
                reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)
                gen_texts, gt_texts = self.model(images, reports_ids, reports_masks, sk_ids, sk_masks,
                                                 inc_ids, inc_masks, mode='sample', is_contain_indication=True)

                scores = compute_all_scores(gt_texts, gen_texts, self.args)
                print(image_id, scores)
                # ori_images
                ori_image_path = id2image_path[image_id]['image_path']
                ori_image = np.array(Image.open(os.path.join(self.args['image_dir'], ori_image_path)).convert('RGB'))

                chile_dir = f'{root}/attentions_entities/{image_id}'
                os.makedirs(chile_dir, exist_ok=True)
                p = os.path.join(self.args['image_dir'], ori_image_path)
                d = f'{chile_dir}/ori.jpg'
                os.system(f'cp {p} "{d}"')

                attention_weights = [layer.src_attn.attn.cpu().numpy()[:, :, :-1] for layer in
                                     self.model.text_decoder.model.decoder.layers][0][0]
                image_id_idx = pred_df['image_id'] == image_id
                gen_fs, gen_fs_idx = pred_df.loc[image_id_idx, 'gen_fs'].item(), pred_df.loc[image_id_idx, 'gen_fs_index'].item()
                gen_fs, gen_fs_idx = eval(gen_fs), eval(gen_fs_idx)
                gt_report, gen_report = pred_df.loc[image_id_idx, 'ground_truth'].item(), pred_df.loc[image_id_idx, 'gen_text'].item()
                item_result = {
                    'gt_report': gt_report,
                    'gen_report': gen_report,
                    'gt_fs': id2image_path[image_id]['gt_fs'],
                    'gen_fs': gen_fs,
                    'similar_historical_cases': id2image_path[image_id]['similar_historical_cases'],
                    'indication': id2image_path[image_id]['indication'],
                    'indication_core_findings': id2image_path[image_id]['indication_core_findings']
                }
                for m in all_metrics:
                    item_result[m] = scores[m]

                # the heatmap for factual sequences
                gen_report_words, old_gen_report_words = gen_texts[0].split(' '), gen_report.split(' ')
                for sen_idx, (fs_idx, factual_sequence) in enumerate(zip(gen_fs_idx, gen_fs)):
                    words_fs = ' '.join(np.array(gen_report_words)[fs_idx])
                    # assert words_fs == factual_sequence
                    atten = attention_weights[:, fs_idx, :]
                    heatmap = generate_heatmap(ori_image, atten.mean(0).mean(0).squeeze())
                    image_path = f'{chile_dir}/{sen_idx}_{factual_sequence}.png'
                    cv2.imwrite(image_path, heatmap)
                final_analysis_cases[image_id] = item_result

        with open(f'{root}/SEI-plot-cases-partial.json', 'w') as outfile:
            json.dump(final_analysis_cases, outfile, indent=2)


