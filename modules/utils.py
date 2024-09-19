import argparse
import json
import os.path
import random
import warnings
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import Tensor, device
from torch import nn
from tqdm import tqdm

from .metrics.metrics import compute_ce_scores


class SetLogger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def info(self, s):
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(s + '\n')
        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()


def setup_seed(seed):
    # seed init
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch seed init
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_arguments():
    parse = argparse.ArgumentParser()
    # basic configuration
    # pretrain: multi-granularity cross-modal alignment
    # pretrain_inference: retrieve similar historical cases for each radiographs without relying on gradient
    # finetune: train text decoder based on similar historical cases
    # inference: text generation for test dataset and compute performance
    parse.add_argument('--task', type=str, default='test',
                       choices=['pretrain', 'pretrain_inference', 'finetune', 'test'])
    # data configuration
    parse.add_argument('--data_name', type=str, choices=['mimic_cxr'], default='mimic_cxr')
    parse.add_argument('--mimic_cxr_ann_path', type=str,
                       default='mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_with_fs_v0227.json')
    parse.add_argument('--iu_xray_ann_path', type=str)
    parse.add_argument('--text_decoder', type=str, default='r2gen',
                       help='may be only support r2gen')
    parse.add_argument('--visual_encoder', type=str, default='resnet101',
                       help='may be only support resnet101')
    parse.add_argument('--tokenizer_model', type=str, choices=['wordlevel', 'wordpiece'], default='wordlevel')
    parse.add_argument('--tokenizer_type', type=str, choices=['uncased', 'cased'], default='uncased')
    parse.add_argument('--max_seq_len', type=int, default=100)
    parse.add_argument('--freeze_image_encoder', action='store_true', help='whether freeze the image encoder')
    parse.add_argument('--freeze_text_encoder', action='store_true', help='whether freeze the text encoder')
    parse.add_argument('--is_save_checkpoint', action='store_true', help='whether save checkpoint')
    # specific knowledge configuration
    parse.add_argument('--sk_type', type=str, choices=['report', 'keywords'], default='keywords',
                       help="presentation form of similar historical cases, default is factual entity sequences")
    parse.add_argument('--sk_topk', type=int, default=5, help='the number of similar historical cases')
    parse.add_argument('--is_add_indication', action='store_true', help='whether add indication section')
    # parse.add_argument('--is_add_indication', action='store_false', help='whether add indication section')
    parse.add_argument('--sk_fusion_strategy', type=str, choices=['mean', 'cat'], default='cat')
    parse.add_argument('--sk_fusion_num_layers', type=int, default=1)
    parse.add_argument('--sk_file_name', type=str, default='_v0107_')
    # trainer configuration
    parse.add_argument('--optim', type=str, choices=['AdamW', 'RAdam', "Adam"], default='RAdam',
                       help='in the first stage, the optimization is AdamW with lr=5.0e-5, '
                            'and in the second stage, optimizer is RAdam')
    parse.add_argument('--lr_scheduler', type=str, choices=['StepLR', 'ReduceLROnPlateau'], default='ReduceLROnPlateau')
    parse.add_argument('--lr', type=float, default=5.0e-5)  # 5.0e-5
    parse.add_argument('--ft_monitor_metric', type=str, default='RCB')  # choices={metrics, RC, RB, RCB}
    parse.add_argument('--epochs', type=int, default=100)
    parse.add_argument('--batch_size', type=int, default=1)
    parse.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parse.add_argument('--load', type=str, help='whether to load the pre-trained model.')
    parse.add_argument('--version', type=str, default='long_sentence', help='the name of experiment')
    # sk_type and align_type is the same.
    parse.add_argument('--align_type', type=str, choices=['report', 'keywords'], default='keywords',
                       help='default use factual entity sequences for cross-modal alignment')
    parse.add_argument('--align_loss', type=str, choices=['local', 'global', 'multi-level'], default='multi-level')
    parse.add_argument('--ckpt_zoo_dir', type=str,
                       default='/home/miao/data/dataset/checkpoints',
                       help='checkpoint dir for calculating metrics')
    parse.add_argument('--chexbert_path', type=str, default='chexbert.pth', help='checkpoint')
    parse.add_argument('--bert_path', type=str, default='bert-base-uncased', help='checkpoint')
    parse.add_argument('--radgraph_path', type=str, default='radgraph', help='checkpoint')
    parse.add_argument('--resnet_path', type=str, default='resnet101-5d3b4d8f.pth', help='checkpoint')
    parse.add_argument('--scibert_path', type=str, default='scibert_scivocab_uncased',
                       help='checkpoint of text encoder')
    cmd = parse.parse_args()
    cmd.config = '../config/finetune_config.yaml'
    args = yaml.load(open(cmd.config), Loader=yaml.FullLoader)
    cmd = vars(cmd)
    args.update(cmd)
    if args['ckpt_zoo_dir']:
        # all checkpoints are placed in ckpt_zoo_dir
        ckpts = ['chexbert_path', "bert_path", 'radgraph_path', 'resnet_path', 'scibert_path']
        for k in ckpts:
            args[k] = os.path.join(args['ckpt_zoo_dir'], args[k])
    args['image_dir'] = args[f'{args["data_name"]}_image_dir']
    args['ann_path'] = args[f'{args["data_name"]}_ann_path']
    args['text_decoder'] = args['text_decoder'].lower()
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['result_dir'] = f'{args["result_dir"]}/{args["data_name"]}/{args["task"]}/{args["version"]}'
    os.makedirs(args['result_dir'], exist_ok=True)

    logger = SetLogger(f'{args["result_dir"]}/{args["task"]}_{args["text_decoder"]}_{args["sk_topk"]}.log', 'a')
    if args['task'] in ['pretrain', 'pretrain_inference']:
        args['monitor_mode'] = args['pt_monitor_mode']
        args['monitor_metric'] = args['pt_monitor_metric']
        args['lr_monitor_metric'] = args['pt_lr_monitor_metric']
    else:
        args['monitor_mode'] = args['ft_monitor_mode']
        args['monitor_metric'] = args['ft_monitor_metric']
        args['lr_monitor_metric'] = args['ft_lr_monitor_metric']
    return args, logger


def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def generate_heatmap(image, weights):
    # image = image.transpose(1, 2, 0)
    height, width, _ = image.shape
    weights = weights.reshape(int(weights.shape[0] ** 0.5), int(weights.shape[0] ** 0.5))
    weights = weights - np.min(weights)
    weights = weights / np.max(weights)
    weights = cv2.resize(weights, (width, height))
    weights = np.uint8(255 * weights)
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    result = heatmap * 0.5 + image * 0.5
    return result


class GlobalEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 2048,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        return self.head(x)


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),  # change the dimension
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)

        return x.permute(0, 2, 1)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def temp_compute_scores(ori_report, topk_reports, args):
    ori_report = [ori_report] * len(topk_reports)
    result = compute_ce_scores(ori_report, topk_reports, args)
    return result


def plot_images(images, metrics, images_dir, image_id, specific_knowledge_dir, split):
    fig = plt.figure(figsize=(10, 8))
    for i in range(len(images)):
        image = plt.imread(os.path.join(images_dir, images[i]))
        axes = fig.add_subplot(int(f'22%d' % (i+1)))
        axes.axis('off')
        if i == 0:
            if len(metrics) != 0:
                str_metric = ''
                for k, v in metrics.items():
                    str_metric += f'{k.lower()}:{v:.3f} '
                axes.set_title(str_metric.strip())
            else:
                axes.set_title('original image')
        axes.imshow(image, cmap='viridis')
    plt.savefig(os.path.join(specific_knowledge_dir, f'{split}_{image_id}_specific_knowledge.jpg'))
    # plt.show()
    plt.cla()
    plt.clf()


class PretrainTestAnalysis(object):
    def __init__(self, ann_path, topk, image_dir, sk_analysis_dir=None, data_name='mimic_cxr'):
        assert 1 <= topk <= 30
        # self.topk_ann_path = topk_ann_path
        self.ann_path = ann_path
        self.topk = topk
        self.image_dir = image_dir
        if sk_analysis_dir is not None:
            os.makedirs(sk_analysis_dir, exist_ok=True)
        self.sk_analysis_dir = sk_analysis_dir
        self.data_name = data_name


    def get_report_dict(self):
        id2report = {}
        ann_data = json.load(open(self.ann_path))
        for split, value in ann_data.items():
            for idx, item in tqdm(enumerate(value)):
                if self.data_name == 'mimic_cxr':
                    cur_idx = '_'.join([str(item['subject_id']), str(item['study_id']), item['id']])
                else:
                    cur_idx = item['id']
                id2report[cur_idx] = [item['report'], item['core_findings']]
        return id2report

    def get_specific_knowledge(self, id2image):
        ann_data = json.load(open(self.ann_path))
        # id2image = json.load(open(self.topk_ann_path))
        id2report = self.get_report_dict()
        new_ann_data = {}
        for split, value in ann_data.items():
            new_ann_data[split] = []
            for idx, item in tqdm(enumerate(value)):
                if self.data_name == 'mimic_cxr':
                    cur_idx = '_'.join([str(item['subject_id']), str(item['study_id']), item['id']])
                else:
                    cur_idx = item['id']
                try:
                    topk_images_id = id2image[cur_idx][:self.topk]
                    sk_reports = [id2report[i][0] for i in topk_images_id]
                    sk_keywords = [id2report[i][1] for i in topk_images_id]
                    specific_knowledge = {'reports': sk_reports, 'sk_keywords': sk_keywords}
                except:
                    specific_knowledge = {'reports': [], 'keywords': []}
                new_item = {
                    **item,
                    'specific_knowledge': specific_knowledge
                }
                new_ann_data[split].append(new_item)

        save_file_name = self.ann_path.split('.json')[0] + f'_best_reports_keywords_{self.topk}.json'
        json.dump(new_ann_data, open(save_file_name, 'w'), indent=2)

    def show_topk_images(self, args, id2image):
        ann_data = json.load(open(self.ann_path))
        # id2image = json.load(open(self.topk_ann_path))
        id2report = self.get_report_dict()
        for split, value in ann_data.items():
            idx = torch.randperm(len(value))[:10]
            for i in tqdm(idx):
                item = value[i]
                cur_idx = '_'.join([str(item['subject_id']), str(item['study_id']), item['id']])
                try:
                    topk_images_id = id2image[cur_idx][:self.topk]
                except:
                    continue

                # calculate the similarity between the report and its corresponding topk reports
                cur_reports = item['report']
                topk_reports = [id2report[ids][0] for ids in topk_images_id]
                metrics = temp_compute_scores(cur_reports, topk_reports, args)

                # show_topk_images including images and their reports
                print("--------------------------------------------------------------")
                print(i, metrics)
                print(cur_idx, item['core_findings'])
                topk_images_path = [item['image_path'][0]]
                for k, image_id in enumerate(topk_images_id[:3]):
                    _subject_id, _study_id, _dicom_id = image_id.split('_')
                    _image_path = f"p{_subject_id[:2]}/p{_subject_id}/s{_study_id}/{_dicom_id}.jpg"
                    topk_images_path.append(_image_path)
                    print(image_id, 'topk_image %d' % k, id2report[image_id][1])

                print("--------------------------------------------------------------")
                plot_images(topk_images_path, metrics, self.image_dir, cur_idx, self.sk_analysis_dir, split)


def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None):
    if device is not None:
        warnings.warn(
            "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
        )
    else:
        device = attention_mask.device
    batch_size, seq_length = input_shape
    seq_ids = torch.arange(seq_length, device=device)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    # in case past_key_values are used we need to add a prefix ones mask to the causal mask
    # causal and attention masks must have same type with pytorch version < 1.3
    causal_mask = causal_mask.to(attention_mask.dtype)

    if causal_mask.shape[1] < attention_mask.shape[1]:
        prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
        causal_mask = torch.cat(
            [
                torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                causal_mask,
            ],
            axis=-1,
        )

    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    return extended_attention_mask


def get_extended_attention_mask(
        attention_mask: Tensor, input_shape: Tuple[int], device: device = None, dtype: torch.float = None,
        is_decoder: bool = False
) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.
        device
        dtype
        is_decoder
    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    if dtype is None:
        dtype = torch.float

    if device is None:
        device = attention_mask.device

    if not (attention_mask.dim() == 2 and is_decoder):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if is_decoder:
            extended_attention_mask = create_extended_attention_mask_for_decoder(
                input_shape, attention_mask, device
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask