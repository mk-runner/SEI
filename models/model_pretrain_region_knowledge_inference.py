import copy
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from models.vision_encoder.vit import build_vit_extractor
from modules.visual_extractor import ResNet
from modules.utils import GlobalEmbedding, LocalEmbedding


class PretrainInference(nn.Module):
    def __init__(self, args: dict, data_name: str):
        super(PretrainInference, self).__init__()
        self.args = args

        # ==========define model modules===================#
        # define visual encoder
        assert args['visual_encoder'] in ['resnet101', 'ViT-B-32']
        if args["visual_encoder"] == 'resnet101':
            self.visual_extractor = ResNet(args)
            visual_dim = 2048
        elif args['visual_encoder'] == 'ViT-B-32':
            self.visual_extractor = build_vit_extractor(args)
            visual_dim = 768
        else:
            raise ValueError(f'the visual encoder {args["visual_encoder"]} is not support!')

        # ==========define contrastive learning modules===================#
        # define the local embedding and global embedding for uni-modal
        if args['data_name'] == 'mimic_cxr':
            self.visual_global = GlobalEmbedding(visual_dim, hidden_dim=visual_dim, output_dim=args['output_dim'])
        else:
            self.visual_global = GlobalEmbedding(visual_dim, hidden_dim=visual_dim, output_dim=args['output_dim'])
        self.visual_local = LocalEmbedding(input_dim=visual_dim, hidden_dim=visual_dim, output_dim=args['output_dim'])

        # region-level contrastive learning
        # patch local attention layer
        # self.patch_local_layer = nn.MultiheadAttention(args['output_dim'], args['proj_num_heads'], batch_first=True)
        # sentence local attention layer
        # self.word_local_layer = nn.MultiheadAttention(args['output_dim'], args['proj_num_heads'], batch_first=True)

        # define the visual forward
        assert data_name in ['iu_xray', 'mimic_cxr', 'mix']
        if data_name == 'iu_xray':
            self.visual_forward = self.visual_forward_iu_xray
        else:
            self.visual_forward = self.visual_forward_mimic_cxr
        self.freeze_encoder_models(freeze_image_encoder=True)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params / 1e6)

    def freeze_encoder_models(self, freeze_image_encoder: bool = True):
        if freeze_image_encoder:
            # visual encoder
            for param in self.visual_extractor.parameters():
                param.requires_grad = False
            for param in self.visual_local.parameters():
                param.requires_grad = False
            for param in self.visual_global.parameters():
                param.requires_grad = False

    def visual_forward_iu_xray(self, images):
        # note that the first images are used to retrieve corresponding similar historical cases
        att_feats, fc_feats = self.visual_extractor(images[:, 0])
        # att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        # fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        # att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        # fc_feats = torch.mean(torch.stack([fc_feats_0, fc_feats_1], 0), dim=0)
        # fc
        return fc_feats, att_feats

    def visual_forward_mimic_cxr(self, images):
        att_feats, fc_feats = self.visual_extractor(images)  # attr_feats: patch feature; fc_feats: avg pool feats
        return fc_feats, att_feats

    def forward(self, images):
        fc_feats, att_feats = self.visual_forward(images)  # fc_feats and att_feats are global and focal features
        v_att_feats = F.normalize(self.visual_local(att_feats), dim=-1)
        v_fc_feats = F.normalize(self.visual_global(fc_feats), dim=-1)

        v_feats = torch.cat([v_fc_feats.unsqueeze(1), v_att_feats], dim=1)
        return v_feats
