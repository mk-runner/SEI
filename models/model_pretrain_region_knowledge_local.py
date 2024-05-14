import copy
import math

import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from models.language_encoder.language_model import TextEncoderModel
from models.vision_encoder.vit import build_vit_extractor
from modules.visual_extractor import ResNet
from modules.utils import GlobalEmbedding, LocalEmbedding


class LocalPretrain(nn.Module):
    def __init__(self, args: dict, tokenizer: object, data_name: str):
        super(LocalPretrain, self).__init__()
        self.args = args
        self.tokenizer = tokenizer

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

        # define text encoder
        self.text_encoder = TextEncoderModel(args, tokenizer)

        # ==========define contrastive learning modules===================#
        # define the local embedding and global embedding for uni-modal
        if args['data_name'] == 'mimic_cxr':
            self.visual_global = GlobalEmbedding(visual_dim, hidden_dim=visual_dim, output_dim=args['output_dim'])
        else:
            self.visual_global = GlobalEmbedding(visual_dim, hidden_dim=visual_dim, output_dim=args['output_dim'])
        self.visual_local = LocalEmbedding(input_dim=visual_dim, hidden_dim=visual_dim, output_dim=args['output_dim'])
        text_dim = self.text_encoder.encoder.config.hidden_size
        self.text_global = GlobalEmbedding(input_dim=text_dim, hidden_dim=text_dim, output_dim=args['output_dim'])
        self.text_local = LocalEmbedding(input_dim=text_dim, hidden_dim=text_dim, output_dim=args['output_dim'])

        # region-level contrastive learning
        # patch local attention layer
        self.patch_local_layer = nn.MultiheadAttention(args['output_dim'], args['proj_num_heads'], batch_first=True)
        # sentence local attention layer
        self.word_local_layer = nn.MultiheadAttention(args['output_dim'], args['proj_num_heads'], batch_first=True)

        # define the visual forward
        assert data_name in ['iu_xray', 'mimic_cxr', 'mix']
        if data_name == 'iu_xray':
            self.visual_forward = self.visual_forward_iu_xray
        else:
            self.visual_forward = self.visual_forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params / 1e6)

    def visual_forward_iu_xray(self, images):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        # fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        fc_feats = torch.mean(torch.stack([fc_feats_0, fc_feats_1], 0), dim=0)
        return fc_feats, att_feats

    def visual_forward_mimic_cxr(self, images):
        att_feats, fc_feats = self.visual_extractor(images)  # attr_feats: patch feature; fc_feats: avg pool feats
        return fc_feats, att_feats

    def forward(self, images, radgraph_ids, radgraph_masks):
        # image embedding
        device, batch_size = images.device, images.shape[0]
        fc_feats, att_feats = self.visual_forward(images)  # fc_feats and att_feats are global and focal features
        v_att_feats = F.normalize(self.visual_local(att_feats), dim=-1)
        v_fc_feats = F.normalize(self.visual_global(fc_feats), dim=-1)

        # text embedding
        text_feats = self.text_encoder(input_ids=radgraph_ids, attention_mask=radgraph_masks)
        t_att_feats = F.normalize(self.text_local(text_feats[:, 1:, :]), dim=-1)
        t_fc_feats = F.normalize(self.text_global(text_feats[:, 0, :]), dim=-1)

        # # ====instance-level contrastive loss====
        # instance_sim = v_fc_feats @ t_fc_feats.t()
        # instance_sim_1 = t_fc_feats @ v_fc_feats.t()
        # instance_targets = torch.arange(batch_size).long().to(device)
        # loss_instance_1 = F.cross_entropy(instance_sim / self.args['instance_temp'], instance_targets)
        # loss_instance_2 = F.cross_entropy(instance_sim_1 / self.args['instance_temp'], instance_targets)
        # itc_instance_loss = (loss_instance_2 + loss_instance_1) / 2.0

        # ====region-level contrastive loss====
        # (word to visual_feats)
        # this local alignment can be modified using prior method (this is next work)
        is_self_att = False
        if is_self_att:
            t_att_output, _ = self.word_local_layer(t_att_feats, v_att_feats, v_att_feats)
        else:  # have no sqrt(D)
            t_att_sim = torch.bmm(t_att_feats, v_att_feats.permute(0, 2, 1))
            t_att_sco = F.softmax(t_att_sim / math.sqrt(t_att_feats.shape[2]), dim=-1)
            t_att_output = torch.bmm(t_att_sco, v_att_feats)
        t_att_output = F.normalize(t_att_output, dim=-1)
        word_sim = torch.bmm(t_att_feats, t_att_output.permute(0, 2, 1)) / self.args['region_temp']
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")  # the similarity between each word and each each
        word_targets = torch.arange(word_sim.shape[1]).long().repeat(word_sim.shape[0]).to(device)
        loss_word_1 = F.cross_entropy(word_sim_1, word_targets)

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = F.cross_entropy(word_sim_2, word_targets)
        loss_word = (loss_word_2 + loss_word_1) / 2.0

        # # (visual to text_feats)
        # if is_self_att:
        #     v_att_output, _ = self.patch_local_layer(v_att_feats, t_att_feats, t_att_feats)
        # else:
        #     v_att_sim = torch.bmm(v_att_feats, t_att_feats.permute(0, 2, 1))
        #     v_att_sco = F.softmax(v_att_sim / self.args['region_temp'], dim=-1)
        #     v_att_output = torch.bmm(v_att_sco, t_att_feats)
        #
        # v_att_output = F.normalize(v_att_output, dim=-1)
        # patch_sim = torch.bmm(v_att_feats, v_att_output.permute(0, 2, 1)) / self.args['region_temp']
        # patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")  # the similarity between each word and each each
        # patch_targets = torch.arange(patch_sim.shape[1]).long().repeat(patch_sim.shape[0]).to(device)
        # loss_patch_1 = F.cross_entropy(patch_sim_1, patch_targets)
        #
        # patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
        # loss_patch_2 = F.cross_entropy(patch_sim_2, patch_targets)
        # loss_path = (loss_patch_1 + loss_patch_2) / 2.0
        # itc_region_loss = (loss_patch_1 + loss_patch_2 + loss_word_1 + loss_word_2) / 4.0
        itc_region_loss = loss_word

        return {
            'itc_region': itc_region_loss,
            'itc_instance': torch.zeros((1,)),
            'all_loss': itc_region_loss
        }

    def obtain_visual_embeds(self, images):
        fc_feats, att_feats = self.visual_forward(images)  # fc_feats and att_feats are global and focal features
        v_att_feats = F.normalize(self.visual_local(att_feats), dim=-1)
        v_fc_feats = F.normalize(self.visual_global(fc_feats), dim=-1)
        v_feats = torch.stack([v_fc_feats.unsqueeze(1), v_att_feats], dim=1)
        return v_feats
