import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoConfig

from models.language_encoder.language_model import TextDecoderModel, TextEncoderModel
from models.language_encoder.bert_model import BertCrossLayer
from models.vision_encoder.vit import build_vit_extractor
from modules.base_cmn import BaseCMN
from modules.encoder_decoder import EncoderDecoder
from modules.loss import compute_lm_loss
from modules.utils import GlobalEmbedding, LocalEmbedding, init_weights, get_extended_attention_mask
from modules.visual_extractor import ResNet


class FineTune(nn.Module):
    def __init__(self, args: dict, tokenizer: object, data_name: str):
        super(FineTune, self).__init__()
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

        # define text decoder
        assert args['text_decoder'] in ['r2gen', 'cmn', 'bert']
        if args['text_decoder'] == 'r2gen':
            self.text_decoder = EncoderDecoder(args, tokenizer)
        elif args['text_decoder'] == 'cmn':
            self.text_decoder = BaseCMN(args, tokenizer)
        elif args['text_decoder'] == 'bert':
            self.text_decoder = TextDecoderModel(args, tokenizer)
        else:
            raise ValueError(f'the text decoder {args["text_decoder"]} is not support!')

        # ==========define contrastive learning modules===================#
        # define the local embedding and global embedding for uni-modal
        if args['data_name'] == 'mimic_cxr':
            self.visual_global = GlobalEmbedding(visual_dim, hidden_dim=visual_dim, output_dim=args['output_dim'])
        else:
            self.visual_global = GlobalEmbedding(2 * visual_dim, hidden_dim=visual_dim, output_dim=args['output_dim'])
        self.visual_local = LocalEmbedding(input_dim=visual_dim, hidden_dim=visual_dim, output_dim=args['output_dim'])
        text_dim = self.text_encoder.encoder.config.hidden_size
        self.text_global = GlobalEmbedding(input_dim=text_dim, hidden_dim=text_dim, output_dim=args['output_dim'])
        self.text_local = LocalEmbedding(input_dim=text_dim, hidden_dim=text_dim, output_dim=args['output_dim'])

        # # fusion module
        # decoder_layer = nn.TransformerDecoderLayer(d_model=args['output_dim'], nhead=args['fusion_num_heads'],
        #                                            batch_first=True)
        # self.fusion_module = nn.TransformerDecoder(decoder_layer, num_layers=args['fusion_num_layers'])\
        fusion_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args['fusion_checkpoint'],
            vocab_size=args["vocab_size"],
            hidden_size=args["output_dim"],
            num_hidden_layers=args["sk_fusion_num_layers"],
            max_position_embeddings=args["max_seq_len"],
            num_attention_heads=args["fusion_num_heads"],
            eos_token_id=tokenizer.token_to_id('[EOS]'),
            bos_token_id=tokenizer.token_to_id('[BOS]'),
            pad_token_id=tokenizer.token_to_id('[PAD]'),
        )
        # fusion_config.hidden_dropout_prob=args["fusion_drop_rate"]
        # fusion_config.attention_probs_dropout_prob=args["fusion_drop_rate"]
        # fusion_config.intermediate_size = args['fusion_intermediate_size']
        # fusion_config.hidden_act = args['fusion_hidden_act']
        self.multi_modal_vision_layers = nn.ModuleList(
            [BertCrossLayer(fusion_config) for _ in range(args['sk_fusion_num_layers'])])

        # region-level contrastive learning
        # patch local attention layer
        # self.patch_local_layer = nn.MultiheadAttention(args['output_dim'], args['proj_num_heads'], batch_first=True)
        # sentence local attention layer
        # self.word_local_layer = nn.MultiheadAttention(args['output_dim'], args['proj_num_heads'], batch_first=True)

        # define the modality type
        self.modality_type_embeddings = nn.Embedding(2, args['output_dim'])
        self.modality_type_embeddings.apply(init_weights)

        # define the visual forward
        assert data_name in ['iu_xray', 'mimic_cxr', 'mix']
        if data_name == 'iu_xray':
            self.visual_forward = self.visual_forward_iu_xray
        else:
            self.visual_forward = self.visual_forward_mimic_cxr

        # define the text_decoder forward
        if args['text_decoder'] in ['r2gen', 'cmn']:
            self.text_decoder_forward = self.text_decoder_forward_r2gen
        else:
            self.text_decoder_forward = self.text_decoder_forward_bert

        self.freeze_encoder_models(args['freeze_image_encoder'], args['freeze_text_encoder'])

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params / 1e6)

    def freeze_encoder_models(self, freeze_image_encoder: bool = True, freeze_text_encoder: bool = False):
        if freeze_image_encoder:
            # visual encoder
            for param in self.visual_extractor.parameters():
                param.requires_grad = False
            for param in self.visual_local.parameters():
                param.requires_grad = False
            for param in self.visual_global.parameters():
                param.requires_grad = False
        if freeze_text_encoder:
            # text encoder
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.text_local.parameters():
                param.requires_grad = False
            for param in self.text_global.parameters():
                param.requires_grad = False

    def visual_forward_iu_xray(self, images):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        # fc_feats = torch.mean(torch.stack([fc_feats_0, fc_feats_1], 0), dim=0)
        return fc_feats, att_feats

    def visual_forward_mimic_cxr(self, images):
        att_feats, fc_feats = self.visual_extractor(images)  # attr_feats: patch feature; fc_feats: avg pool feats
        return fc_feats, att_feats

    def text_decoder_forward_r2gen(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, mode='train'):

        if mode == 'train':
            output = self.text_decoder(input_ids, encoder_hidden_states, attention_mask=attention_mask,
                                       encoder_attention_mask=encoder_attention_mask, mode='forward')
            lm_loss = compute_lm_loss(output, input_ids, attention_mask)
            return lm_loss
        else:
            output, _ = self.text_decoder(encoder_hidden_states, encoder_attention_masks=encoder_attention_mask,
                                          mode='sample')
            gen_texts, gt_texts = [], []
            for pred, gt in zip(output.cpu().tolist(), input_ids.cpu().tolist()):
                gen_text = self.tokenizer.decode(pred)
                gt_text = self.tokenizer.decode(gt)
                gen_texts.append(gen_text)
                gt_texts.append(gt_text)
            gen_texts = [text if len(text) > 0 else "there is no evidence of pulmonary." for text in gen_texts]
            return [gen_texts, gt_texts]

    def text_decoder_forward_bert(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, mode='train'):
        if mode == 'train':  # att_feats (16, 49, 2048), fc_feats (16, 2048), target (16, 100)
            lm_loss = self.text_decoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        encoder_hidden_states=encoder_hidden_states,
                                        encoder_attention_mask=encoder_attention_mask)
            return lm_loss
        elif mode == 'sample':
            [gen_texts, gt_texts] = self.text_decoder.evaluation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                tokenizer=self.tokenizer
            )
            return [gen_texts, gt_texts]  # []
        else:
            raise ValueError('it is not implement!')

    def forward(self, images, report_ids, report_masks, sk_ids, sk_masks, mode='train'):
        # image embedding
        device = images.device
        v_fc_feats, v_att_feats = self.visual_forward(images)  # fc_feats and att_feats are global and focal features
        v_att_feats = F.normalize(self.visual_local(v_att_feats), dim=-1)
        v_fc_feats = F.normalize(self.visual_global(v_fc_feats), dim=-1)
        # obtain the encoder_hidden_states and encoder_attention_mask only using image feats and corresponding masks
        visual_feats = torch.cat([v_fc_feats.unsqueeze(1), v_att_feats], dim=1)
        encoder_attention_mask = torch.ones(visual_feats.size()[:2], dtype=torch.long).to(device)
        # == Begin: Assign Type Embeddings (uni-modal features + modal_type) ==
        # text modal type is one; vision modal type is zero.
        encoder_hidden_states = visual_feats

        # obtain the encoder_hidden_states and encoder_attention_mask
        if len(sk_ids) != 0:
            # two methods for obtaining the text features and attention mask of specific knowledge
            if self.args['sk_fusion_strategy'] == 'mean':
                encoder_hidden_states = encoder_hidden_states + self.modality_type_embeddings(torch.zeros_like(encoder_attention_mask))
                # Text features are composed of the average values of multiple specific knowledge
                text_feats = []
                for s_ids, s_masks in zip(sk_ids, sk_masks):
                    s_ids, s_masks = s_ids.to(device, non_blocking=True), s_masks.to(device, non_blocking=True)
                    t_feats = self.text_encoder(input_ids=s_ids, attention_mask=s_masks)
                    # if it has normalization, all values have been averaged.
                    # text_feats.append(F.normalize(t_feats * s_masks.unsqueeze(2), dim=-1))
                    text_feats.append(t_feats * s_masks.unsqueeze(2))
                text_feats = torch.sum(torch.stack(text_feats, dim=1), dim=1)  # (b, seq_len, dim)
                # text attention mask
                text_masks = torch.sum(torch.stack(sk_masks, 1), dim=1).to(device)
                text_feats = text_feats / (text_masks.float().unsqueeze(2) + 1e-9)
                text_masks = torch.where(text_masks != 0, 1, 0)  # (b, seq_len)

                # text projection and normalization
                t_att_feats = F.normalize(self.text_local(text_feats[:, 1:, :]), dim=-1)
                t_fc_feats = F.normalize(self.text_global(text_feats[:, 0, :]), dim=-1)
                text_feats = torch.cat([t_fc_feats.unsqueeze(1), t_att_feats], dim=1)

                # obtain encoder_hidden_states using image feats and text feats via concat operation
                text_feats = text_feats + self.modality_type_embeddings(torch.ones_like(text_masks))
                encoder_hidden_states = torch.cat([encoder_hidden_states, text_feats], dim=1)
                encoder_attention_mask = torch.cat([encoder_attention_mask, text_masks], dim=1)
            else:  # concat
                # Text features are composed of the concatenation of multiple specific knowledge
                text_att_feats, text_masks, text_fc_feats = [], [], []
                for i, (s_ids, s_masks) in enumerate(zip(sk_ids, sk_masks)):
                    s_ids, s_masks = s_ids.to(device, non_blocking=True), s_masks.to(device, non_blocking=True)
                    t_feats = self.text_encoder(input_ids=s_ids, attention_mask=s_masks)
                    text_fc_feats.append(t_feats[:, 0, :])
                    text_att_feats.append(t_feats[:, 1:, :])
                    if i != 0:
                        text_masks.append(s_masks[:, 1:])
                    else:
                        text_masks.append(s_masks)
                text_att_feats, text_masks = torch.cat(text_att_feats, dim=1), torch.cat(text_masks, dim=1)
                # the global text feature is the mean values of multiple specific knowledge
                text_fc_feats = torch.mean(torch.stack(text_fc_feats, 0), 0)
                t_att_feats = F.normalize(self.text_local(text_att_feats), dim=-1)
                t_fc_feats = F.normalize(self.text_global(text_fc_feats), dim=-1)
                text_feats = torch.cat([t_fc_feats.unsqueeze(1), t_att_feats], dim=1)

                extended_image_masks = get_extended_attention_mask(encoder_attention_mask, encoder_attention_mask.size())
                extended_text_masks = get_extended_attention_mask(text_masks, text_masks.size())

                x, y = encoder_hidden_states, text_feats
                for layer_idx, image_layer in enumerate(self.multi_modal_vision_layers):
                    # == Begin: Fetch the intermediate outputs (different layers to perform MIM) ==
                    # == End  : Fetch the intermediate outputs (different layers to perform MIM) ==
                    # == Begin: Co-Attention ==
                    x1 = image_layer(x, y, attention_mask=extended_image_masks,
                                     encoder_attention_mask=extended_text_masks, output_attentions=True)
                    x = x1[0]
                    # == End: Co-Attention ==
                    # == Begin: For visualization: Return the attention weights ==
                encoder_hidden_states = x

        ret = self.text_decoder_forward(report_ids, report_masks, encoder_hidden_states, encoder_attention_mask, mode=mode)
        if mode == 'train':  # att_feats (16, 49, 2048), fc_feats (16, 2048), target (16, 100)
            return {
                'lm': ret,
                'all_loss': ret
            }
        elif mode == 'sample':
            gen_texts, gt_texts = ret
            return [gen_texts, gt_texts]
        else:
            raise ValueError
