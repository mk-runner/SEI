import torch
import torch.nn as nn
import functools

import tqdm
from transformers.models.auto import AutoModelForCausalLM, AutoConfig, AutoModel
from transformers.models.bert_generation import BertGenerationConfig, BertGenerationDecoder, BertGenerationEncoder
from .beam_search import prepare_inputs_for_generation, _validate_model_kwargs, beam_search


class TextDecoderModel(nn.Module):
    """
    If proto is mentioned in decoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationDecoder model from decoder dict.
    """

    def __init__(self, config, tokenizer):
        super().__init__()
        if config['text_checkpoint'] is not None:
            dec_config = AutoConfig.from_pretrained(config['text_checkpoint'])
            dec_config.is_decoder = True
            dec_config.add_cross_attention = True
            dec_config.vocab_size = config['vocab_size']
            dec_config.eos_token_id = tokenizer.token_to_id('[EOS]')
            dec_config.bos_token_id = tokenizer.token_to_id('[BOS]')
            dec_config.pad_token_id = tokenizer.token_to_id('[PAD]')
            dec_config.hidden_size = config['decoder_hidden_size']
            dec_config.num_hidden_layers = config['decoder_num_hidden_layers']
            dec_config.max_length = config['max_seq_len']
            dec_config.num_attention_heads = config['decoder_num_attention_heads']
            self.decoder = AutoModelForCausalLM.from_pretrained(config['text_checkpoint'], config=dec_config,
                                                                ignore_mismatched_sizes=True)
        else:
            dec_config = BertGenerationConfig()
            dec_config.is_decoder = True
            dec_config.add_cross_attention = True
            dec_config.vocab_size = config['vocab_size']
            dec_config.eos_token_id = tokenizer.token_to_id('[EOS]')
            dec_config.bos_token_id = tokenizer.token_to_id('[BOS]')
            dec_config.pad_token_id = tokenizer.token_to_id('[PAD]')
            dec_config.hidden_size = config['decoder_hidden_size']
            dec_config.num_hidden_layers = config['decoder_num_hidden_layers']
            dec_config.max_length = config['max_seq_len']
            self.decoder = BertGenerationDecoder(dec_config)

        # Evaluation
        self.decoder.prepare_inputs_for_generation = functools.partial(prepare_inputs_for_generation, self.decoder)
        # We override _validate_model_kwargs width empty function because we add custom model kwargs that triggers
        # errors in original _validate_model_kwargs
        self.decoder._validate_model_kwargs = functools.partial(_validate_model_kwargs, self.decoder)

        # Inference
        self.generate = self.decoder.generate
        self.config = self.decoder.config
        self.beam_size = config['beam_size']

    def forward(self, input_ids, attention_mask, encoder_hidden_states=None, encoder_attention_mask=None, **kwargs):
        out = self.decoder(input_ids=input_ids,
                           attention_mask=attention_mask,
                           encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask,
                           labels=input_ids,
                           **kwargs)
        return out.loss

    def evaluation(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, tokenizer):
        # We are in an ensembling scenario, we override huggingface beam-search function
        self.decoder.beam_search = functools.partial(beam_search, self.decoder)

        # Get tokenizer and reference sentences from dataloader
        max_len = self.config.max_length
        bos_token_id, eos_token_id, pad_token_id = tokenizer.token_to_id('[BOS]'), tokenizer.token_to_id(
            '[EOS]'), tokenizer.token_to_id('[PAD]')

        with torch.no_grad():
            batch_size = input_ids.shape[0]
            bos_input_ids = torch.ones((batch_size, 1), dtype=torch.long).to(input_ids) * bos_token_id
            expanded_return_idx = (
                torch.arange(batch_size).view(-1, 1).repeat(1, self.beam_size).view(-1).to(input_ids)
            )
            model_kwargs = {
                'encoders_outputs': [
                    {
                        "encoder_hidden_states": encoder_hidden_states.index_select(0, expanded_return_idx),
                        "encoder_attention_mask": encoder_attention_mask.index_select(0, expanded_return_idx)
                    }
                ],
                "hf_models": [self.decoder]
            }
            output = self.decoder.generate(
                input_ids=bos_input_ids,
                num_return_sequences=1,
                max_length=max_len,
                num_beams=self.beam_size,
                length_penalty=1.0,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                use_cache=True,
                **model_kwargs
            )
            gen_texts, gt_texts = [], []
            for pred, gt in zip(output.cpu().tolist(), input_ids.cpu().tolist()):
                gen_text = tokenizer.decode(pred)
                gt_text = tokenizer.decode(gt)
                gen_texts.append(gen_text)
                gt_texts.append(gt_text)
            gen_texts = [text if len(text) > 0 else "there is no evidence of pulmonary." for text in gen_texts]
            return [gen_texts, gt_texts]

    def __repr__(self):
        s = str(type(self.decoder).__name__) + '(' + str(self.decoder.config) + ')\n'
        return s


class TextEncoderModel(nn.Module):
    """
    If proto is mentioned in decoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationDecoder model from decoder dict.
    """

    def __init__(self, config, tokenizer):
        super().__init__()
        if config['text_checkpoint'] is not None:
            enc_config = AutoConfig.from_pretrained(config['text_checkpoint'])
            enc_config.vocab_size = config['vocab_size']
            enc_config.eos_token_id = tokenizer.token_to_id('[EOS]')
            enc_config.bos_token_id = tokenizer.token_to_id('[BOS]')
            enc_config.pad_token_id = tokenizer.token_to_id('[PAD]')
            enc_config.hidden_size = config['encoder_hidden_size']
            enc_config.num_hidden_layers = config['encoder_num_hidden_layers']
            enc_config.max_length = config['max_seq_len']
            self.encoder = AutoModel.from_pretrained(config['text_checkpoint'], config=enc_config,
                                                     ignore_mismatched_sizes=True)
        else:
            enc_config = BertGenerationConfig()
            enc_config.vocab_size = config['vocab_size']
            enc_config.eos_token_id = tokenizer.token_to_id('[EOS]')
            enc_config.bos_token_id = tokenizer.token_to_id('[BOS]')
            enc_config.pad_token_id = tokenizer.token_to_id('[PAD]')
            enc_config.hidden_size = config['encoder_hidden_size']
            enc_config.num_hidden_layers = config['encoder_num_hidden_layers']
            enc_config.max_length = config['max_seq_len']
            self.encoder = BertGenerationEncoder(enc_config)

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask,
                           **kwargs)[0]
        return out

    def __repr__(self):
        s = str(type(self.encoder).__name__) + '\n'
        return s
