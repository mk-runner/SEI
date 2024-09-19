import re
import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def get_vocab_file(ann_path, data_name, case_type, corpus_path):
    all_ids = []  # avoid redundant reports
    data = json.load(open(ann_path))
    for item in data['train']:
        if data_name == 'mimic_cxr':
            ids = '_'.join([str(item['subject_id']), str(item["study_id"])])
        else:
            ids = item['id']
        if ids not in all_ids:
            all_ids.append(ids)
            if case_type == 'uncased':
                item['report'] = item['report'].lower()
                # item['indication_core_findings'] = item['indication_core_findings'].lower()
            report = item['report']
            # inc_core_findings = item['indication_core_findings']
            with open(corpus_path, 'a+') as f:
                f.write(report)
                f.write('\n')
                # f.write(inc_core_findings)
                # f.write('\n')


def train_tokenizer(corpus_path, tokenizer_path, model: str = 'wordpiece'):
    if model == 'wordpiece':
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.WordPiece()
        trainer = trainers.WordPieceTrainer(
            special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
        )
    else:
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(
            special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
        )
    files = [corpus_path]
    tokenizer.train(files, trainer)
    tokenizer.save(tokenizer_path)


def build_my_tokenizer(tokenizer_dir: str = "config/tokenizer", model: str = 'wordpiece', data_name: str = 'mimic_cxr',
                       ann_path: str = None, tokenizer_type='case', is_same_tokenizer: bool = False):
    model = model.lower()
    data_name = data_name.lower()
    tokenizer_type = tokenizer_type.lower()
    assert model in ['wordpiece', 'wordlevel']
    assert data_name in ['mimic_cxr', 'iu_xray']
    assert tokenizer_type in ['case', 'uncased']
    if is_same_tokenizer:
        data_name = 'mimic_cxr'
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_path = os.path.join(tokenizer_dir, f'{data_name}_{model}_{tokenizer_type}_tokenizer.json')
    if not os.path.exists(tokenizer_path):
        # check corpus
        corpus_path = os.path.join(tokenizer_dir, f"{data_name}_train_{tokenizer_type}_corpus.txt")
        if not os.path.exists(corpus_path):
            get_vocab_file(ann_path, data_name, tokenizer_type, corpus_path)
        train_tokenizer(corpus_path, tokenizer_path, model)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.add_special_tokens(['[BOS]', '[EOS]'])
    return tokenizer
