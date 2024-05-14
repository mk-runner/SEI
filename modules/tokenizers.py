import json
import re
from collections import Counter


class Tokenizer(object):
    def __init__(self, args):
        if args['finetune_data_name'] == 'mimic_cxr':
            self.threshold = args['mimic_threshold']
            self.clean_report = self.clean_report_mimic_cxr
            ann_path = args['mimic_ann_path']
        elif args['finetune_data_name'] == 'iu_xray':
            self.threshold = args['iu_threshold']
            self.clean_report = self.clean_report_iu_xray
            ann_path = args['iu_ann_path']
        else:
            raise ValueError(f"this dataset {args['finetune_data_name']} is not support!")
        self.ann = json.loads(open(ann_path, 'r').read())['train']
        self.pad_token_id = args['pad_idx']
        self.eos_token_id = args['eos_idx']
        self.bos_token_id = args['eos_idx']
        self.cls_token_id = 0
        self.token2idx, self.idx2token = self.create_vocabulary()

    def __len__(self):
        return len(self.idx2token)

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        token2idx['[special_token]'] = 0
        idx2token[0] = '[special_token]'
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            elif i != 0:
                break
        return txt

    def decode_special_token(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if i >= 1:
                txt += ' '
            txt += self.idx2token[idx]
        return txt

    def batch_decode(self, ids_batch, skip_special_tokens=True):
        out = []
        if skip_special_tokens:
            for ids in ids_batch:
                out.append(self.decode(ids))
        else:
            for ids in ids_batch:
                out.append(self.decode_special_token(ids))
        return out


class MixTokenizer(object):
    def __init__(self, args):
        self.threshold = {'iu_xray': args['iu_threshold'],
                          'mimic_cxr': args['mimic_threshold']}
        self.ann = {'iu_xray': json.loads(open(args['iu_ann_path'], 'r').read())['train'],
                    'mimic_cxr': json.loads(open(args['mimic_ann_path'], 'r').read())['train']}

        self.pad_token_id = args['pad_idx']
        self.eos_token_id = args['eos_idx']
        self.bos_token_id = args['eos_idx']
        self.cls_token_id = 0
        self.token2idx, self.idx2token = self.create_vocabulary()

    def __len__(self):
        return len(self.idx2token)

    def create_vocabulary(self):
        total_tokens_iu_xray = []
        total_tokens_mimic_cxr = []

        for example in self.ann['iu_xray']:
            tokens = self.clean_report_iu_xray(example['report']).split()
            for token in tokens:
                total_tokens_iu_xray.append(token)

        for example in self.ann['mimic_cxr']:
            tokens = self.clean_report_mimic_cxr(example['report']).split()
            for token in tokens:
                total_tokens_mimic_cxr.append(token)

        counter_iu_xray = Counter(total_tokens_iu_xray)
        counter_mimic_cxr = Counter(total_tokens_mimic_cxr)
        # counter vocab which more than [threshold] times appear
        vocab = [k for k, v in counter_iu_xray.items() if v >= self.threshold['iu_xray']] + ['<unk>']
        vocab += [k for k, v in counter_mimic_cxr.items() if v >= self.threshold['mimic_cxr'] and k not in vocab]
        vocab.sort()
        token2idx, idx2token = {}, {}
        token2idx['[special_token]'] = 0  # should consider that whether the token is necessary.
        idx2token[0] = '[special_token]'
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report, dataset='iu_xray'):
        if dataset == 'iu_xray':
            tokens = self.clean_report_iu_xray(report).split()
        else:
            tokens = self.clean_report_mimic_cxr(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            elif i != 0:
                break
        return txt

    def decode_special_token(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if i >= 1:
                txt += ' '
            txt += self.idx2token[idx]
        return txt

    def batch_decode(self, ids_batch, skip_special_tokens=True):
        out = []
        if skip_special_tokens:
            for ids in ids_batch:
                out.append(self.decode(ids))
        else:
            for ids in ids_batch:
                out.append(self.decode_special_token(ids))
        return out
