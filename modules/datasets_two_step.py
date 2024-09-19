import re
import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class PretrainBaseDataset(Dataset):  # finetune and inference phase
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args['image_dir']
        self.transform = transform
        self.tokenizer = tokenizer
        ann = json.loads(open(args['ann_path'], 'r').read())
        ann = ann[split]
        self.examples = []
        self.max_seq_length = args['max_seq_len']
        if args['align_type'] == 'keywords':
            for i in range(len(ann)):
                if len(ann[i]['core_findings']) == 0:
                    continue
                core_findings = ann[i]['core_findings']
                if args['tokenizer_type'] == 'uncased':
                    core_findings = list(map(lambda x: str(x).lower(), core_findings))
                if args['data_name'] == 'mimic_cxr':
                    item_id = '_'.join([str(ann[i]['subject_id']), str(ann[i]['study_id']), ann[i]['id']])
                else:
                    item_id = ann[i]['id']
                self.examples.append({
                    # "report": report.strip(),
                    'image_path': ann[i]['image_path'],
                    'radgraph': ' [SEP] '.join(core_findings),
                    'id': item_id
                })
        else:   # report
            for i in range(len(ann)):
                if len(ann[i]['core_findings']) == 0:
                    continue
                core_findings = ann[i]['report']
                if args['tokenizer_type'] == 'uncased':
                    core_findings = core_findings.lower()
                if args['data_name'] == 'mimic_cxr':
                    item_id = '_'.join([str(ann[i]['subject_id']), str(ann[i]['study_id']), ann[i]['id']])
                else:
                    item_id = ann[i]['id']
                self.examples.append({
                    # "report": report.strip(),
                    'image_path': ann[i]['image_path'],
                    'radgraph': core_findings,
                    'id': item_id
                })
        # self.examples = self.examples[:100]

    def __len__(self):
        return len(self.examples)


class IuxrayPretrainDataset(PretrainBaseDataset):  # finetune and inference phase
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        # radgraph
        radgraph = example['radgraph']
        radgraph_ids = self.tokenizer.encode('[CLS]' + radgraph + "[SEP]").ids[: self.max_seq_length]
        radgraph_len = len(radgraph_ids)
        radgraph_masks = [1] * radgraph_len

        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        sample = (image_id, image, radgraph_ids, radgraph_masks, radgraph_len)
        return sample


class MimiccxrPretrainDataset(PretrainBaseDataset):  # finetune and inference phase
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        # radgraph
        radgraph = example['radgraph']
        radgraph_ids = self.tokenizer.encode('[CLS]' + radgraph + "[SEP]").ids[: self.max_seq_length]
        radgraph_len = len(radgraph_ids)
        radgraph_masks = [1] * radgraph_len

        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        sample = (image_id, image, radgraph_ids, radgraph_masks, radgraph_len)
        return sample


class PretrainInferenceBaseDataset(Dataset):  # finetune and inference phase
    def __init__(self, args, split, transform=None):
        self.image_dir = args['image_dir']
        self.transform = transform
        ann = json.loads(open(args['ann_path'], 'r').read())
        ann = ann[split]
        self.examples = []
        for i in range(len(ann)):
            if len(ann[i]['core_findings']) == 0:
                continue
            if args['data_name'] == 'mimic_cxr':
                item_id = '_'.join([str(ann[i]['subject_id']), str(ann[i]['study_id']), ann[i]['id']])
            else:
                item_id = ann[i]['id']
            self.examples.append({
                'image_path': ann[i]['image_path'],
                'id': item_id
            })

        # self.examples = self.examples[:100]

    def __len__(self):
        return len(self.examples)


class MimiccxrPretrainInferenceDataset(PretrainInferenceBaseDataset):  # finetune and inference phase
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        sample = (image_id, image)
        return sample


class IuxrayPretrainInferenceDataset(PretrainInferenceBaseDataset):  # finetune and inference phase
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        sample = (image_id, image)
        return sample


class FinetuneBaseDatasetNotIndication(Dataset):  # finetune and inference phase
    # indication is None and similar historical cases is not None
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args['image_dir']
        self.transform = transform
        self.tokenizer = tokenizer
        self.split = split
        self.sk_topk = args['sk_topk']
        ann = json.loads(open(args['ann_path'], 'r').read())
        ann = ann[split]
        self.examples = []
        self.max_seq_length = args['max_seq_len']
        for i in range(len(ann)):
            # remove samples that have no clinical significance according to the content of report
            if args['is_add_indication']:
                if len(ann[i]['core_findings']) == 0 or len(ann[i]['specific_knowledge']) == 0 or len(ann[i]['indication_core_findings']) != 0:
                    continue
            else:
                if len(ann[i]['core_findings']) == 0 or len(ann[i]['specific_knowledge']) == 0:
                    continue
            if args['data_name'] == 'mimic_cxr':
                item_id = '_'.join([str(ann[i]['subject_id']), str(ann[i]['study_id']), ann[i]['id']])
            else:
                item_id = ann[i]['id']
            # is add similar historical cases
            if args['sk_topk'] != 0:
                report, specific_knowledge = ann[i]['report'], ann[i]['specific_knowledge']
                if args['tokenizer_type'] == 'uncased':
                    if args['sk_type'] == 'report':
                        specific_knowledge = [sk.lower() for sk in specific_knowledge['reports'][: args['sk_topk']]]
                    else:  # sk_type is keywords
                        specific_knowledge = [' [SEP] '.join(sk).strip().lower() for sk in
                                              specific_knowledge['sk_keywords'][: args['sk_topk']]]
                    report = report.strip().lower()

                self.examples.append({
                    "report": report.strip(),
                    'image_path': ann[i]['image_path'],
                    'specific_knowledge': specific_knowledge,
                    'id': item_id,
                })
            else:  # have not sk_knowledge
                report = ann[i]['report']
                if args['tokenizer_type'] == 'uncased':
                    report = report.strip().lower()

                self.examples.append({
                    "report": report.strip(),
                    'image_path': ann[i]['image_path'],
                    'id': item_id,
                })

    def __len__(self):
        return len(self.examples)


class IuxrayFinetuneDatasetNotIndication(FinetuneBaseDatasetNotIndication):  # finetune and inference phase
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        # report
        report = example['report']
        report_ids = self.tokenizer.encode('[BOS] ' + report + " [EOS]").ids[: self.max_seq_length]
        # if self.split == 'train':
        #     report_ids = self.tokenizer.encode('[BOS] ' + report + " [EOS]").ids[: self.max_seq_length]
        # else:  # val or test
        #     report_ids = self.tokenizer.encode('[BOS] ' + report + " [EOS]").ids[: self.max_seq_length]
        report_len = len(report_ids)
        report_masks = [1] * report_len

        # specific knowledge
        specific_knowledge_ids, specific_knowledge_masks, specific_knowledge_len = [], [], []
        if self.sk_topk != 0:
            for sk in example['specific_knowledge']:
                sk_ids = self.tokenizer.encode('[BOS] ' + sk + ' [EOS]').ids
                sk_len = len(sk_ids)
                sk_masks = [1] * sk_len
                specific_knowledge_ids.append(sk_ids)
                specific_knowledge_masks.append(sk_masks)
                specific_knowledge_len.append(sk_len)

        # image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            # image = self.transform(image)
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        sample = (image_id, image, report_ids, report_masks, report_len,
                  specific_knowledge_ids, specific_knowledge_masks, specific_knowledge_len)
        return sample


class MimiccxrFinetuneDatasetNotIndication(FinetuneBaseDatasetNotIndication):  # finetune and inference phase
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        # report
        report = example['report']
        report_ids = self.tokenizer.encode('[BOS] ' + report + " [EOS]").ids[: self.max_seq_length]
        report_len = len(report_ids)
        report_masks = [1] * report_len

        # specific knowledge
        specific_knowledge_ids, specific_knowledge_masks, specific_knowledge_len = [], [], []
        if self.sk_topk != 0:
            # I think this position should not be added max_seq_len
            if self.split == 'train':
                for sk in example['specific_knowledge']:
                    sk_ids = self.tokenizer.encode('[BOS] ' + sk + ' [EOS]').ids
                    sk_len = len(sk_ids)
                    sk_masks = [1] * sk_len
                    specific_knowledge_ids.append(sk_ids)
                    specific_knowledge_masks.append(sk_masks)
                    specific_knowledge_len.append(sk_len)
            else:
                for sk in example['specific_knowledge']:
                    sk_ids = self.tokenizer.encode('[BOS] ' + sk + ' [EOS]').ids
                    sk_len = len(sk_ids)
                    sk_masks = [1] * sk_len
                    specific_knowledge_ids.append(sk_ids)
                    specific_knowledge_masks.append(sk_masks)
                    specific_knowledge_len.append(sk_len)

        # image
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        sample = (image_id, image, report_ids, report_masks, report_len,
                  specific_knowledge_ids, specific_knowledge_masks, specific_knowledge_len)
        return sample


class FinetuneBaseDatasetHasIndication(Dataset):  # finetune and inference phase
    # indication is not None and similar historical cases is not None
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args['image_dir']
        self.transform = transform
        self.tokenizer = tokenizer
        self.split = split
        self.sk_topk = args['sk_topk']
        self.is_add_indication = args['is_add_indication']
        ann = json.loads(open(args['ann_path'], 'r').read())
        ann = ann[split]
        self.examples = []
        self.max_seq_length = args['max_seq_len']
        for i in range(len(ann)):
            # remove samples that have no clinical significance according to the content of report
            if len(ann[i]['core_findings']) == 0 or len(ann[i]['specific_knowledge']) == 0 or len(ann[i]['indication_core_findings']) == 0:
                continue
            if args['data_name'] == 'mimic_cxr':
                item_id = '_'.join([str(ann[i]['subject_id']), str(ann[i]['study_id']), ann[i]['id']])
            else:
                item_id = ann[i]['id']
            # is add similar historical cases
            if args['sk_topk'] != 0:
                report, specific_knowledge = ann[i]['report'], ann[i]['specific_knowledge']
                if args['tokenizer_type'] == 'uncased':
                    if args['sk_type'] == 'report':
                        specific_knowledge = [sk.lower() for sk in specific_knowledge['reports'][: args['sk_topk']]]
                    else:  # sk_type is keywords
                        specific_knowledge = [' [SEP] '.join(sk).strip().lower() for sk in
                                              specific_knowledge['sk_keywords'][: args['sk_topk']]]
                    report = report.strip().lower()

                self.examples.append({
                    "report": report.strip(),
                    'image_path': ann[i]['image_path'],
                    'specific_knowledge': specific_knowledge,
                    'id': item_id,
                })
            else:  # have not sk_knowledge
                report = ann[i]['report']
                if args['tokenizer_type'] == 'uncased':
                    report = report.strip().lower()

                self.examples.append({
                    "report": report.strip(),
                    'image_path': ann[i]['image_path'],
                    'id': item_id,
                })

            # is add indication section
            if args['is_add_indication']:
                # default is lower
                self.examples[-1]['indication'] = ann[i]['indication_core_findings'].strip().lower()

        # self.examples = self.examples[:5]

    def __len__(self):
        return len(self.examples)


class IuxrayFinetuneDatasetHasIndication(FinetuneBaseDatasetHasIndication):  # finetune and inference phase
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        # report
        report = example['report']
        report_ids = self.tokenizer.encode('[BOS] ' + report + " [EOS]").ids[: self.max_seq_length]
        # if self.split == 'train':
        #     report_ids = self.tokenizer.encode('[BOS] ' + report + " [EOS]").ids[: self.max_seq_length]
        # else:  # val or test
        #     report_ids = self.tokenizer.encode('[BOS] ' + report + " [EOS]").ids[: self.max_seq_length]
        report_len = len(report_ids)
        report_masks = [1] * report_len

        # specific knowledge
        specific_knowledge_ids, specific_knowledge_masks, specific_knowledge_len = [], [], []
        if self.sk_topk != 0:
            for sk in example['specific_knowledge']:
                sk_ids = self.tokenizer.encode('[BOS] ' + sk + ' [EOS]').ids
                sk_len = len(sk_ids)
                sk_masks = [1] * sk_len
                specific_knowledge_ids.append(sk_ids)
                specific_knowledge_masks.append(sk_masks)
                specific_knowledge_len.append(sk_len)

        # image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            # image = self.transform(image)
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        sample = (image_id, image, report_ids, report_masks, report_len,
                  specific_knowledge_ids, specific_knowledge_masks, specific_knowledge_len)
        return sample


class MimiccxrFinetuneDatasetHasIndication(FinetuneBaseDatasetHasIndication):  # finetune and inference phase
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        # report
        report = example['report']
        report_ids = self.tokenizer.encode('[BOS] ' + report + " [EOS]").ids[: self.max_seq_length]
        report_len = len(report_ids)
        report_masks = [1] * report_len

        # specific knowledge
        specific_knowledge_ids, specific_knowledge_masks, specific_knowledge_len = [], [], []
        if self.sk_topk != 0:
            # I think this position should not be added max_seq_len
            if self.split == 'train':
                for sk in example['specific_knowledge']:
                    sk_ids = self.tokenizer.encode('[BOS] ' + sk + ' [EOS]').ids
                    sk_len = len(sk_ids)
                    sk_masks = [1] * sk_len
                    specific_knowledge_ids.append(sk_ids)
                    specific_knowledge_masks.append(sk_masks)
                    specific_knowledge_len.append(sk_len)
            else:
                for sk in example['specific_knowledge']:
                    sk_ids = self.tokenizer.encode('[BOS] ' + sk + ' [EOS]').ids
                    sk_len = len(sk_ids)
                    sk_masks = [1] * sk_len
                    specific_knowledge_ids.append(sk_ids)
                    specific_knowledge_masks.append(sk_masks)
                    specific_knowledge_len.append(sk_len)

        # indication section
        inc_ids, inc_masks, inc_len = [], [], 0
        if self.is_add_indication:
            inc_ids = self.tokenizer.encode('[BOS] ' + example['indication'] + " [EOS]").ids[: self.max_seq_length]
            inc_len = len(inc_ids)
            inc_masks = [1] * inc_len
        # image
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        sample = (image_id, image, report_ids, report_masks, report_len,
                  specific_knowledge_ids, specific_knowledge_masks, specific_knowledge_len,
                  inc_ids, inc_masks, inc_len)
        return sample


class MixSingleImageDataset(Dataset):  # pretrain phase
    def __init__(self, args, tokenizer, split, transform=None):
        # default dataset is mixture
        self.image_dir = {'iu_xray': args['iu_image_dir'],
                          'mimic_cxr': args['mimic_image_dir']}
        ann_path = {'iu_xray': args['iu_ann_path'],
                    'mimic_cxr': args['mimic_ann_path']}
        all_ann = {'iu_xray': json.loads(open(ann_path['iu_xray'], 'r').read()),
                   'mimic_cxr': json.loads(open(ann_path['mimic_cxr'], 'r').read())}
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_seq_length = args['max_seq_len']

        self.examples = []
        if args['pretrain_data_name'] in ['iu_xray', 'mix']:
            ann = all_ann['iu_xray'][split]
            for i in range(len(ann)):
                report = re.sub('Frontal and lateral views of the chest were obtained. ', '', ann[i]['report'])
                if len(report) < 4:
                    # print(f"drop this sample:{ann[i]['id']}, report: {report}")
                    continue
                self.examples.append({
                    "report": report.strip(),
                    'image_path': ann[i]['image_path'],
                    'id': ann[i]['id']
                })
        if args['pretrain_data_name'] in ['mimic_cxr', 'mix']:
            ann = all_ann['mimic_cxr'][split]
            for i in range(len(ann)):
                report = re.sub('Frontal and lateral views of the chest were obtained. ', '', ann[i]['report'])
                if len(report) < 4:
                    # print(f"drop this sample:{ann[i]['id']}, report: {report}")
                    continue
                self.examples.append({
                    "report": report.strip(),
                    'image_path': ann[i]['image_path'],
                    'id': ann[i]['id']
                })
        # self.examples = self.examples[:100]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        report = example['report']
        report_ids = self.tokenizer.encode('[CLS]' + report + "[SEP]").ids
        report_length = len(report_ids)
        report_masks = [1] * report_length

        trunc_ids = report_ids[:self.max_seq_length]
        trunc_length = len(trunc_ids)
        trunc_masks = [1] * trunc_length
        # obtain the image
        if len(image_path) > 1:
            if torch.rand(1) > 0.5:
                image = Image.open(os.path.join(self.image_dir['iu_xray'], image_path[0])).convert('RGB')
            else:
                image = Image.open(os.path.join(self.image_dir['iu_xray'], image_path[1])).convert('RGB')
        else:
            image = Image.open(os.path.join(self.image_dir['mimic_cxr'], image_path[0])).convert('RGB')
        # preprocessing the image
        if self.transform is not None:
            image = self.transform(image)
        sample = (image_id, image, report_ids, report_masks, trunc_ids, trunc_masks, report_length, trunc_length)
        return sample


class MimiccxrPretrainInferenceDatasetOne(Dataset):  # finetune and inference phase
    def __init__(self, args, split, transform=None):
        self.image_dir = args['mimic_cxr_image_dir']
        self.transform = transform
        ann = json.loads(open(args['mimic_cxr_ann_path'], 'r').read())
        ann = ann[split]
        self.examples = []
        for i in range(len(ann)):
            if len(ann[i]['core_findings']) == 0:
                continue

            item_id = '_'.join([str(ann[i]['subject_id']), str(ann[i]['study_id']), ann[i]['id']])
            self.examples.append({
                'image_path': ann[i]['image_path'],
                'id': item_id
            })

        # self.examples = self.examples[:200]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        sample = (image_id, image)
        return sample
