import copy
import datetime
import os
import json
import bisect
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from spacy.tokens import Span
from stanza import Pipeline

"""
===================environmental setting=================
# Basic Setup (One time activity)

# 1. Clone the DYGIE++ repository from: https://github.com/dwadden/dygiepp. This repositiory is managed by Wadden et al., authors of the paper Entity, Relation, and Event Extraction with Contextualized Span Representations (https://www.aclweb.org/anthology/D19-1585.pdf).

# git clone https://github.com/dwadden/dygiepp.git

# 2. Navigate to the root of repo in your system and use the following commands to setup the conda environment:

# conda create --name dygiepp python=3.7
# conda activate dygiepp
# cd dygiepp
# pip install -r requirements.txt
# conda develop .   # Adds DyGIE to your PYTHONPATH

# Running Inference on Radiology Reports

# 3. Activate the conda environment:

# conda activate dygiepp

"""


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            print("MyEncoder-datetime.datetime")
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Span):
            return str(obj)
        else:
            return super(MyEncoder, self).default(obj)


class RadGraphNER:
    # change the data architecture
    def __init__(self, corpus=None, ann_path=None, model_path=None, cuda='1', is_get_output=True, is_mimic=False):
        """

        Args:
            corpus: dict: {id1: s1, id2: s2, id3: s3, ...}. if corpus is None, temp_dygie_output.json should be at current path
            model_path: the official checkpoint for radgraph
            cuda: the id for gpu
            is_get_input: Whether to convert to the format processed by radgraph
        """
        self.model_path = model_path
        self.cuda = cuda
        # user defined
        self.input_path = "/home/miao/data/Code/MSC-V1212-ablation-study/knowledge_encoder/temp_dygie_input.json"
        self.output_path = '/home/miao/data/Code/MSC-V1212-ablation-study/knowledge_encoder/temp_dygie_output.json'
        if is_get_output:
            if is_mimic:
                self.get_mimic_temp_dygie_input(ann_path)
            else:
                self.get_corpus_temp_dygie_input(corpus)
            # extract entities and relationships using RadGraph
            self.extract_triplets()

    def get_mimic_temp_dygie_input(self, ann_path):
        # note that only the training corpus can be used.
        ann = json.load(open(ann_path))
        print("initialization the input data")
        del ann['val']
        del ann['test']
        with open(self.input_path, encoding='utf8', mode='w') as f:
            for split, value in ann.items():
                print(f"preprocessing the {split} data...")
                subject_study = []
                for item in tqdm(value):
                    subject, study = str(item['subject_id']), str(item['study_id'])
                    cur_subject_study = subject + '_' + study
                    if cur_subject_study not in subject_study:
                        subject_study.append(cur_subject_study)
                        sen = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ',
                                     item['report'])
                        input_item = {
                            'doc_key': cur_subject_study,
                            "sentences": [sen.strip().split()]
                        }
                        f.write(json.dumps(input_item, cls=MyEncoder))
                        f.write('\n')

    def get_corpus_temp_dygie_input(self, corpus):
        # note that only the training corpus can be used.
        with open(self.input_path, encoding='utf8', mode='w') as f:
            for item_id, value in corpus.items():
                input_item = {
                    'doc_key': item_id,
                    "sentences": [value.strip().split()]
                }
                f.write(json.dumps(input_item, cls=MyEncoder))
                f.write('\n')

    def extract_triplets(self):
        print("extract output files using radgraph.")
        os.system(f"allennlp predict {self.model_path} {self.input_path} \
                    --predictor dygie --include-package dygie \
                    --use-dataset-reader \
                    --output-file {self.output_path} \
                    --silent")

    def preprocess_mimic_radgraph_output(self):
        # =====================build the serialized string=============#
        final_dict = {}
        useless_findings = useless_core_findings_new()
        # negative_list = ['no ', 'not ', 'free of', 'negative', 'without', 'clear of']  # delete unremarkable
        print("obtain the triples for each report.")
        with open(self.output_path, 'r') as f:
            for line in tqdm(f):
                data_item = json.loads(line)
                n = data_item['predicted_ner'][0]
                # r = data_item['predicted_relations'][0]
                s = data_item['sentences'][0]
                if len(n) == 0:
                    # print(len(n), " ".join(s))
                    continue

                doc_key = data_item['doc_key']
                # if doc_key in ['10000935_50578979', '10116310_50782200', '17277688_55816958', ]:
                #     print()
                n = preprocessing_entities(n, s, doc_key)
                dict_entity = {'text': ' '.join(s)}

                # ====Remove some useless entities and relationships====
                # initialized the variables
                dot_index = [index for index, token in enumerate(s) if token in ['.', '?', '!']]
                if len(dot_index) != 0:
                    if dot_index[0] != 0:
                        dot_index = [0, *dot_index]
                    if dot_index[-1] != len(s) - 1:
                        dot_index = [*dot_index, len(s)]
                    else:
                        dot_index[-1] += 1
                else:
                    dot_index = [0, len(s)]  # the last index + 1
                core_findings = []
                dot_s_idx, dot_e_idx, pre_sen_idx = -1, -1, -1
                cur_core_findings, previous_node_modified = [], False
                # cur_core_findings: words of each sentence
                # core_findings: finding of each sentence
                for idx, ent_item in enumerate(n):
                    start_idx, end_idx, ent_label = ent_item[0], ent_item[1], ent_item[2].strip()
                    cur_ent = " ".join(s[start_idx:end_idx + 1]).strip('"').strip("'").strip()
                    # delete unrelated entities
                    if cur_ent in list(',:;!()*&-_?'):
                        continue

                    sen_idx = bisect.bisect_left(dot_index, start_idx)
                    if sen_idx != pre_sen_idx:
                        if len(cur_core_findings) != 0:
                            if len(cur_core_findings) == 1 and cur_core_findings[0] in useless_findings:
                                pass
                            else:
                                core_findings.append(' '.join(cur_core_findings))
                        cur_core_findings, previous_node_modified = [], False
                        if start_idx == dot_index[sen_idx]:
                            dot_s_idx = dot_index[sen_idx]
                            dot_e_idx = dot_index[sen_idx] + 1 if sen_idx == len(dot_index) - 1 else dot_index[
                                sen_idx + 1]
                            pre_sen_idx = sen_idx + 1
                        else:
                            dot_s_idx = dot_index[sen_idx - 1]
                            dot_e_idx = dot_index[sen_idx]
                            pre_sen_idx = sen_idx

                    if start_idx <= dot_e_idx < end_idx:
                        print(doc_key, "error!!", cur_ent)
                        cur_ent = cur_ent.split('.')[0].strip()
                    if "DA" in ent_label and not previous_node_modified:
                        cur_core_findings = ['no', *cur_core_findings]
                        previous_node_modified = True
                    elif "U" in ent_label and not previous_node_modified:
                        cur_core_findings = ["maybe", *cur_core_findings]
                        previous_node_modified = True
                    cur_core_findings.append(cur_ent)  # add word

                if len(cur_core_findings) != 0:
                    if len(cur_core_findings) == 1 and cur_core_findings[0] in useless_findings:
                        pass
                    else:
                        core_findings.append(' '.join(cur_core_findings))
                dict_entity['core_findings'] = core_findings
                final_dict.update({
                    doc_key: dict_entity
                })

        # =================== save files ======================#
        # with open(ent_path, 'w') as outfile:
        #     json.dump(final_dict, outfile, indent=2)
        return final_dict

    def preprocess_corpus_radgraph_output(self):
        # =====================build the serialized string=============#
        final_dict = {}
        useless_findings = useless_core_findings_new()
        print("obtain the triples for each report.")
        with open(self.output_path, 'r') as f:
            for line in tqdm(f):
                data_item = json.loads(line)
                n = data_item['predicted_ner'][0]
                # r = data_item['predicted_relations'][0]
                s = data_item['sentences'][0]
                if len(n) == 0:
                    continue

                doc_key = data_item['doc_key']
                # if doc_key in ['10000935_50578979', '10116310_50782200', '17277688_55816958', ]:
                #     print()
                n = preprocessing_entities(n, s, doc_key)
                dict_entity = {'text': ' '.join(s)}

                # ====Remove some useless entities and relationships====
                # initialized the variables
                dot_index = [index for index, token in enumerate(s) if token in ['.', '?', '!']]
                if len(dot_index) != 0:
                    if dot_index[0] != 0:
                        dot_index = [0, *dot_index]
                    if dot_index[-1] != len(s) - 1:
                        dot_index = [*dot_index, len(s)]
                    else:
                        dot_index[-1] += 1
                else:
                    dot_index = [0, len(s)]  # the last index + 1
                core_findings = []
                core_findings_index = []
                cur_ent_index_list = []
                dot_s_idx, dot_e_idx, pre_sen_idx = -1, -1, -1
                cur_core_findings, previous_node_modified = [], False
                # cur_core_findings: words of each sentence
                # core_findings: finding of each sentence
                for idx, ent_item in enumerate(n):
                    start_idx, end_idx, ent_label = ent_item[0], ent_item[1], ent_item[2].strip()
                    cur_ent = " ".join(s[start_idx:end_idx + 1]).strip('"').strip("'").strip()
                    # delete unrelated entities
                    if cur_ent in list(',:;!()*&-_?'):
                        continue
                    cur_ent_index = list(range(start_idx, end_idx + 1))
                    sen_idx = bisect.bisect_left(dot_index, start_idx)
                    if sen_idx != pre_sen_idx:
                        if len(cur_core_findings) != 0:
                            if len(cur_core_findings) == 1 and cur_core_findings[0] in useless_findings:
                                pass
                            else:
                                core_findings.append(' '.join(cur_core_findings))
                                core_findings_index.append(cur_ent_index_list)
                        cur_core_findings, previous_node_modified = [], False
                        cur_ent_index_list = []
                        if start_idx == dot_index[sen_idx]:
                            dot_s_idx = dot_index[sen_idx]
                            dot_e_idx = dot_index[sen_idx] + 1 if sen_idx == len(dot_index) - 1 else dot_index[
                                sen_idx + 1]
                            pre_sen_idx = sen_idx + 1
                        else:
                            dot_s_idx = dot_index[sen_idx - 1]
                            dot_e_idx = dot_index[sen_idx]
                            pre_sen_idx = sen_idx

                    if start_idx <= dot_e_idx < end_idx:
                        print(doc_key, "error!!", cur_ent)
                        temp = cur_ent.split('.')[0].strip()
                        _idx = cur_ent.find(temp)
                        cur_ent_index = cur_ent_index[_idx: (_idx + len(temp))]
                        cur_ent = temp
                    if "DA" in ent_label and not previous_node_modified:
                        cur_core_findings = ['no', *cur_core_findings]
                        previous_node_modified = True
                    elif "U" in ent_label and not previous_node_modified:
                        cur_core_findings = ["maybe", *cur_core_findings]
                        previous_node_modified = True
                    cur_core_findings.append(cur_ent)  # add word
                    cur_ent_index_list.extend(cur_ent_index)

                if len(cur_core_findings) != 0:
                    if cur_core_findings[0] in useless_findings:
                        pass
                    else:
                        core_findings.append(' '.join(cur_core_findings))
                        core_findings_index.append(cur_ent_index_list)
                dict_entity['report'] = s
                dict_entity['core_findings'] = core_findings
                dict_entity['core_findings_index'] = core_findings_index
                final_dict.update({
                    doc_key: dict_entity
                })

        # =================== save files ======================#
        # with open(ent_path, 'w') as outfile:
        #     json.dump(final_dict, outfile, indent=2)
        return final_dict

    def preprocess_indication_corpus_radgraph_output(self):
        # =====================build the serialized string=============#
        final_dict = {}
        useless_findings = useless_core_findings_new()
        print("obtain the triples for each report.")
        with open(self.output_path, 'r') as f:
            for line in tqdm(f):
                data_item = json.loads(line)
                n = data_item['predicted_ner'][0]
                # r = data_item['predicted_relations'][0]
                s = data_item['sentences'][0]
                if len(n) == 0:
                    continue

                doc_key = data_item['doc_key']
                # if doc_key in ['10000935_50578979', '10116310_50782200', '17277688_55816958', ]:
                #     print()
                n = preprocessing_entities(n, s, doc_key)
                dict_entity = {'text': ' '.join(s)}

                # ====Remove some useless entities and relationships====
                # initialized the variables
                dot_index = [index for index, token in enumerate(s) if token in ['.', '?', '!']]
                if len(dot_index) != 0:
                    if dot_index[0] != 0:
                        dot_index = [0, *dot_index]
                    if dot_index[-1] != len(s) - 1:
                        dot_index = [*dot_index, len(s)]
                    else:
                        dot_index[-1] += 1
                else:
                    dot_index = [0, len(s)]  # the last index + 1
                core_findings = []
                core_findings_index = []
                cur_ent_index_list = []
                dot_s_idx, dot_e_idx, pre_sen_idx = -1, -1, -1
                cur_core_findings, previous_node_modified = [], False
                # cur_core_findings: words of each sentence
                # core_findings: finding of each sentence
                for idx, ent_item in enumerate(n):
                    start_idx, end_idx, ent_label = ent_item[0], ent_item[1], ent_item[2].strip()
                    cur_ent = " ".join(s[start_idx:end_idx + 1]).strip('"').strip("'").strip()
                    # delete unrelated entities
                    # if cur_ent in list(',:;!()*&-_?'):
                    #     continue
                    cur_ent = re.sub('[,:;!()*&-_?]', '', cur_ent)
                    cur_ent_index = list(range(start_idx, end_idx + 1))
                    sen_idx = bisect.bisect_left(dot_index, start_idx)
                    if sen_idx != pre_sen_idx:
                        if len(cur_core_findings) != 0:
                            if cur_core_findings[0] in useless_findings:
                                pass
                            else:
                                core_findings.append(' '.join(cur_core_findings))
                                core_findings_index.append(cur_ent_index_list)
                        cur_core_findings, previous_node_modified = [], False
                        cur_ent_index_list = []
                        if start_idx == dot_index[sen_idx]:
                            dot_s_idx = dot_index[sen_idx]
                            dot_e_idx = dot_index[sen_idx] + 1 if sen_idx == len(dot_index) - 1 else dot_index[
                                sen_idx + 1]
                            pre_sen_idx = sen_idx + 1
                        else:
                            dot_s_idx = dot_index[sen_idx - 1]
                            dot_e_idx = dot_index[sen_idx]
                            pre_sen_idx = sen_idx

                    if start_idx <= dot_e_idx < end_idx:
                        print(doc_key, "error!!", cur_ent)
                        temp = cur_ent.split('.')[0].strip()
                        _idx = cur_ent.find(temp)
                        cur_ent_index = cur_ent_index[_idx: (_idx + len(temp))]
                        cur_ent = temp
                    if "DA" in ent_label and not previous_node_modified:
                        # cur_core_findings = ['no', *cur_core_findings]
                        previous_node_modified = True
                    elif "U" in ent_label and not previous_node_modified:
                        # cur_core_findings = ["maybe", *cur_core_findings]
                        previous_node_modified = True
                    cur_core_findings.append(cur_ent)  # add word
                    cur_ent_index_list.extend(cur_ent_index)

                if len(cur_core_findings) != 0:
                    if cur_core_findings[0] in useless_findings:
                        pass
                    else:
                        core_findings.append(' '.join(cur_core_findings))
                        core_findings_index.append(cur_ent_index_list)
                dict_entity['report'] = s
                dict_entity['core_findings'] = core_findings
                dict_entity['core_findings_index'] = core_findings_index
                final_dict.update({
                    doc_key: dict_entity
                })

        # =================== save files ======================#
        # with open(ent_path, 'w') as outfile:
        #     json.dump(final_dict, outfile, indent=2)
        return final_dict


def preprocessing_entities(n, s, doc_key):
    new_n = []
    head_end_idx = -1
    for idx, item in enumerate(n, start=1):
        start_idx, end_idx, ent_label = item[0], item[1], item[2].strip()
        if start_idx > end_idx:
            continue
        elif start_idx <= head_end_idx:
            ori_s_idx, ori_e_idx = new_n[-1][0], new_n[-1][1]
            cur_best_str = ' '.join(s[ori_s_idx: (ori_e_idx + 1)])
            cur_str = ' '.join(s[start_idx: (end_idx + 1)])
            if ' .' in cur_best_str:
                if ' .' not in cur_str:
                    new_n.pop(-1)
                    new_n.append(item)
                    head_end_idx = end_idx
                    print(f"{doc_key} drop entities1: {cur_str} | {cur_best_str}")
                else:
                    print(f"{doc_key} drop entities2: {cur_best_str} | {cur_str}")
            else:
                if ' .' not in cur_str and ori_e_idx - ori_s_idx < (end_idx - start_idx):
                    new_n.pop(-1)
                    new_n.append(item)
                    head_end_idx = end_idx
                    print(f"{doc_key} drop entities3: {cur_str} | {cur_best_str}")
                else:
                    print(f"{doc_key} drop entities4: {cur_best_str} | {cur_str}")
            continue
        else:
            new_n.append(item)
            head_end_idx = end_idx
    return new_n


def useless_core_findings_new():
    result = {'It', 'it', 'otherwise', 'They', 'These', 'This'}
    return result


def get_mimic_cxr_annotations(ann_path, ent_data, file_name):
    ann_data = json.load(open(ann_path))
    new_ann_data = {}
    for split, value in ann_data.items():
        print(f"current preprocessing the {split}....")
        new_ann_data[split] = []
        for item in tqdm(value):
            try:
                doc_key = str(item['subject_id']) + '_' + str(item['study_id'])
                sample_core_finding = ent_data[doc_key]
                core_findings = sample_core_finding['core_findings']
                report = sample_core_finding['text']
            except:
                core_findings = []
                report = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ',
                                item['report'])
            sample_item = {
                'id': item['id'],
                "study_id": item['study_id'],
                'subject_id': item['subject_id'],
                "report": report,
                'image_path': item['image_path'],
                'core_findings': core_findings,
            }

            new_ann_data[split].append(sample_item)
    with open(file_name, 'w') as f:
        json.dump(new_ann_data, f, indent=2)


def get_mimic_cxr_annotations_temp(ann_path, ent_data, file_name):
    ann_data = json.load(open(ann_path))
    new_ann_data = {}
    for split, value in ann_data.items():
        print(f"current preprocessing the {split}....")
        new_ann_data[split] = []
        for item in tqdm(value):
            try:
                doc_key = str(item['subject_id']) + '_' + str(item['study_id'])
                sample_core_finding = ent_data[doc_key]
                core_findings = sample_core_finding['core_findings']
                report = sample_core_finding['text']
            except:
                core_findings = []
                report = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ',
                                item['report'])
            sample_item = {
                'id': item['id'],
                "study_id": item['study_id'],
                'subject_id': item['subject_id'],
                "report": report,
                'image_path': item['image_path'],
                'core_findings': core_findings,
            }

            new_ann_data[split].append(sample_item)
    with open(file_name, 'w') as f:
        json.dump(new_ann_data, f, indent=2)


def get_plot_cases_factual_serialization():
    root = '/home/miao/data/Code/results/ablation study/plot_cases/'
    test_pred_path = os.path.join(root, 'test_prediction_temp.csv')
    pred_df = pd.read_csv(test_pred_path)
    image_id_list = pred_df['images_id'].tolist()
    radgraph_path = '/home/miao/data/dataset/checkpoints/radgraph/model.tar.gz'
    corpus = {image_id: gen_text for image_id, gen_text in
              zip(pred_df['images_id'].tolist(), pred_df['pred_report'].tolist())}
    radgraph = RadGraphNER(corpus=corpus, is_get_output=True, is_mimic=False, model_path=radgraph_path, cuda='0')
    pred_fs = radgraph.preprocess_corpus_radgraph_output()
    gen_fs_list = [pred_fs[img_id]['core_findings'] for img_id in image_id_list]
    gen_fs_index_list = [pred_fs[img_id]['core_findings_index'] for img_id in image_id_list]
    pred_df['gen_fs'] = gen_fs_list
    pred_df['gen_fs_index'] = gen_fs_index_list
    pred_df.to_csv(os.path.join(root, 'test_prediction.csv'), index=False)


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


def extract_indication_factual_serialization_by_radgraph(radgraph_model_path, logger=None):
    ann_path = '/home/miao/data/dataset/MIMIC-CXR/mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_v0227.json'
    ann_data = json.load(open(ann_path))
    for split, value in ann_data.items():
        for item in tqdm(value):
            if len(item['indication']) == 0:
                item['indication_core_findings'] = []
                continue
            # years old historical
            indication = re.sub('[//?_,!]+', '', item['indication'])
            indication = re.sub('(?<=:)(?=\S)', ' ', indication)
            indication = re.sub('(?<=\s)\s+', '', indication)
            indication = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ', indication)
            indication = indication.lower()
            sex = ''
            if 'f' in indication or 'women' in indication or 'woman' in indication:
                sex = 'woman'
            elif 'm' in indication or 'men' in indication or 'man' in indication:
                sex = 'man'

            indication_list = indication.lower().split()
            # add "."
            if indication_list[-1] != '.':
                indication_list.append('.')

            corpus = {0: ' '.join(indication_list * 10)}
            radgraph = RadGraphNER(corpus=corpus, is_get_output=True, is_mimic=False, model_path=radgraph_model_path,
                                   cuda='0')
            indication_fs = radgraph.preprocess_corpus_radgraph_output()
            try:
                indication_core_findings_list = indication_fs[0]['core_findings']
                indication_core_findings = max(indication_core_findings_list, key=len)
                if len(sex) == 0:
                    item['indication_core_findings'] = indication_core_findings
                else:
                    item['indication_core_findings'] = sex + " " + indication_core_findings
                logger.info(f"indication: {indication}, fs: {item['indication_core_findings']}")
            except:
                item['indication_core_findings'] = []
                print(item['id'], indication, 'not core findings!!')
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                logger.info(f"not fs!!! {' '.join(indication_list)}, {item['id']}")
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    with open(
            '/home/miao/data/dataset/MIMIC-CXR/mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_fs_v0227.json',
            'w') as f:
        json.dump(ann_path, f, indent=2)


def extract_indication_factual_serialization_delete_keywords(logger=None):
    ann_path = '/home/miao/data/dataset/MIMIC-CXR/mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_v0227.json'
    ann_data = json.load(open(ann_path))
    new_ann_data = {}
    for split, value in ann_data.items():
        new_ann_data[split] = []
        for item in tqdm(value):
            new_item = copy.deepcopy(item)
            if len(item['indication']) == 0:
                new_item['indication_core_findings'] = ''
            else:
                if item['id'] in ['a5bb1dd6-32ef2b29-b27f45f5-4980a5b0-34f11cf0',
                                  'ae711ffd-03ebb7b3-cc16c95e-e6f64de7-d2bf7de4']:
                    new_item[
                        'indication'] = 'History: ___F with abd pain and pancreatitis, DKA, WBC elevation to ___, PNA? effusion?'
                # years old historical
                indication = re.sub(r'[//?_!*@]+', '', item['indication'].lower())
                indication = indication.replace('--', '')
                indication = re.sub(r'history:|-year-old|year old', '', indication)
                indication = re.sub(r'\bf\b|\bwomen\b|\bfemale\b', 'woman', indication)
                indication = re.sub(r'\bm\b|\bmen\b|\bmale\b', 'man', indication)
                indication = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ', indication)
                indication = re.sub(r'[.,]$', '', indication.strip())
                indication = re.sub(r'(?<=\s)\s+', '', indication)
                indication = indication.replace('.', ',')
                indication = indication.replace(', ,', ',')
                indication = indication.replace('man ,', 'man')
                indication = indication.replace('woman ,', 'woman')
                indication = indication.replace(': :', ':')
                indication = indication.replace(': ( )', '')
                indication = indication.replace('ped struck cxr : ptxarm : fractures',
                                                'ped struck cxr , ptxarm , fractures')

                print(indication.strip(), item['id'])
                # print(item['id'], indication, 'not core findings!!')
                # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                # logger.info(f"{indication.strip()} {item['id']}")
                # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                new_item['indication_core_findings'] = indication.strip()
            new_ann_data[split].append(new_item)

    with open('mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_with_fs_v0227.json', 'w') as f:
        json.dump(new_ann_data, f, indent=2)


def plot_attention_for_sei_extract_fs():
    candi_image_path = '/home/miao/data/Code/SEI-Results/mimic-cxr/sample_results_for samples with indication.csv'
    _df = pd.read_csv(candi_image_path)
    _df['image_id'] = _df['image_id'].apply(lambda x: x.split('_')[-1])
    candi_image_list = _df['image_id'].tolist()
    # candi_image_list = ['1d1ad085-bc04d368-4062c6ff-8388f25c-c9acb192', 'befa8b27-2bfd96b0-d50f7eda-deffa4f9-dd7e7314',
    #                     '14ff31ea-afb9a3f3-fca0fe57-1fb4e5d4-9f537945']

    pred_df = pd.read_csv('../results/mimic_cxr/finetune/ft_100_top1/test_prediction.csv')
    pred_df = pred_df.loc[12:, ['images_id', 'ground_truth', 'pred_report_23']]
    candi_ann_data, image_id_list = [], []
    for idx, row in pred_df.iterrows():
        image_id, gt, pred = row.tolist()
        image_id = image_id.split('_')[-1]
        if image_id in candi_image_list:
            candi_ann_data.append({
                'image_id': image_id,
                'ground_truth': gt,
                'gen_text': pred
            })
            image_id_list.append(image_id)
    corpus = {item['image_id']: item['gen_text'] for item in candi_ann_data}
    radgraph_path = '/home/miao/data/dataset/checkpoints/radgraph/model.tar.gz'
    radgraph = RadGraphNER(corpus=corpus, is_get_output=True, is_mimic=False, model_path=radgraph_path, cuda='0')
    pred_fs = radgraph.preprocess_corpus_radgraph_output()
    for i, image_id in enumerate(image_id_list):
        gen_fs_list = pred_fs[image_id]['core_findings']
        gen_fs_index_list = pred_fs[image_id]['core_findings_index']
        candi_ann_data[i].update({'gen_fs': gen_fs_list, 'gen_fs_index': gen_fs_index_list})

    candi_ann_df = pd.DataFrame(candi_ann_data)
    root = '/home/miao/data/Code/SEI-Results/mimic-cxr'
    candi_ann_df.to_csv(os.path.join(root, 'test_prediction_with_factual_serialization.csv'), index=False)


if __name__ == '__main__':
    # radgraph  from official checkpoint
    radgraph_model_path = '/home/miao/data/dataset/checkpoints/radgraph/model.tar.gz'

    # root = '/media/miao/data/Dataset/MIMIC-CXR'
    # ann_path = os.path.join(root, 'annotation.json')
    # sen_ann_path = 'mimic_cxr_annotation_sen.json'
    # # extract mimic-cxr factual serialization
    # radgraph = RadGraphNER(ann_path=ann_path, is_get_output=True, is_mimic=True, model_path=radgraph_model_path, cuda=1)
    # factual_serialization = radgraph.preprocess_mimic_radgraph_output()
    # get_mimic_cxr_annotations(ann_path, factual_serialization, sen_ann_path)

    # extract item_report factual serialization
    # hyps = ["patient is status post median sternotomy and cabg . the lungs are clear without focal consolidation . no pleural effusion or pneumothorax is seen . the cardiac and mediastinal silhouettes are unremarkable . no pulmonary edema is seen .",
    #         "___ year old woman with cirrhosis.",
    # ]
    # # note that too short reports cannot be extracted due to the limitation of radgraph
    # hyps = [
    #     "___ year old woman with cirrhosis . ___ year old woman with cirrhosis . ___ year old woman with cirrhosis. ___ year old woman with cirrhosis . ___ year old woman with cirrhosis .",
    #     "___F with new onset ascites // eval for infection . ___F with new onset ascites // eval for infection .",
    #     ]
    # corpus = {i: item for i, item in enumerate(hyps)}
    #
    # # radgraph
    # radgraph = RadGraphNER(corpus=corpus, is_get_output=True, is_mimic=False, model_path=radgraph_model_path, cuda='0')
    # factual_serialization = radgraph.preprocess_corpus_radgraph_output()
    # print(factual_serialization)
    # logger = SetLogger(f'extact_indication_fs.log', 'a')

    # extract_indication_factual_serialization(radgraph_model_path, logger)
    # extract_indication_factual_serialization_delete_keywords(logger)

    # for SEI plot attention for factual serialization
    # extract factual serialization for predications
    plot_attention_for_sei_extract_fs()
    # get_plot_cases_factual_serialization()
