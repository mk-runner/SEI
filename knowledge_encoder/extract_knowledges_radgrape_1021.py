import copy
import datetime
import os
import json
import bisect
import pandas as pd
import logging
import argparse
import numpy as np
import spacy
from tqdm import tqdm
import networkx as nx
import pickle
import io
import re
from stanza import Pipeline
from spacy.tokens import Span
from collections import Counter, defaultdict
# from scispacy.abbreviation import AbbreviationDetector
# from scispacy.umls_linking import UmlsEntityLinker
from nltk.tokenize import sent_tokenize

from OpenKE.train_transe import train_transe


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


# extract entities using stanza
class StanzaNER:
    NER_BATCH_SIZE = 256

    def __init__(self, ann_path, output_path):
        self.output_path = output_path
        # Pipeline(download_method=None)
        config = {'tokenize_batch_size': self.NER_BATCH_SIZE, 'ner_batch_size': self.NER_BATCH_SIZE}
        self.ner = Pipeline(lang='en', package='radiology', processors={'tokenize': 'default', 'ner': 'radiology'},
                            download_method=None, **config)
        with open(ann_path, 'r') as f:
            self.reports = json.load(f)
        # self.facts = self.read_facts(args.knowledge_file)

    def extract(self):

        for split in self.reports.keys():
            for item in tqdm(self.reports[split]):
                report = item['report'].lower()

                docs = self.ner(report)  # extract entities by stanza
                # 将同一类型的实体放在一起，避免语义混乱, otherwise: UNCERTAINTY
                entities = {'ANATOMY': [], 'OBSERVATION': [], 'OBSERVATION_MODIFIER': [], 'ANATOMY_MODIFIER': []}

                for ent in docs.ents:
                    if ent.type in entities:
                        entities[ent.type].append(ent.text)

                item['entity'] = entities

        with open(self.output_path, 'w') as f:
            json.dump(self.reports, f)


class UMLSNER:
    def __init__(self):
        self.nlp = spacy.load("en_core_sci_lg")
        self.nlp.add_pipe("abbreviation_detector")
        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

    def get_sentence_entities_info(self, caption, all_cui_list):
        # record each entity information including CUI, TUI, name, and definition.
        # [{e1: ...}, {e2: ...}]
        # ents_dict = {}  # {CUI: {def: def, name: name}, CUI2: def2}
        entities_dict = {}
        sub_caption_doc = self.nlp(caption)
        cur_cui_list = []
        for entity in sub_caption_doc.ents:
            linker = self.nlp.get_pipe('scispacy_linker')
            for umls_ent in entity._.kb_ents:  # the first three values
                umls_ent_info = linker.kb.cui_to_entity[umls_ent[0]]
                # this name is canonical name. umls_ent_info[2] is aliases, list
                CUI, Name = umls_ent_info[0], umls_ent_info[1]  # 对每个entity得到其对应的一系列信息
                TUI, Definition = umls_ent_info[3][0], umls_ent_info[4]
                Definition = re.sub(r'<b>Description:</b>|<p>|</p>|//|\\|<|>|\**Description:\**|', '', str(Definition))
                Definition = re.sub('\[.*?\]', '', Definition.strip('"'))
                Name = re.sub(r'|<|>|\(finding\)', '', str(Name))

                if Definition is None or len(Definition) > 300 or Definition.lower() == 'none':
                    continue
                entities_dict.update({
                    CUI: {
                        'Name': Name.strip(),
                        'Definition': Definition.strip(),
                        'TUI': TUI,
                        'report_name': entity.text
                    }
                })
                if CUI not in all_cui_list:
                    cur_cui_list.append(CUI)
                    all_cui_list.append(CUI)

        return entities_dict, all_cui_list, cur_cui_list


class RadGraphNER:
    # change the data architecture
    def __init__(self, ann_path, model_path=None, cuda=0, is_get_input=True):
        if model_path is None:
            model_path = '/home/miao/data/dataset/checkpoints/radgraph/model.tar.gz'
        self.model_path = model_path
        self.cuda = cuda
        # note that only the training corpus can be used.
        ann = json.load(open(ann_path))
        self.input_path = "temp_dygie_input.json"
        self.output_path = 'temp_dygie_output.json'
        print("initialization the input data")
        if is_get_input:
            with open(self.input_path, encoding='utf8', mode='w') as f:
                for split, value in ann.items():
                    print(f"preprocessing the {split} data...")
                    subject_study = []
                    for item in tqdm(value):
                        subject, study = str(item['subject_id']), str(item['study_id'])
                        cur_subject_study = subject + '_' + study
                        if cur_subject_study not in subject_study:
                            subject_study.append(cur_subject_study)
                            # ?=[/,;,:,.,!?()] front add space
                            # ?<=[/,-,:,.,!?()] below add space
                            sen = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ',
                                         item['report'])
                            sen = re.sub(r'\s+', ' ', sen)
                            # sen = re.sub(r'(?<!\d)(?=[/,;:!?()])|(?<=[/,;:!?()])(?!\d)|\n', r' ', item['report'])
                            # re.sub(r'(?<=[a-zA-Z0-9])([.,:;]) (?=[a-zA-Z])|(?<=[a-zA-Z0-9])([.,:;])$', r' \1 ', text)
                            # re.sub(r'(?<=[a-zA-Z0-9])([.,:;]) (?=[a-zA-Z])|(?<=[a-zA-Z0-9])([.,:;])$', r' \1 ', text)
                            # sen = re.sub(r'Frontal and lateral views of the chest were obtained.', '',
                            #              sen.strip()).split()
                            input_item = {
                                'doc_key': cur_subject_study,
                                "sentences": [sen.strip().split()]
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

    def preprocess_triplets_sen(self, ent_path):
        # Directly extract key content using sentences and entities. Basically, one sentence corresponds
        # to one key content
        if not os.path.exists(self.output_path):
            self.extract_triplets()

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
        with open(ent_path, 'w') as outfile:
            json.dump(final_dict, outfile, indent=2)
        # return final_dict

    def preprocess_triplets_sen_punc(self, ent_path):
        # Directly extract key content using sentences and entities, while retaining symbols between
        # key content for each sentences. Basically, one sentence corresponds to one key content
        if not os.path.exists(self.output_path):
            self.extract_triplets()

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
                # other punctuation marks
                marks_index, cur_mark_index, _idx = [], [], 1
                for idx, cur_s in enumerate(s):
                    if cur_s in list(',;:'):
                        if idx < dot_index[_idx]:
                            cur_mark_index.append(idx)
                    if idx >= dot_index[_idx] or idx + 1 == len(s):
                        marks_index.append(cur_mark_index)
                        cur_mark_index = []
                        _idx += 1
                del cur_mark_index, _idx

                core_findings = []
                dot_s_idx, dot_e_idx, pre_sen_idx = -1, -1, -1
                cur_core_findings, previous_node_modified = [], False
                # other punctuation marks
                cur_marks_index, pre_mark_idx, pre_mark_value = [], -1, ''
                # cur_core_findings: words of each sentence
                # core_findings: finding of each sentence
                for idx, ent_item in enumerate(n):
                    start_idx, end_idx, ent_label = ent_item[0], ent_item[1], ent_item[2].strip()
                    cur_ent = " ".join(s[start_idx:end_idx + 1]).strip('"').strip("'").strip()
                    # delete unrelated entities
                    if cur_ent in list(',:;!()*&-_?'):
                        continue

                    sen_idx = bisect.bisect_left(dot_index, start_idx)
                    ori_sen_idx = bisect.bisect_left(dot_index[1:-1], start_idx)
                    ori_mark_idx = min(len(marks_index) - 1, ori_sen_idx)
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
                        cur_marks_index, pre_mark_idx, pre_mark_value = marks_index[ori_mark_idx], -1, ''

                    # is add punctuation marks include [",", ":", ";"]?
                    # Preserve only punctuation between multiple entities in the same sentence
                    mark_idx = bisect.bisect_left(cur_marks_index, start_idx)
                    if mark_idx != pre_mark_idx:
                        if len(cur_marks_index) == 0:
                            pass
                        else:
                            if pre_mark_idx != -1:
                                if pre_mark_value == '':
                                    raise ValueError("error, pre_mark_value is null")
                                cur_core_findings.append(pre_mark_value)
                            if mark_idx < len(cur_marks_index):
                                pre_mark_idx = mark_idx
                                pre_mark_value = s[cur_marks_index[pre_mark_idx]]
                            else:
                                cur_marks_index, pre_mark_idx, pre_mark_value = marks_index[ori_mark_idx], -1, ''

                    if 'no' in [item.lower() for item in cur_core_findings]:
                        print(f"exist two no!, {cur_core_findings}")
                        previous_node_modified = True
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


def useless_core_findings():
    result = {'down', 'surgery', 'port',
              'wire', 'board',
              'It', 'it',
              'masses', 'Body', 'substantially', 'upper',
              'fullness', 'line',
              'anterior', 'support', 'Status', 'grossly',
              'New', 'young', 'progression',
              'rightward',
              'Apices',
              'leftward', 'hardware', 'resident', 'system',
              'level',
              'substantial', 'middle',
              'Hardware', 'Suboptimal', 'small',
              'location', 'better', 'misdirected',
              'Improvement', 'suboptimal', 'external', 'Rotated', 'obliqued', 'disease', 'failure',
              'loops', 'course', 'off', 'normally', 'partially', 'inferior',
              'new', 'opposite', 'lead', 'apices', 'pericardial', 'congested',
              'aspiration', 'Overlying', 'extremely', 'midline', 'compression', 'longstanding',
              'prominence', 'monitoring', 'bilaterally', 'size',
              'Position', 'overall',
              'artifact', 'scattered',
              'wires', 'standard', 'improved', 'CoreValve', 'angled', 'increased',
              'markedly', 'medial',
              'postsurgical', 'improvement', 'status',
              'anteriorly', 'right', 'patient', 'overlying', 'feeding', 'interval',
              'tilted', 'Mild', 'Advancement',
              'Otherwise', 'less',
              'nodes', 'adjusted', 'progressed',
              'Multiple', 'healed',
              'median', 'minimally', 'blunting', 'moderately', 'increase',
              'blurring', 'rotation',
              'poorer', 'findings', 'interstitial',
              'metallic', 'habitus', 'positioning', 'pronounced',
              'Surgical', 'Accessed',
              'reading', 'placement', 'read', 'Shallow',
              'stable', 'levels', 'position', 'volumes',
              'more',
              'decrease', 'semi',
              'little',
              'Overyling', 'otherwise', 'similar', 'asymmetric', 'curve', 'decreased',
              'postoperative', 'bases', 'surveillance',
              'sited', 'medially', 'fully',
              'advancement',
              'wet', 'region', 'Large', 'Obliquity',
              'densities',
              'conspicuous',
              'ascending', 'poor',
              'low', 'Right', 'large', 'They',
              'mL', 'smaller', 'deformity', 'leaning',
              'under',
              'full',
              'deformities',
              'Left', 'stronger', 'limited',
              'reposition', 'Semisupine',
              'repositioning', 'severely',
              'limitation', 'These', 'mass', 'left',
              'R', 'indeterminate', 'rotated', 'Good', 'This',
              'subtle', 'lower', 'post', 'severity', 'congestive',
              'unaltered', 'Markedly', 'larger',
              'positioned', 'Post', }
    return result


def useless_core_findings_new():
    # result = {'down', 'surgery', 'port', 'wire', 'board', 'It', 'it', 'Body', 'upper',
    #           'fullness', 'line', 'anterior', 'support', 'Status', 'New', 'young',
    #           'rightward', 'Apices', 'leftward', 'hardware', 'resident', 'system',
    #           'level', 'Hardware', 'location', 'misdirected', 'external', 'Rotated', 'disease',
    #           'loops', 'course', 'off', 'new', 'opposite', 'lead', 'apices',
    #           'aspiration', 'midline', 'monitoring', 'bilaterally', 'size',
    #           'Position', 'overall', 'wires', 'standard', 'status',
    #           'anteriorly', 'right', 'patient', 'overlying', 'feeding', 'interval',
    #           'Otherwise', 'nodes', 'Multiple', 'rotation', 'findings',
    #           'habitus', 'positioning', 'Accessed', 'reading', 'placement', 'read',
    #           'levels', 'position', 'semi', 'otherwise', 'similar', 'curve',
    #           'postoperative', 'bases', 'surveillance', 'sited', 'medially',
    #           'region', 'They', 'mL', 'under', 'Left', 'limited', 'reposition',
    #           'repositioning', 'limitation', 'These', 'left', 'R', 'rotated', 'This',
    #           'post', 'Markedly', 'positioned', 'Post', }
    result = {'It', 'it', 'resident', 'system', 'Otherwise', 'reading', 'read',
              'otherwise', 'They', 'These', 'This'}
    return result


def get_ent_annotations_networks(ann_path, ent_path):
    ann_data = json.load(open(ann_path))
    ent_data = json.load(open(ent_path))
    new_ann_data = {}
    for split, value in ann_data.items():
        print(f"current preprocessing the {split}....")
        new_ann_data[split] = []
        for item in tqdm(value):
            item['report'] = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ',
                                    item['report'])
            new_item = copy.deepcopy(item)
            # obtain the entities
            try:
                doc_key = str(item['subject_id']) + '_' + str(item['study_id'])
                entities = ent_data[doc_key]['entities']
                serialized_ret = get_sample_radgraph_serialize(entities)
            except:
                # entities = {}
                serialized_ret = []

            new_item.update({
                'serialized_ret': serialized_ret,
                # 'entities': entities,
            })
            new_ann_data[split].append(new_item)
    with open("mimic_entities_annotation.json", 'w') as f:
        json.dump(new_ann_data, f, indent=2)


def get_core_findings_annotations(ann_path, ent_data, file_name):
    ann_data = json.load(open(ann_path))
    # ent_data = json.load(open(ent_path))
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


def build_graph(sample):
    """
    Creates a directed graph using the given sample data.

    Args:
        sample (dict): A dictionary containing "entities" and "text" keys.

    Returns:
        networkx.DiGraph: A directed graph representing the entities and relations in the sample.
    """
    G = nx.DiGraph()

    # Add nodes for each entity
    for entity_id, entity_data in sample.items():
        G.add_node(entity_id, label=entity_data["label"], tokens=entity_data["tokens"])
        # 将ID作为实体编号，不存在同一个实体存在多种label的情况，这种处理办法挺巧妙的（label和token作为它的属性）

    # Add edges for each relation
    for entity_id, entity_data in sample.items():
        for relation in entity_data["relations"]:
            relation_type, related_entity_id = relation
            G.add_edge(entity_id, related_entity_id, relation_type=relation_type)

    return G


def get_subgraph(G):
    """
    Gets all the weakly connected components in the given directed graph.

    Args:
        G (networkx.DiGraph): A directed graph.

    Returns:
        list of networkx.DiGraph: A list of subgraphs, where each subgraph represents
        a weakly connected component in the input graph.
    """
    weakly_connected_components = list(nx.weakly_connected_components(G))

    merged_graphs = []
    for component in weakly_connected_components:
        # Create a new graph with merged nodes and edges
        H = nx.DiGraph()
        for node in component:
            H.add_node(node)
        H.add_edges_from(G.subgraph(component).edges())

        # Add the new graph to the list of merged graphs
        merged_graphs.append(H)

    return merged_graphs


def serialize_subgraph(subgraph, G):
    """
    Serializes a subgraph into a string representation.

    Args:
        subgraph (networkx.DiGraph): A directed graph representing a weakly connected component.
        G (networkx.DiGraph): The original directed graph.

    Returns:
        str: A string representation of the subgraph, where nodes are sorted by their integer value
        and their "tokens" attributes are concatenated into a string.
        If a node has a "label" containing "DA", "no" is prepended to the string.
        If a node has a "label" containing "U", "maybe" is prepended to the string.
    """
    # the ordering of token is the same as report, so each nodes should be sorted.
    sorted_nodes = sorted(subgraph.nodes, key=lambda x: int(x))
    out = ""
    previous_node_modified = False  # 防止被加入多次no or maybe
    for node in sorted_nodes:
        node_label = G.nodes[node]["label"]
        node_tokens = G.nodes[node]["tokens"]
        if "DA" in node_label and not previous_node_modified:
            out = "no " + out  # prepend the no to the subgraph serialization
            previous_node_modified = True
        elif "U" in node_label and not previous_node_modified:
            out = "maybe " + out  # prepend the maybe to the subgraph serialization
            previous_node_modified = True
        out += node_tokens + " "  # 这里的损失不处理一下吗？

    return out.strip()


def get_sample_radgraph_serialize(sample_entities):
    # only preprocessing the training data
    G = build_graph(sample_entities)
    subgraph = get_subgraph(G)
    ret = []  #
    for sub_graph in subgraph:
        serialized = serialize_subgraph(sub_graph, G)
        ret.append(serialized)
    return ret


def get_entity_type(start_idx, ent, entities2type):
    # entities2type: {ent: [[type, s_idx], [type, s_idx]], ent: [[type, s_idx], [type, s_idx]]}
    if ent not in entities2type:
        return 'UNK'
    if len(entities2type[ent]) == 1:
        return entities2type[ent][0][0]
    else:
        # select the closest type
        difference = []
        for item in entities2type[ent]:
            difference.append(abs(item[1] - start_idx))
        idx = np.argmin(difference)
        return entities2type[ent][idx][0]


def get_entity(n, r, s):
    """Gets the entities for individual reports

    Args:
        n: list of entities in the report
        r: list of relations in the report
        s: list containing tokens of the sentence

    Returns:
        dict_entity: Dictionary containing the entites in the format similar to train.json

    """

    dict_entity = {}
    rel_list = [item[0:2] for item in r]
    ner_list = [item[0:2] for item in n]
    for idx, item in enumerate(n):
        temp_dict = {}
        start_idx, end_idx, label = item[0], item[1], item[2]
        temp_dict['tokens'] = " ".join(s[start_idx:end_idx + 1])
        temp_dict['label'] = label
        temp_dict['start_ix'] = start_idx
        temp_dict['end_ix'] = end_idx
        rel = []
        relation_idx = [i for i, val in enumerate(rel_list) if val == [start_idx, end_idx]]
        for i, val in enumerate(relation_idx):
            obj = r[val][2:4]
            lab = r[val][4]
            try:
                object_idx = ner_list.index(obj) + 1
            except:
                continue
            rel.append([lab, str(object_idx)])
        temp_dict['relations'] = rel
        dict_entity[str(idx + 1)] = temp_dict

    return dict_entity


def get_entity_embedding(ent_embed_name='mimic_entities_embed_1004.ckpt'):
    train_transe('knowledge', ent_embed_name)


if __name__ == '__main__':
    root = '/home/miao/data/dataset/MIMIC-CXR'
    ann_path = os.path.join(root, 'annotation.json')
    sen_path = 'mimic_cxr_core_findings_sen.json'
    sen_punc_path = 'mimic_cxr_core_findings_sen_punc.json'
    sen_ann_path = 'mimic_cxr_annotation_sen.json'
    sen_punc_ann_path = 'mimic_cxr_annotation_sen_punc.json'
    # output_path = 'ann_entity_temp.json'

    # extract triplets
    radgraph = RadGraphNER(ann_path, is_get_input=True)
    # radgraph.preprocess_triplets_sen(sen_path)
    sen_punc_data = radgraph.preprocess_triplets_sen_punc(sen_punc_path)

    # get_core_findings_annotations(ann_path, sen_path, sen_ann_path)
    get_core_findings_annotations(ann_path, sen_punc_data, sen_punc_ann_path)

    ## extract entities and relations embedding
    # get_entity_embedding()
    # del radgraph

    ## get entity annotation
    # print("obtain the annotations of entities")
    # get_ent_annotations(ann_path, ent_path)

    # "id": "4c3c1335-0fce9b11-027c582b-a0ed8d89-ca614d90",
    # "study_id": 50042142,
    # "subject_id": 10268877_50042142,
    # "report": "The ET tube is 3 . 5 cm above the carina .  The NG tube tip is off the film , at least in the stomach .  Right IJ Cordis tip is in the proximal SVC .  The heart size is moderately enlarged .  There is ill-defined vasculature and alveolar infiltrate , right greater than left .  This is markedly increased compared to the film from two hours prior and likely represents fluid overload . ",
    # "image_path": [
    #     "p10/p10268877/s50042142/4c3c1335-0fce9b11-027c582b-a0ed8d89-ca614d90.jpg"
    # ],
    # "split": "test",
    # "serialized_ret": [
    #     "ET tube 3 . 5 cm above . 5 cm above 5 cm above carina",  # problem
    #     "NG tube tip off stomach",
    #     "Right IJ Cordis tip proximal SVC",
    #     "heart size moderately enlarged",
    #     "vasculature alveolar infiltrate right greater left",
    #     "markedly increased",
    #     "maybe fluid overload"
    # ]

    # "study_id": 51513702,
    # "subject_id": 10268877,
    # "report": "Single AP portable view of the chest .  No prior .  The lungs are clear of large confluent consolidation .  Cardiac silhouette enlarged but could be accentuated by positioning and relatively low inspiratory effort .  Calcifications noted at the aortic arch .  Degenerative changes noted at the glenohumeral joints bilaterally .  Osseous and soft tissue structures otherwise unremarkable . ",
    # "image_path": [
    #     "p10/p10268877/s51513702/053e0fdd-17dbee89-17885e49-08249a30-7f829c9c.jpg"
    # ],
    # "split": "test",
    # "serialized_ret": [
    #     "no lungs clear large confluent consolidation",
    #     "Cardiac silhouette enlarged",
    #     "positioning",
    #     "low",
    #     "Calcifications aortic arch",
    #     "Degenerative changes",
    #     "glenohumeral",
    #     "joints",
    #     "bilaterally",
    #     "Osseous structures unremarkable"
    # ]
