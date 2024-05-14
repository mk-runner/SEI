import copy
import json
import re

from section_parser import section_text
import os
import json
from copy import deepcopy
from tqdm import tqdm


def parse_report_sections(ann_path, report_dir=None):
    # extract indication, comparison, impression sections from raw reports
    if report_dir is None:
        report_dir = '/home/miao/data/dataset/MIMIC-CXR/mimic-cxr-reports/files'
    ann_data = json.load(open(ann_path))
    new_ann_data = {}
    for key, value in ann_data.items():
        new_ann_data[key] = []
        print(f"current data is {key}, the number of samples is {len(value)}!")
        for item in tqdm(value):
            # obtain all medical images for current sample
            report_path = os.path.join(report_dir, *item['image_path'][0].split('/')[:-1]) + '.txt'
            with open(report_path, encoding="utf-8") as fin:
                full_report_text = fin.read()
            # sparse the report file
            sections, section_names, _ = section_text(full_report_text)
            # obtain several sections
            new_item = deepcopy(item)
            for section_name in ['impression', 'comparison', 'indication', 'findings', 'examination']:
                if section_name not in section_names:
                    section_value = ''
                else:
                    idx = section_names.index(section_name)
                    section_value = sections[idx]
                new_item.update({section_name: section_value})

            new_ann_data[key].append(new_item)
        print(f"current data is {key}, current the number of samples is {len(new_ann_data[key])}!")

    # with open(os.path.join('/home/miao/data/dataset/MIMIC-CXR',
    #                        'mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_v0227.json'), 'w') as f:
    #     json.dump(new_ann_data, f, indent=2)
    return new_ann_data


def preprocess_indication_section(ann_data, save_ann_file_name):
    # ann_path = '/home/miao/data/dataset/MIMIC-CXR/mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_v0227.json'
    # ann_data = json.load(open(ann_path))
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

    with open(save_ann_file_name, 'w') as f:
        json.dump(new_ann_data, f, indent=2)


if __name__ == '__main__':
    # extract mimic_cxr
    # extract indication, comparison, impression sections from raw reports
    ann_path = 'mimic_cxr_annotation_sen_best_reports_keywords_20.json'
    report_dir = '/home/miao/data/dataset/MIMIC-CXR/mimic-cxr-reports/files'
    ann_data = parse_report_sections(ann_path, report_dir)
    # preprocessing the indication section using re package.
    save_ann_file_name = 'mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_with_fs_v0227.json'
    preprocess_indication_section(ann_data, save_ann_file_name)
