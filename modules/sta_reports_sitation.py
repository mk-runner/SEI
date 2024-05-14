import json
from tqdm import tqdm
from collections import Counter


def sta_report_len():
    path = '../knowledge_encoder/mimic_cxr_annotation_sen_best_reports_keywords_20.json'
    data = json.load(open(path))
    len_list = []
    split_len_list = {'train': [], 'val': [], 'test': []}
    for key, value in data.items():
        for item in tqdm(value):
            report = item['report']
            r_len = len(report.split())
            len_list.append(r_len)
            split_len_list[key].append(r_len)
    all_counter = Counter(len_list)
    all_counter = sorted(all_counter.items(), key=lambda x: x[0], reverse=True)
    print("whole")
    print(all_counter[:5], max(len_list))
    print("**************************************")
    for key, value in split_len_list.items():
        all_counter = Counter(value)
        all_counter = sorted(all_counter.items(), key=lambda x: x[0], reverse=True)
        print(f"{key}")
        print(all_counter[:20], max(value))
        print("**************************************")


sta_report_len()

# max_seq_len: 384, max_seq_len: 400
