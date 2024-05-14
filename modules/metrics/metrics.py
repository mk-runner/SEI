from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from .Radgraph import F1RadGraph
from .f1chexbert import F1CheXbert
from bert_score import score
import numpy as np
import torch


def compute_nlg_scores(gts, res, args=None):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    gts = {i: [gt] for i, gt in enumerate(gts)}
    res = {i: [re] for i, re in enumerate(res)}
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), 'CIDer'),
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


def compute_ce_scores(gts, res, args):
    # gts and res is list, e.g., [str1, str2]
    # roberta-large
    # model_type = 'distilbert-base-uncased',
    # P, R, F1 = score(res, gts, model_type=args['bertscore_checkpoint'],
    #                  num_layers=5, batch_size=64, nthreads=4, all_layers=False, idf=False, baseline_path=None,
    #                  device='cuda' if torch.cuda.is_available() else 'cpu', lang='en', rescale_with_baseline=True)
    # bertscore = F1.mean().cpu().item()

    f1chexbert = F1CheXbert(chexbert_checkpoint=args['chexbert_checkpoint'], model_checkpoint=args['chexbert_model_checkpoint'],
                            tokenizer_checkpoint=args['chexbert_tokenizer_checkpoint'])
    accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = f1chexbert(hyps=res, refs=gts)
    # default is chexbert_5_micro_f1
    # micro: each sample has the same weight; macro: each class has the same weight
    chexbert_5_micro_f1 = chexbert_5["micro avg"]["f1-score"]
    chexbert_all_micro_f1 = chexbert_all["micro avg"]["f1-score"]
    chexbert_5_macro_f1 = chexbert_5["macro avg"]["f1-score"]
    chexbert_all_macro_f1 = chexbert_all["macro avg"]["f1-score"]
    # chexbertscore = class_report_5["micro avg"]["f1-score"]

    f1radgraph_partial = F1RadGraph(reward_level='partial', model_path=args['radgraph_checkpoint'])
    partial_mean_reward, reward_list, hypothesis_ann_lists, reference_ann_lists = f1radgraph_partial(hyps=res, refs=gts)

    # f1radgraph_all = F1RadGraph(reward_level='all', model_path=args['radgraph_checkpoint'])
    # all_mean_reward, reward_list, hypothesis_ann_lists, reference_ann_lists = f1radgraph_all(hyps=res, refs=gts)
    metrics = {
        # "BERTScore": bertscore,
        "F1-Radgraph-partial": partial_mean_reward,
        # "F1-Radgraph-all": all_mean_reward,
        "chexbert_5_micro_f1": chexbert_5_micro_f1,
        "chexbert_5_macro_f1": chexbert_5_macro_f1,
        "chexbert_all_micro_f1": chexbert_all_micro_f1,
        "chexbert_all_macro_f1": chexbert_all_macro_f1,
    }
    return metrics


def compute_all_scores(gts, res, args):
    # compute clinical efficacy metrics
    ce_metrics = compute_ce_scores(gts, res, args)

    # compute natural language generation (NLG) metrics
    nlg_metrics = compute_nlg_scores(gts, res)
    ce_metrics.update(nlg_metrics)
    return ce_metrics
