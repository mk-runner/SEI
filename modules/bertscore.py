import torch
from bert_score import score
from f1chexbert import F1CheXbert
from radgraph import F1RadGraph


def compute_ce_scores(gts, res, model_type=r"D:\Code\checkpoints\distilbert-base-uncased"):
    # gts and res is list, e.g., [str1, str2]
    P, R, F1 = score(res, gts, model_type=model_type,
                     num_layers=5, batch_size=64, nthreads=4, all_layers=False, idf=False, baseline_path=None,
                     device='cuda' if torch.cuda.is_available() else 'cpu', lang='en', rescale_with_baseline=True)
    bertscore = F1.mean().cpu().item()

    f1chexbert = F1CheXbert()
    accuracy, accuracy_not_averaged, class_report, class_report_5 = f1chexbert(hyps=res, refs=gts)
    chexbertscore = class_report_5["micro avg"]["f1-score"]

    f1radgraph = F1RadGraph(reward_level="partial")
    mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=res, refs=gts)

    metrics = {
        "BERTScore": bertscore,
        "F1-CheXbert": chexbertscore,
        "F1-Radgraph": mean_reward,
    }
    return metrics



if __name__ == '__main__':
    x, y = ([
            "nothing to do lol",
            "nothing to do x",
            'there are moderate bilateral pleural effusions with overlying atelectasis,  underlying consolidation not excluded. mild prominence of the interstitial  markings suggests mild pulmonary edema. the cardiac silhouette is mildly  enlarged. the mediastinal contours are unremarkable. there is no evidence of  pneumothorax.'
        ],
        [
            'heart size is moderately enlarged. the mediastinal and hilar contours are unchanged. there is no pulmonary edema. small left pleural effusion is present. patchy opacities in the lung bases likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.',
            'heart size is mildly enlarged. the mediastinal and hilar contours are normal. there is mild pulmonary edema. moderate bilateral pleural effusions are present, left greater than right. bibasilar airspace opacities likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.',
            'heart size is mildly enlarged. the mediastinal and hilar contours are normal. there is mild pulmonary edema. moderate bilateral pleural effusions are present, left greater than right. bibasilar airspace opacities likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.'
        ]
    )

    metrics = compute_ce_scores(x, y)
    print(metrics)

