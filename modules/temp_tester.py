import os
import cv2
import numpy as np
import spacy
import torch
import json
from PIL import Image
import pandas as pd
from tqdm import tqdm
from abc import abstractmethod, ABC
from modules.utils import generate_heatmap
from modules.metrics.metrics import compute_all_scores
from knowledge_encoder.factual_serialization import RadGraphNER
import subprocess


class BaseTester(object):
    def __init__(self, model, metric_ftns, args, task, logger):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args['n_gpu'])
        self.model = model.to(self.device)
        self.task = task
        self.logger = logger
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.metric_ftns = metric_ftns

        self.epochs = self.args['epochs']

        self.checkpoint_dir = os.path.join(args['result_dir'], 'checkpoint')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if args['load'] is not None and args['load'] != '':
            self._load_checkpoint(args['load'])

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):

        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)


class Tester(BaseTester):
    def __init__(self, model, metric_ftns, args, test_dataloader, logger, task, runner):
        super(Tester, self).__init__(model, metric_ftns, args, task, logger)
        self.test_dataloader = test_dataloader
        self.runner = runner

    def test(self):
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, test_images_ids = [], [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, sk_ids, sk_masks) in enumerate(
                    self.test_dataloader):
                images = images.to(self.device)
                reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)
                gen_texts, gt_texts = self.model(images, reports_ids, reports_masks, sk_ids, sk_masks, mode='sample')
                test_res.extend(gen_texts)
                test_gts.extend(gt_texts)
                test_images_ids.extend(images_id)
            test_met = self.metric_ftns(gts=test_gts, res=test_res, args=self.args)
            logg_info = ''
            for k, v in test_met.items():
                logg_info += f"{k}: {v}; "
            self.logger.info(f"test metrics: {logg_info}")
            print(f"test metrics: {logg_info}")

            # save the metrics and the predict results
            temp_ids, temp_test_gts, temp_test_res = list(test_met.keys()), [None] * len(test_met), list(
                test_met.values())
            temp_ids.extend(test_images_ids)
            temp_test_gts.extend(test_gts)
            temp_test_res.extend(test_res)
            cur_test_ret = pd.DataFrame({'images_id': temp_ids, 'ground_truth': temp_test_gts,
                                         f'pred_report': temp_test_res})
            cur_test_ret['images_id'] = cur_test_ret['images_id'].apply(lambda x: x.split('_')[-1])
            test_pred_path = os.path.join(self.args['result_dir'], 'test_prediction.csv')
            cur_test_ret.to_csv(test_pred_path, index=False)

    def pred_gen_results(self):
        root = '/home/miao/data/Code/results/ablation study/plot_cases/'
        data = pd.read_excel(os.path.join(root, 'FSE-plot_cases.xlsx'))
        image_id_list = data['images_id'].tolist()
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, test_images_ids = [], [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, sk_ids, sk_masks) in enumerate(
                    self.test_dataloader):
                curr_image_id = images_id[0].split('_')[-1]
                if curr_image_id not in image_id_list:
                    continue
                images = images.to(self.device)
                reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)
                gen_texts, gt_texts = self.model(images, reports_ids, reports_masks, sk_ids, sk_masks, mode='sample')
                test_res.extend(gen_texts)
                test_gts.extend(gt_texts)
                test_images_ids.extend(images_id)
            test_met = self.metric_ftns(gts=test_gts, res=test_res, args=self.args)
            logg_info = ''
            for k, v in test_met.items():
                logg_info += f"{k}: {v}; "
            self.logger.info(f"test metrics: {logg_info}")
            print(f"test metrics: {logg_info}")

            # save the metrics and the predict results
            cur_test_ret = pd.DataFrame({'images_id': test_images_ids, 'ground_truth': test_gts,
                                         f'pred_report': test_res})
            cur_test_ret['images_id'] = cur_test_ret['images_id'].apply(lambda x: x.split('_')[-1])
            test_pred_path = os.path.join(root, 'test_prediction_temp.csv')
            cur_test_ret.to_csv(test_pred_path, index=False)
        print("gen result finished!")

    def extract_factual_serialization(self):
        root = '/home/miao/data/Code/results/ablation study/plot_cases/'
        test_pred_path = os.path.join(root, 'test_prediction_temp.csv')
        pred_df = pd.read_csv(test_pred_path)
        image_id_list = pred_df['images_id'].tolist()
        radgraph_path = '/home/miao/data/dataset/checkpoints/radgraph/model.tar.gz'
        corpus = {image_id: gen_text for image_id, gen_text in zip(pred_df['images_id'].tolist(), pred_df['pred_report'].tolist())}
        radgraph = RadGraphNER(corpus=corpus, is_get_output=True, is_mimic=False, model_path=radgraph_path, cuda='0')
        pred_fs = radgraph.preprocess_corpus_radgraph_output()
        gen_fs_list = [pred_fs[img_id]['core_findings'] for img_id in image_id_list]
        gen_fs_index_list = [pred_fs[img_id]['core_findings_index'] for img_id in image_id_list]
        pred_df['gen_fs'] = gen_fs_list
        pred_df['gen_fs_index'] = gen_fs_index_list
        pred_df.to_csv(os.path.join(root, 'test_prediction_wit_factual_serialization.csv'), index=False)

    def plot(self):
        root = '/home/miao/data/Code/results/ablation study/plot_cases/'
        assert self.args['batch_size'] == 1
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(root, "attentions_entities"), exist_ok=True)
        data_path = os.path.join(root, 'FSE-plot_cases.xlsx')
        data = pd.read_excel(data_path)
        # image_id_list = data['images_id'].tolist()
        image_id_list = ['c6d9dcd8-49e961d7-227e2c94-92994086-9831113b', 'b529320a-394d7b79-a3e8c3da-c28c6b94-7ec08b51',
                         '37f7e3ca-93ef1bc3-81e615c8-a061addd-3a3b6dbf', 'f2b4864c-c60e842d-258889c6-61e08bca-a7990195']
        all_metrics = ['BERTScore', 'F1-Radgraph-partial', 'chexbert_5_micro_f1',
                       'chexbert_all_micro_f1', 'BLEU_1', 'BLEU_2', 'BLEU_3',
                       'BLEU_4', 'METEOR', 'ROUGE_L']
        ann_data = json.load(open(self.args['ann_path']))
        del ann_data['train']
        del ann_data['val']
        id2image_path = {}
        for value in ann_data['test']:
            if len(value['core_findings']) == 0:
                continue
            if value['id'] not in image_id_list:
                continue
            id2image_path[value['id']] = {
                'image_path': value['image_path'][0],
                'gt_fs': value['core_findings'],
                'gt_report': value['report'],
                'similar_historical_cases': value['specific_knowledge']['sk_keywords'][:5],
            }
        del ann_data

        self.model.eval()
        final_analysis_cases = {}
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks, sk_ids, sk_masks) in enumerate(self.test_dataloader):
                image_id = images_id[0].split('_')[-1]
                if image_id not in image_id_list:
                    continue
                images = images.to(self.device)
                reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)
                gen_texts, gt_texts = self.model(images, reports_ids, reports_masks, sk_ids, sk_masks, mode='sample')
                scores = compute_all_scores(gt_texts, gen_texts, self.args)
                print(image_id, scores)
                # ori_images
                ori_image_path = id2image_path[image_id]['image_path']
                ori_image = np.array(Image.open(os.path.join(self.args['image_dir'], ori_image_path)).convert('RGB'))

                chile_dir = f'{root}/attentions_entities/{image_id}'
                os.makedirs(chile_dir, exist_ok=True)
                p = os.path.join(self.args['image_dir'], ori_image_path)
                d = f'{chile_dir}/ori.jpg'
                os.system(f'cp {p} "{d}"')

                attention_weights = [layer.src_attn.attn.cpu().numpy()[:, :, :-1] for layer in
                                     self.model.text_decoder.model.decoder.layers][0][0]
                image_id_idx = data['images_id'] == image_id
                gen_fs, gen_fs_idx = data.loc[image_id_idx, 'gen_fs'].item(), data.loc[image_id_idx, 'gen_fs_idx'].item()
                gen_fs, gen_fs_idx = eval(gen_fs), eval(gen_fs_idx)
                gt_report, gen_report = data.loc[image_id_idx, 'ground_truth'].item(), data.loc[image_id_idx, 'pred_report'].item()
                item_result = {
                    'gt_report': gt_report,
                    'gen_report': gen_report,
                    'gt_fs': id2image_path[image_id]['gt_fs'],
                    'gen_fs': gen_fs,
                    'similar_historical_cases': id2image_path[image_id]['similar_historical_cases'],
                }
                for m in all_metrics:
                    item_result[m] = scores[m]

                # the heatmap for factual sequences
                gen_report_words, old_gen_report_words = gen_texts[0].split(' '), gen_report.split(' ')
                for sen_idx, (fs_idx, factual_sequence) in enumerate(zip(gen_fs_idx, gen_fs)):
                    words_fs = ' '.join(np.array(gen_report_words)[fs_idx])
                    # assert words_fs == factual_sequence
                    atten = attention_weights[:, fs_idx, :]
                    heatmap = generate_heatmap(ori_image, atten.mean(0).mean(0).squeeze())
                    image_path = f'{chile_dir}/{sen_idx}_{factual_sequence}.png'
                    cv2.imwrite(image_path, heatmap)
                final_analysis_cases[image_id] = item_result

        with open(f'{root}/FSE-plot_cases_partial.json', 'w') as outfile:
            # json.dump(final_analysis_cases, outfile, indent=2, cls=MyEncoder)
            json.dump(final_analysis_cases, outfile, indent=2)

    def ori_plot(self):
        assert self.args['batch_size'] == 1
        self.args['beam_size'] = 1
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(self.args['result_dir'], "attentions_entities"), exist_ok=True)

        self.model.eval()
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        # ner = spacy.load("en_core_sci_sm")
        mean = mean[:, None, None]
        std = std[:, None, None]
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks, sk_ids, sk_masks) in enumerate(self.test_dataloader):
                images = images.to(self.device)
                reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)
                gen_texts, gt_texts = self.model(images, reports_ids, reports_masks, sk_ids, sk_masks, mode='sample')
                scores = compute_all_scores(gt_texts, gen_texts, self.args)
                print(images_id, scores)
                gen_text_words = gen_texts[0].split(" ")
                attention_weights = [layer.src_attn.attn.cpu().numpy()[:, :, :-1] for layer in
                                     self.model.text_decoder.model.decoder.layers][0]
                print()

                # the heatmap for each word
                # image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                # for layer_idx, attns in enumerate(attention_weights[0][0]):
                #     assert len(attns) == len(gen_texts)
                #     for word_idx, (attn, word) in enumerate(zip(attns, gen_texts)):
                #         os.makedirs(os.path.join(self.args['result_dir'], "attentions", "{:04d}".format(batch_idx),
                #                                  "layer_{}".format(layer_idx)), exist_ok=True)
                #         att = att.mean(0).mean(0)
                #         heatmap = generate_heatmap(image, attn)
                #         cv2.imwrite(os.path.join(self.args['result_dir'], "attentions", "{:04d}".format(batch_idx),
                #                                  "layer_{}".format(layer_idx), "{:04d}_{}.png".format(word_idx, word)),
                #                     heatmap)



    # def plot(self):
    #     assert self.args['batch_size'] == 1 and self.args['beam_size'] == 1
    #     self.logger.info('Start to plot attention weights in the test set.')
    #     os.makedirs(os.path.join(self.args['result_dir'], "attentions"), exist_ok=True)
    #     os.makedirs(os.path.join(self.args['result_dir'], "attentions_entities"), exist_ok=True)
    #
    #     data = pd.read_excel('/home/miao/data/Code/results/ablation study/FSE-plot_cases.xlsx')
    #     image_id_list = data['images_id'].tolist()
    #     radgraph_path = '/home/miao/data/dataset/checkpoints/radgraph/model.tar.gz'
    #     all_metrics = ['BERTScore', 'F1-Radgraph-partial', 'chexbert_5_micro_f1',
    #                    'chexbert_all_micro_f1', 'BLEU_1', 'BLEU_2', 'BLEU_3',
    #                    'BLEU_4', 'METEOR', 'ROUGE_L']
    #     ann_data = json.load(open(self.args['ann_path']))
    #     del ann_data['train']
    #     del ann_data['val']
    #     id2image_path = {}
    #     for value in ann_data['test']:
    #         if len(value['core_findings']) == 0:
    #             continue
    #         if value['id'] not in image_id_list:
    #             continue
    #         id2image_path[value['id']] = {
    #             'image_path': value['image_path'][0],
    #             'gt_fs': value['core_findings'],
    #             'gt_report': value['report'],
    #             'similar_historical_cases': value['specific_knowledge']['sk_keywords'][:5],
    #         }
    #     del ann_data
    #
    #     self.model.eval()
    #     final_analysis_cases = {}
    #     with torch.no_grad():
    #         for batch_idx, (images_id, images, reports_ids, reports_masks, sk_ids, sk_masks) in enumerate(self.test_dataloader):
    #             image_id = images_id[0].split('_')[-1]
    #             if image_id not in image_id_list:
    #                 continue
    #             images = images.to(self.device)
    #             reports_ids, reports_masks = reports_ids.to(self.device), reports_masks.to(self.device)
    #             gen_texts, gt_texts = self.model(images, reports_ids, reports_masks, sk_ids, sk_masks, mode='sample')
    #             # gt_texts = [id2image_path[image_id]['gt_report']]
    #             scores = compute_all_scores(gt_texts, gen_texts, self.args)
    #             print(scores)
    #             attention_weights = [layer.src_attn.attn.cpu().numpy()[:, :, :-1] for layer in
    #                                  self.model.text_decoder.model.decoder.layers]
    #             ori_image_path = id2image_path[image_id]['image_path']
    #             ori_image = np.array(Image.open(os.path.join(self.args['image_dir'], ori_image_path)))
    #
    #             corpus = {image_id: gen_texts[0]}
    #             subprocess.run('conda deactivate base', shell=True)
    #             subprocess.run('conda activate py37', shell=True)
    #             radgraph = RadGraphNER(corpus=corpus, is_get_output=True, is_mimic=False, model_path=radgraph_path, cuda='0')
    #             pred_fs = radgraph.preprocess_corpus_radgraph_output()
    #             subprocess.run('conda deactivate py37', shell=True)
    #             subprocess.run('conda activate base', shell=True)
    #             gen_fs, gen_fs_idx = pred_fs[image_id]['core_findings'], pred_fs[image_id]['core_findings_index']
    #             item_result = {
    #                 'gt_report': gt_texts[0],
    #                 'gen_report': gen_texts[0],
    #                 'gt_fs': id2image_path[image_id]['gt_fs'],
    #                 'gen_fs': gen_fs,
    #                 'similar_historical_cases': id2image_path[image_id]['similar_historical_cases'],
    #             }
    #             for m in all_metrics:
    #                 item_result[m] = scores[m]
    #             # the heatmap for each word
    #             # for layer_idx, attns in enumerate(attention_weights):
    #             #     assert len(attns) == len(gen_texts)
    #             #     for word_idx, (attn, word) in enumerate(zip(attns, gen_texts)):
    #             #         os.makedirs(os.path.join(self.args['result_dir'], "attentions", "{:04d}".format(batch_idx),
    #             #                                  "layer_{}".format(layer_idx)), exist_ok=True)
    #             #         att = att.mean(0).mean(0)
    #             #         heatmap = generate_heatmap(image, attn)
    #             #         cv2.imwrite(os.path.join(self.args['result_dir'], "attentions", "{:04d}".format(batch_idx),
    #             #                                  "layer_{}".format(layer_idx), "{:04d}_{}.png".format(word_idx, word)),
    #             #                     heatmap)
    #             # the heatmap for each entity
    #             # images_id = images_id[0].split("_")[-1]
    #             # for ne_idx, ne in enumerate(ner(" ".join(gen_texts)).ents):
    #             #     for layer_idx in range(len(attention_weights[0])):
    #             #         chile_dir = f'{self.args["result_dir"]}/attentions_entities/{images_id}/layer_{layer_idx}'
    #             #         os.makedirs(chile_dir, exist_ok=True)
    #             #         attn = attention_weights[layer_idx][:, :, char2word[ne.start_char]:char2word[ne.end_char] + 1, :]
    #             #         heatmap = generate_heatmap(image, attn.mean(1).mean(1).squeeze())
    #             #         image_path = f'{chile_dir}/{ne_idx}_{ne}.png'
    #             #         cv2.imwrite(image_path, heatmap)
    #
    #             # the heatmap for factual sequences
    #             gen_report_words = gen_texts[0].split(' ')
    #             for sen_idx, (fs_idx, factual_sequence) in enumerate(zip(gen_fs_idx, gen_fs)):
    #                 words_fs = gen_report_words[fs_idx]
    #                 for layer_idx in range(len(attention_weights[0])):
    #                     chile_dir = f'{self.args["result_dir"]}/attentions_entities/{images_id}/layer_{layer_idx}'
    #                     os.makedirs(chile_dir, exist_ok=True)
    #                     attn = attention_weights[layer_idx][:, :, fs_idx, :]
    #                     # attn = attention_weights[layer_idx][:, :, char2word[ne.start_char]:char2word[ne.end_char] + 1,
    #                     #        :]
    #                     heatmap = generate_heatmap(ori_image, attn.mean(1).mean(1).squeeze())
    #                     image_path = f'{chile_dir}/{sen_idx}_{factual_sequence}.png'
    #                     cv2.imwrite(image_path, heatmap)
    #             final_analysis_cases[image_id] = item_result
    #     with open('/home/miao/data/Code/results/ablation study/FSE-plot_cases.json', 'w') as outfile:
    #         # json.dump(final_analysis_cases, outfile, indent=2, cls=MyEncoder)
    #         json.dump(final_analysis_cases, outfile, indent=2)
