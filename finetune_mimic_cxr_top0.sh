CUDA_VISIBLE_DEVICES=1 python main_v0201.py \
--task finetune \
--data_name mimic_cxr \
--mimic_cxr_ann_path "knowledge_encoder/mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_with_fs_v0227.json" \
--ft_monitor_metric RCB \
--version ft_100_top0 \
--max_seq_len 100 \
--epochs 50 \
--load "SEI-1-pretrain-model-best.pth" \
--freeze_text_encoder \
--sk_type keywords \
--is_add_indication \
--lr 5.0e-5 \
--sk_topk 0 \
--optim RAdam \
--is_save_checkpoint \
--batch_size 64
