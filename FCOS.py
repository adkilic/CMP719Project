# MMDetection 
!git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection
!pip install -q -r requirements/build.txt
!pip install -q -e .

!pip install -U openmim
!mim install 'mmengine==0.9.0'
!mim install 'mmcv==2.1.0'
!mim install 'mmdet==3.1.0'

from mmdet.evaluation.functional import eval_recalls
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.preprocessing import label_binarize

from google.colab import drive
drive.mount('/content/drive')


from mmengine.config import Config

# Load FCOS from mmdetection
cfg = Config.fromfile('/content/mmdetection/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py')

# 2. Sınıf isimleri ve dataset tipi
cfg.dataset_type = 'CocoDataset'
class_names = [
    'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
    'tricycle', 'awning-tricycle', 'bus', 'motor'
]
cfg.metainfo = dict(classes=class_names)

# Constract pipeline
common_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

#  Train Dataloader
cfg.train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        ann_file='/content/drive/MyDrive/CV-Proje/coco_data/annotations/train.json',
        data_prefix=dict(img='/content/drive/MyDrive/CV-Proje/visdrone_yolo/train/images'),
        metainfo=cfg.metainfo,
        pipeline=common_pipeline
    )
)

#  Val Dataloader
cfg.val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        ann_file='/content/drive/MyDrive/CV-Proje/coco_data/annotations/val.json',
        data_prefix=dict(img='/content/drive/MyDrive/CV-Proje/visdrone_yolo/val/images'),
        metainfo=cfg.metainfo,
        pipeline=common_pipeline
    )
)

#  Test Dataloader
cfg.test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        ann_file='/content/drive/MyDrive/CV-Proje/coco_data/annotations/test.json',
        data_prefix=dict(img='/content/drive/MyDrive/CV-Proje/visdrone_yolo/test/images'),
        metainfo=cfg.metainfo,
        pipeline=common_pipeline
    )
)

# Evaluators
cfg.val_evaluator = dict(
    type='CocoMetric',
    ann_file='/content/drive/MyDrive/CV-Proje/coco_data/annotations/val.json',
    metric='bbox',
    format_only=False
)

cfg.test_evaluator = dict(
    type='CocoMetric',
    ann_file='/content/drive/MyDrive/CV-Proje/coco_data/annotations/test.json',
    metric='bbox',
    format_only=False
)

# number of classes for visdrone it is 10
cfg.model.bbox_head.num_classes = len(class_names)

# working directory set
cfg.work_dir = './work_dirs/fcos_visdrone'
cfg.train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=1)
cfg.default_hooks.logger.interval = 10
cfg.default_hooks.checkpoint.interval = 1

# start training with FCOS
from mmengine.runner import Runner
runner = Runner.from_cfg(cfg)
runner.train()

#start testing
eval_results = runner.test()

with open('./work_dirs/fcos_visdrone/visdrone_eval_results.json', 'r') as f:
    results = json.load(f)



# ground truth and prediction 
gt_labels = []
pred_labels = []

for det in results['annotations']:
    gt_labels.append(det['category_id'])

for det in results['detections']:
    pred_labels.append(det['category_id'])

# confusion matrix
cm = confusion_matrix(gt_labels, pred_labels, labels=range(len(class_names)))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

