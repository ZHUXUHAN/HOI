from vsrl_eval import VCOCOeval
import json
vsrl_annot_file='/home/priv-lab1/workspace/zxh/HOI/DATA/v-coco'+'/data/vcoco/vcoco_test.json'
coco_file='/home/priv-lab1/workspace/zxh/HOI/DATA/v-coco'+'/data/instances_vcoco_all_2014.json'
split_file='/home/priv-lab1/workspace/zxh/HOI/DATA/v-coco'+'/data/splits/vcoco_test.ids'
det_file='/home/priv-lab1/workspace/zxh/HOI/vcoco_det/detection.pkl'
vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)
vcocoeval._do_eval(det_file, ovr_thresh=0.05)