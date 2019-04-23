
from collections import defaultdict


MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)

"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
             precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab:    for i=numel(mpre)-1:-1:1
                                mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab:    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def calc_gt_num(gts):
    gt_num = defaultdict(int)
    for cls in gts:
        for file_id in gts[cls]:
            for obj in gts[cls][file_id]:

                if not obj['difficult']:
                    gt_num[cls] += 1
    return gt_num

def set_tags(gts, tag, val):
    for cls in gts:
        for file_id in gts[cls]:
            for obj in gts[cls][file_id]:
                obj[tag] = val

def del_tags(gts, tag):
    for cls in gts:
        for file_id in gts[cls]:
            for obj in gts[cls][file_id]:
                del obj[tag]

def eval_mAP(groundtruths, predictions, specific_iou_classes=None):
    gt_classes = sorted(groundtruths.keys())
    n_classes = len(gt_classes)
    sum_AP = 0.0
    ap_dictionary = {}
    if specific_iou_classes is None:
        specific_iou_classes = [] 
    gt_counter_per_class = calc_gt_num(groundtruths)
    set_tags(groundtruths, 'used', False)
    for class_name in gt_classes:
        # assert class_name in predictions
        predictions_data = predictions[class_name]
        gts_data = groundtruths[class_name]

        nd = len(predictions_data)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd

        min_overlap = MINOVERLAP
        if class_name in specific_iou_classes:
            assert type(specific_iou_classes[class_name]) is float
            min_overlap = specific_iou_classes[class_name]

        for idx, prediction in enumerate(predictions_data):
            # assign prediction to ground truth object if any
            #     open ground-truth with that file_id
            file_id = prediction["file_id"]
            # gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
            # ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            # load prediction bounding-box
            hbb = prediction["hbbox"]
            obb = prediction["obbox"]
            for obj in gts_data[file_id]:
                # look for a class_name match
                # if obj["class_name"] == class_name:
                hbbgt = obj["hbbox"]
                obbgt = obj["obbox"]
                hbi = [max(hbb[0],hbbgt[0]), max(hbb[1],hbbgt[1]), min(hbb[2],hbbgt[2]), min(hbb[3],hbbgt[3])]
                obi = [max(obb[0], obbgt[0]), max(obb[1], obbgt[1]), min(obb[2], obbgt[2]), min(obb[3], obbgt[3])]
                hiw = hbi[2] - hbi[0] + 1
                hih = hbi[3] - hbi[1] + 1
                oiw = obi[2] - obi[0] + 1
                oih = obi[3] - obi[1] + 1
                if hiw > 0 and hih > 0 and oiw >0 and oih>0:
                    # compute overlap (IoU) = area of intersection / area of union
                    hua = (hbb[2] - hbb[0] + 1) * (hbb[3] - hbb[1] + 1) + (hbbgt[2] - hbbgt[0] + 1) * (hbbgt[3] - hbbgt[1] + 1) - hiw * hih
                    hov = hiw * hih / hua

                    oua = (obb[2] - obb[0] + 1) * (obb[3] - obb[1] + 1) + (obbgt[2] - obbgt[0] + 1) * (
                                obbgt[3] - obbgt[1] + 1) - oiw * oih
                    oov = oiw * oih / oua
                    ohmax=min(hov,oov)
                   
                    if ohmax> ovmax :
                        ovmax = ohmax
                        gt_match = obj

            if ovmax >= min_overlap:
                if not gt_match['difficult']:
                    if not gt_match["used"]:
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
                else:
                    # Ignore difficult instances
                    pass 
            else:
                # false positive
                fp[idx] = 1

        #print(tp)
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        #print(tp)
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        #print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        #print(prec)

        ap, mrec, mprec = voc_ap(rec, prec)
        sum_AP += ap
        ap_dictionary[class_name] = (ap, rec, prec)

    mAP = sum_AP / n_classes
    del_tags(groundtruths, 'used')
    return mAP, ap_dictionary


