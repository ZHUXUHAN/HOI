import glob
from .hico_map import read_gt_files, read_pred_files, eval_mAP
import os
def hico_map(map_dir):
    """
    get a list with the ground-truth files
    get a list with the predicted files
    then to canculate map

    """
    if not os.path.exists(os.path.join(map_dir,'ground_truth')) or not os.path.exists(os.path.join(map_dir,'predicted')) :#and os.path.exists(os.path.join(map_dir,'/predicted/*.txt'))
        print("PATH NOT EXITS")


    ground_truth_files_list = glob.glob(os.path.join(map_dir,'ground_truth/*.txt'))
    ground_truth_files_list.sort()
    groundtruths = read_gt_files(ground_truth_files_list)

    predicted_files_list = glob.glob(os.path.join(map_dir,'predicted/*.txt'))
    predicted_files_list.sort()
    predictions = read_pred_files(predicted_files_list)

    mAP, _ = eval_mAP(groundtruths, predictions)
    print ("mAP: %.4f"%mAP)

if __name__ == '__main__':
    hico_map()
