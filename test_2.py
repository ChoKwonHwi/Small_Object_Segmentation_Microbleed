import numpy as np
import os
from PIL import Image
import nibabel as nib
import natsort
import matplotlib.pyplot as plt
from network_cbam import RLKunet, RLKunet2, initialize_weight
from network_cbam import RLKunet3, RLKunet4, RLKunet5
from Microbleed_dataset import Microbleed_png_dataset, Microbleed_png_dataset2
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
import torch.nn as nn
from skimage.measure import label
import torch
from utils import *
from eval import *
from scipy.spatial.distance import euclidean
import warnings

warnings.filterwarnings(action='ignore')
seed = 414
print("current seed : ", seed)
torch.manual_seed(seed)

torch.set_num_threads(2)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Helper function to calculate the center of a region (connected component)
def center(region):
    if len(region) == 0:
        return np.array([np.nan, np.nan])  # Return NaN if region is empty
    p_lst = np.array(region)
    xmin = np.min(p_lst[:,0])
    xmax = np.max(p_lst[:,0])
    ymin = np.min(p_lst[:,1])
    ymax = np.max(p_lst[:,1])
    center_coord = np.mean(region, axis=0)
    if np.any(np.isnan(center_coord)) or np.any(np.isinf(center_coord)):
        return np.array([np.nan, np.nan])  # Return NaN if invalid values
    return ((xmin+xmax)/2, (ymin+ymax)/2)#center_coord

# Helper function to get connected domain (set of connected pixels)
def getConnectedDomain(mask):
    # This is a placeholder function. Replace this with the actual function that finds connected components in the mask.
    return [np.argwhere(mask == 255)]  # Assuming '255' is the positive class pixel value.

# Main function to calculate target metrics
def calculate_metrics(pred_list, gt_list, distance_threshold=4):
    TP, FP, FN, TN = 0, 0, 0, 0

    for idx in range(len(pred_list)):
        pred = pred_list[idx].detach().cpu().numpy()
        gt = gt_list[idx].detach().cpu().numpy()

        pred_cds = getConnectedDomain(255 * pred)
        gt_cds = getConnectedDomain(255 * gt)

        # Calculate TP, FP, FN
        for pred_cd in pred_cds:
            
            detected = False
            pred_center = center(pred_cd)
            
            # Skip invalid centers
            if np.any(np.isnan(pred_center)) or np.any(np.isinf(pred_center)):
                continue

            for gt_cd in gt_cds:
                gt_center = center(gt_cd)
                
                # Skip invalid centers
                if np.any(np.isnan(gt_center)) or np.any(np.isinf(gt_center)):
                    continue
                
                # Check if centers are within the distance threshold
                if not np.any(np.isnan(pred_center)) and not np.any(np.isnan(gt_center)):
                    if euclidean(pred_center, gt_center) <= distance_threshold:
                        TP += 1
                        detected = True
                        break
            if not detected:
                FP += 1

        # Check FN (missed detections in ground truth)
        for gt_cd in gt_cds:
            missed = True
            gt_center = center(gt_cd)
            
            # Skip invalid centers
            if np.any(np.isnan(gt_center)) or np.any(np.isinf(gt_center)):
                continue
            
            for pred_cd in pred_cds:
                pred_center = center(pred_cd)
                
                # Skip invalid centers
                if np.any(np.isnan(pred_center)) or np.any(np.isinf(pred_center)):
                    continue
                
                if not np.any(np.isnan(pred_center)) and not np.any(np.isnan(gt_center)):
                    if euclidean(gt_center, pred_center) <= distance_threshold:
                        missed = False
                        break
            if missed:
                FN += 1

        # TN (True Negative) calculation
        H, W = pred.shape
        total_pixels = H * W
        pred_pixels = np.sum(pred == 1)
        gt_pixels = np.sum(gt == 1)
        TN += total_pixels - (pred_pixels + gt_pixels - TP - FP)

    # Calculate metrics
    if (TP + FN) > 0:
        sensitivity = TP / (TP + FN)  # Sensitivity (Recall)
    else:
        sensitivity = 0

    if (TN + FP) > 0:
        specificity = TN / (TN + FP)  # Specificity
    else:
        specificity = 0

    if (TP + FP) > 0:
        precision = TP / (TP + FP)  # Precision
    else:
        precision = 0

    if (TP + TN + FP + FN) > 0:
        accuracy = (TP + TN) / (TP + TN + FP + FN)  # Overall accuracy
    else:
        accuracy = 0

    return TP, FP, FN, TN, sensitivity, specificity, precision, accuracy

# 전체 코호트 테스트 - 그래프, 점수 입력 / 2코호트 그래프 입력 / 모델 구조 대충 그리기

#test_img_path = '/nasdata4/kwonhwi/RLK_UNET_13cohort/data/3slice/fold_1/test'

#test_img_path = '/nasdata4/kwonhwi/RLK_UNET_CBAM/c2_data_fold_3slice/fold_4/test'

#test_img_path = '/nasdata4/kwonhwi/RLK_UNET_13cohort/data/all_data_3slice/fold_5/test'

test_img_path = '/nasdata4/kwonhwi/yongin_preprocessing/pngfile/test_9'

test_dataset = Microbleed_png_dataset2(img_path=test_img_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = RLKunet4(in_channels=3, out_channels=2, features=64, group_num=8).to(device)
#c2_layer3 / org_layer_all / 3slice

weights_path = '/nasdata4/kwonhwi/RLK_UNET_13cohort/weights/c2_layer2_large/fold_1/Best_RLK-Unet_HF_Tar.pth' # 3c2 3c3 3c4 3c5 3s2 3s4 3sr2 3sr3 3sr5
# c2_l2_large_f1, c2_l2_large_f2, c13_3slice(l4)_f2

#weights_path = '/nasdata4/kwonhwi/RLK_UNET_13cohort/weights/c2_layer2_large/fold_1/Best_RLK-Unet_HF_loss782.pth' #3c1 3s1 3s3 3s5 3sr1 3sr4
# c13_3slice(l4)_f4, # c13_3slice(l4)_f5,c13_3slice(l4)_f1

#weights_path = '/nasdata4/kwonhwi/RLK_UNET_13cohort/weights/c2_layer2_large/fold_1/Best_RLK-Unet_HF_Pxl.pth' #3c1 3s1 3s3 3s5 3sr1 3sr4
# c2_l2_large_f3, c2_l2_large_f4, c2_l2_large_f3, c13_3slice(l4)_f3

#weights_path = '/nasdata4/kwonhwi/RLK_UNET_13cohort/weights/c2_layer2_large/fold_5/Best_RLK-Unet_HF_loss_and_f1.pth'
# c2_l2_large_f5

opt = True
img_path = "/nasdata4/kwonhwi/yongin_preprocessing/resultfile/test_9" # prediction result png 저장 경로
t2s_path = "/nasdata4/kwonhwi/yongin_preprocessing/SWI_reg/9932235_SWI2_Warped.nii.gz" # input nifti 경로
output_nifti_path = "/nasdata4/kwonhwi/yongin_preprocessing/test_nifti/9932235_SWI2_segmented.nii.gz" # prediction result nifti 경로
os.makedirs(img_path, exist_ok=True)
print(img_path)
model.load_state_dict(torch.load(weights_path))
model.eval()

dice = DiceMetric(include_background=False, reduction='mean')
dice_score_list = []
y_truth_list = []
y_pred_list = []

x_data_plt_list = []
y_data_plt_list = []
y_pred_plt_list = []

results_plt = np.zeros((56, 512, 512*3))
num = 0


# 1. t2s_path 데이터 로드 및 동일 크기의 0으로 채워진 array 생성
t2s_img = nib.load(t2s_path)
t2s_data = t2s_img.get_fdata()  # T2* 데이터
affine = t2s_img.affine  # NIfTI의 Affine 행렬
segmentation_array = np.zeros_like(t2s_data)

with torch.no_grad():
    for idx, x_data in enumerate(test_loader):
    
    #for idx, (x_data, y_data) in enumerate(test_loader):    
        x_data = x_data.to(device).float()
        #y_data = y_data.to(device).float()
        
        #y_data = torch.where(y_data > 0.0, 1.0, 0.0)
        
        _, y_pred = model(x_data)
        #_, _, y_pred = model(x_data)
        #_, _, _, y_pred = model(x_data)
        y_pred_fg = y_pred[:, 0, :, :]
        y_pred_fg = y_pred_fg.unsqueeze(1)
        y_pred_fg = torch.where(y_pred_fg > 0.5, 1.0, 0.0) # (1, 1, 224, 224)
        
        #dice_score = dice(y_pred_fg, y_data)
        
        #dice_score_list.append(dice_score.mean().item())
        y_pred_list.append(y_pred_fg[0, 0, :, :])
        #y_truth_list.append(y_data[0, 0, :, :])
        
        if opt :
            pred_np = y_pred_fg[0, 0, :, :].detach().cpu().numpy()  # 첫 번째 배치의 첫 번째 채널 예측
            segmentation_array[:,:,idx+1] = pred_np
            pred_np = (pred_np * 255).astype(np.uint8)  # 0 또는 255로 변환
            img = Image.fromarray(pred_np, mode='L')  # 'L' 모드는 1채널 그레이스케일 이미지
            
            save_path = os.path.join(img_path, f'prediction_{idx}.png')
            img.save(save_path)
    segmented_nifti = nib.Nifti1Image(segmentation_array, affine)
    nib.save(segmented_nifti, output_nifti_path)
    print(f"Saved prediction image to {save_path}")
'''
evaluator = Evaluator(y_pred_list, y_truth_list, tar_area=[0, np.inf], is_print=False)


#print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
#print(f"Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}, Precision: {precision:.3f}, Accuracy: {accuracy:.3f}")
print("-----------------------------------------------")
print(weights_path)

(Pd, Fa, TarPrec, TarRec, TarF1) = evaluator.target_metrics()
(PxlPrec, PxlRec, PxlF1) = evaluator.pixel_metrics()
(TP, FP, FN, TN, precision, sensitivity, f1_score, specificity, accuracy) = evaluator.target_metrics2()
#precision3, recall3, f13 = evaluator.target_metrics3(iou_threshold=0.0)

print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
#print(f"Precision: {precision3:.3f}, Sensitivity: {recall3:.3f}, F1-score: {f13:.3f}")
print(f"Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}, Precision: {precision:.3f}, Accuracy: {accuracy:.3f}")
print("-----------------------------------------------")


tarPrec = TarPrec
tarRec = TarRec
tarf1 = TarF1
pxlPrec = PxlPrec
pxlRec = PxlRec
pxlf1 = PxlF1

if np.isnan(tarPrec) or np.isinf(tarPrec):
    tarPrec = 0
if np.isnan(tarRec) or np.isinf(tarRec):
    tarRec = 0
if np.isnan(tarf1) or np.isinf(tarf1):
    tarf1 = 0
if np.isnan(pxlPrec) or np.isinf(pxlPrec):
    pxlPrec = 0
if np.isnan(pxlRec) or np.isinf(pxlRec):
    pxlRec = 0
if np.isnan(pxlf1) or np.isinf(pxlf1):
    pxlf1 = 0

print("TarPrec : {:.3f}, TarRec : {:.3f}, and TarF1 : {:.3f}".format(tarPrec, tarRec, tarf1))
print("PxlPrec : {:.3f}, PxlRec : {:.3f}, and PxlF1 : {:.3f}".format(pxlPrec, pxlRec, pxlf1))

if tarf1 == 0 or pxlf1 == 0:
    avg_f1 = 0
else:
    avg_f1 = 2/((1/tarf1) + (1/pxlf1))
    
mean_test_dice_score = np.mean(dice_score_list)
    
print("Average F1 score : {:.3f}, mean dice score : {:.3f}".format(avg_f1, mean_test_dice_score))

'''
