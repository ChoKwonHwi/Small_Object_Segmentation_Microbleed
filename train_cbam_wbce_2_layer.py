from network_cbam import RLKunet, RLKunet2, initialize_weight
from network_cbam import RLKunet3, RLKunet4, RLKunet5
import torch
import numpy as np
import os
import argparse
import csv
from Microbleed_dataset import Microbleed_png_dataset
from torchvision import transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from monai.losses.dice import DiceLoss, one_hot
from monai.metrics import DiceMetric
from loss_func import FocalLoss
from loss_func import recall_Loss
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch.nn as nn
import albumentations as A
from eval import *
import warnings

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()
#parser.add_argument('--data_path', type=str, default='/nasdata4/kwonhwi/RLK_UNET_13cohort/data/all_data_3slice/fold_2')
parser.add_argument('--data_path', type=str, default='/nasdata4/kwonhwi/RLK_UNET_CBAM/c2_data_fold_3slice/fold_5')
#parser.add_argument('--data_path', type=str, default='/nasdata4/kwonhwi/RLK_UNET_13cohort/data/3slice/fold_5')
parser.add_argument('--epoch', type=int, default=3000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_path', type=str, default='/nasdata4/kwonhwi/RLK_UNET_13cohort/weights/c2_layer2_large/fold_5_rere')
parser.add_argument('--bce_weight', type=float, default=10.0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--augmentation', type=bool, default=True)
parser.add_argument('--scheduler', type=bool, default=True)
parser.add_argument('--input_channel', type=int, default=3)
parser.add_argument('--output_csv_path', type=str, default = '/nasdata4/kwonhwi/RLK_UNET_13cohort/weights/c2_layer2_large/fold_5_rere/loss.csv')
parser.add_argument('--metric_csv_path', type=str, default = '/nasdata4/kwonhwi/RLK_UNET_13cohort/weights/c2_layer2_large/fold_5_rere/metric.csv')

args = parser.parse_args()
seed = 1234
print("current seed : ", seed)
torch.manual_seed(seed)
torch.set_num_threads(2)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def criterion_weighted_BCE(pred, gt, pos_weight):
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = bce_loss(pred, gt)
    return loss

train_img_png_path = os.path.join(args.data_path, 'train')
val_img_png_path = os.path.join(args.data_path, 'valid')

if args.augmentation == False :
    transform = A.Compose([
        A.Resize(height=512, width=512),
    ])
    test_transform = A.Compose([
        A.Resize(height=512, width=512),
    ])
else :
    transform = A.Compose([
        A.CropNonEmptyMaskIfExists(height=224, width=224, p=0.3),
        A.Resize(height=512, width=512),
        A.VerticalFlip(p=0.3),
        A.HorizontalFlip(p=0.3),
        A.Rotate(limit=(-30, 30), interpolation=cv2.INTER_CUBIC, p=0.3)
        
    ])
    test_transform = A.Compose([
        A.Resize(height=512, width=512),
    ])

train_dataset = Microbleed_png_dataset(img_path=train_img_png_path, transform=transform, aug=args.augmentation)
val_dataset = Microbleed_png_dataset(img_path=val_img_png_path, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

loaders = {"train": train_loader, "valid": val_loader}

model = RLKunet4(in_channels=args.input_channel, out_channels=2, features=64, group_num=8).to(device)
model.apply(initialize_weight)

epochs = args.epoch
learning_rate = args.lr

weights_path = args.weight_path
print("weights_path to save model parameters : ", weights_path)

weights_best_name_tar    = "Best_RLK-Unet_HF_Tar.pth"
weights_best_tar         = os.path.join(weights_path, weights_best_name_tar)

weights_best_name_pxl    = "Best_RLK-Unet_HF_Pxl.pth"
weights_best_pxl        = os.path.join(weights_path, weights_best_name_pxl)

weights_best_name_avg    = "Best_RLK-Unet_HF_Avg.pth"
weights_best_avg       = os.path.join(weights_path, weights_best_name_avg)
    
weights_best_name_rec    = "Best_RLK-Unet_HF_Tar_Rec.pth"    
weights_best_tar_rec       = os.path.join(weights_path, weights_best_name_rec)

weights_best_name_prec    = "Best_RLK-Unet_HF_Tar_prec.pth"    
weights_best_tar_prec       = os.path.join(weights_path, weights_best_name_prec)

weights_best_name_tar_spec    = "Best_RLK-Unet_HF_Tar_Spec.pth"   
weights_best_tar_spec       = os.path.join(weights_path, weights_best_name_tar_spec)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

if args.scheduler == True :
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=100,
        cycle_mult=1.0,
        max_lr=2e-4,
        min_lr=1e-5,
        warmup_steps=50,
        gamma=0.95
    )

dice = DiceMetric(include_background=False, reduction='mean')
criterion = DiceLoss(include_background=False, to_onehot_y=False, softmax=False, reduction="mean")

focal_criterion = FocalLoss(alpha=1.0, gamma=2.0)
recall_criterion = recall_Loss()
step = 0
    
loss_train, loss_valid = [], []
all_loss_train, all_loss_valid = [], []
all_tar_f1, all_tar_rec, all_tar_prec = [], [], []
all_dice_train = []
train_boundary = []

max_tar_f1, max_tar_re, max_tar_prec, max_tar_spec = 0, 0, 0, 0
max_pxl_f1, max_pxl_prec, max_avg_f1 = 0, 0, 0
min_val_loss = 1000

for epoch in range(epochs):
    y_pred_list = []
    y_gt_list = []
    for phase in ["train", "valid"]:
        if phase == 'train':
            model.train()
        else:
            model.eval()
            
        train_dice_list = []
        
        for idx, (x_data, y_data) in enumerate(loaders[phase]):
            
            if phase == "train":
                step += 1
                
            x_data = x_data.to(device).float()
            y_data = y_data.to(device).float()
            # y_data = y_data.unsqueeze(1)
            y_data = torch.where(y_data > 0.0, 1.0, 0.0)
            
            maxpool = nn.MaxPool2d(2, 2)
            
            y_data_2 = maxpool(y_data)
            #y_data_3 = maxpool(y_data_2)
            #y_data_4 = maxpool(y_data_3)
            
            gt1 = torch.cat((y_data, 1 - y_data), dim=1)
            gt2 = torch.cat((y_data_2, 1 - y_data_2), dim=1)
            #gt3 = torch.cat((y_data_3, 1 - y_data_3), dim=1)
            #gt4 = torch.cat((y_data_4, 1 - y_data_4), dim=1)
            
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                y_pred2, y_pred1 = model(x_data) #y_pred4, y_pred3, y_pred2, y_pred1 = model(x_data)
                y_fg1, y_bg1, gt_fg1, gt_bg1 = y_pred1[:,0,:,:], y_pred1[:,1,:,:], gt1[:,0,:,:], gt1[:,1,:,:]
                y_fg2, y_bg2, gt_fg2, gt_bg2 = y_pred2[:,0,:,:], y_pred2[:,1,:,:], gt2[:,0,:,:], gt2[:,1,:,:]
                #y_fg3, y_bg3, gt_fg3, gt_bg3 = y_pred3[:,0,:,:], y_pred3[:,1,:,:], gt3[:,0,:,:], gt3[:,1,:,:]
                #y_fg4, y_bg4, gt_fg4, gt_bg4 = y_pred4[:,0,:,:], y_pred4[:,1,:,:], gt4[:,0,:,:], gt4[:,1,:,:]
                
                y_fg1, y_bg1, gt_fg1, gt_bg1 = y_fg1.unsqueeze(1), y_bg1.unsqueeze(1), gt_fg1.unsqueeze(1), gt_bg1.unsqueeze(1)
                y_fg2, y_bg2, gt_fg2, gt_bg2 = y_fg2.unsqueeze(1), y_bg2.unsqueeze(1), gt_fg2.unsqueeze(1), gt_bg2.unsqueeze(1)
                #y_fg3, y_bg3, gt_fg3, gt_bg3 = y_fg3.unsqueeze(1), y_bg3.unsqueeze(1), gt_fg3.unsqueeze(1), gt_bg3.unsqueeze(1)
                #y_fg4, y_bg4, gt_fg4, gt_bg4 = y_fg4.unsqueeze(1), y_bg4.unsqueeze(1), gt_fg4.unsqueeze(1), gt_bg4.unsqueeze(1)
                
                #loss4 = criterion(y_fg4, gt_fg4) + criterion(y_bg4, gt_bg4)
                #loss3 = criterion(y_fg3, gt_fg3) + criterion(y_bg3, gt_bg3) 
                loss2 = criterion(y_fg2, gt_fg2) + criterion(y_bg2, gt_bg2)
                loss1 = criterion(y_fg1, gt_fg1) + criterion(y_bg1, gt_bg1)
                
                #pos_weight_fg = torch.tensor([args.bce_weight]).to(device).float()
                #pos_weight_bg = torch.tensor([1.0]).to(device).float()
                
                #loss4_bce = criterion_weighted_BCE(y_fg4, gt_fg4, pos_weight_fg) + criterion_weighted_BCE(y_bg4, gt_bg4, pos_weight_bg)
                #loss3_bce = criterion_weighted_BCE(y_fg3, gt_fg3, pos_weight_fg) + criterion_weighted_BCE(y_bg3, gt_bg3, pos_weight_bg)
                #loss2_bce = criterion_weighted_BCE(y_fg2, gt_fg2, pos_weight_fg) + criterion_weighted_BCE(y_bg2, gt_bg2, pos_weight_bg)
                #loss1_bce = criterion_weighted_BCE(y_fg1, gt_fg1, pos_weight_fg) + criterion_weighted_BCE(y_bg1, gt_bg1, pos_weight_bg)

                #loss4_focal = focal_criterion(y_fg4, gt_fg4) + focal_criterion(y_bg4, gt_bg4)
                #loss3_focal = focal_criterion(y_fg3, gt_fg3) + focal_criterion(y_bg3, gt_bg3)
                loss2_focal = focal_criterion(y_fg2, gt_fg2) + focal_criterion(y_bg2, gt_bg2)
                loss1_focal = focal_criterion(y_fg1, gt_fg1) + focal_criterion(y_bg1, gt_bg1)

                #loss4_re = recall_criterion(y_fg4, gt_fg4) + recall_criterion(y_bg4, gt_bg4)

                loss = 0.35*(loss2) + 0.65*(loss1)
                #loss_bce = 0.1*(loss4_bce) + 0.2*(loss3_bce) + 0.3*(loss2_bce) + 0.4*(loss1_bce)
                loss_focal = 0.35*(loss2_focal) + 0.65*(loss1_focal)
                

                total_loss = loss_focal + loss# + loss4_re
                
                if phase == "train":
                    loss_train.append(total_loss.item())
                    y_pred = y_pred1[:,0,:,:]
                    y_pred = torch.unsqueeze(y_pred, 1)
                    y_pred = torch.where(y_pred >= 0.5, 1.0, 0.0)
                    dice_score = dice(y_pred, y_data)
                    train_dice_list.append(dice_score.mean().item())  
                    total_loss.backward()
                    optimizer.step() # backpropagation
                
                if phase == "valid":
                    loss_valid.append(total_loss.item())
                    y_pred = y_pred1[:,0,:,:]
                    y_pred = torch.unsqueeze(y_pred, 1)
                    y_pred = torch.where(y_pred > 0.5, 1.0, 0.0) # (B, 1, 224, 224)
                    
                    for k in range(y_pred.shape[0]):
                        y_pred_list.append(y_pred[k, 0, :, :])
                        y_gt_list.append(y_data[k, 0, :, :])
        
        if phase == "train":
            print("=================================================")
            print("epoch {}  | {}: {:.3f}".format(epoch + 1, "Train loss", np.mean(loss_train)))
            print("         | {}: {:.3f}".format("Train Dice", np.mean(train_dice_list)))
            all_loss_train.append(np.mean(loss_train))
            all_dice_train.append(np.mean(train_dice_list))
            loss_train  = []
    
    
    print("epoch {}  | {}: {:.3f}".format(epoch + 1, "Valid loss", np.mean(loss_valid)))
    print("..................................................")
    val_loss_min = np.mean(loss_valid)
    all_loss_valid.append(np.mean(loss_valid))
    loss_valid  = []
    print("Validation : Target-level and Pixel-level Prec, Rec, F1")
    evaluator = Evaluator(y_pred_list, y_gt_list, tar_area=[0, np.inf], is_print=False)


    #(Pd, Fa, TarPrec, TarRec, TarF1) = evaluator.target_metrics()
    (PxlPrec, PxlRec, PxlF1) = evaluator.pixel_metrics()
    (TP, FP, FN, TN, precision, sensitivity, f1_score, specificity, accuracy) = evaluator.target_metrics2()
    tarPrec = precision
    tarRec = sensitivity
    tarSpec = specificity
    tarf1 = f1_score
    pxlPrec = PxlPrec
    pxlRec = PxlRec
    pxlf1 = PxlF1
    is_save = False
    
    if np.isnan(tarPrec) or np.isinf(tarPrec):
        tarPrec = 0
    if np.isnan(tarRec) or np.isinf(tarRec):
        tarRec = 0
    if np.isnan(tarf1) or np.isinf(tarf1):
        tarf1 = 0
    if np.isnan(tarSpec) or np.isinf(tarSpec):
        tarSpec = 0
    if np.isnan(pxlPrec) or np.isinf(pxlPrec):
        pxlPrec = 0
    if np.isnan(pxlRec) or np.isinf(pxlRec):
        pxlRec = 0
    if np.isnan(pxlf1) or np.isinf(pxlf1):
        pxlf1 = 0
    
    print("Current Val target-level Precision : {:.3f}, Recall : {:.3f}, Specificity : {:.3f}, and F1-score : {:.3f}".format(tarPrec, tarRec, tarSpec, tarf1))
    print("Current Val pixel-level Precision : {:.3f}, Recall : {:.3f}, and F1-score : {:.3f}".format(pxlPrec, pxlRec, pxlf1))
    
    if tarf1 == 0 or pxlf1 == 0:
        avg_f1 = 0
    else:
        avg_f1 = 2/((1/tarf1) + (1/pxlf1))

    if val_loss_min < min_val_loss and tarf1 >= max_tar_f1:

        weights_best_loss_f1 = os.path.join(weights_path, f"Best_RLK-Unet_HF_loss_and_f1.pth")
        torch.save(model.state_dict(), weights_best_loss_f1)
        print(f"Saved Best model with improved loss and F1 score at epoch {epoch+1}")

    if tarf1 > max_tar_f1:
        max_tar_f1 = tarf1
        torch.save(model.state_dict(), weights_best_tar)
        print("Current Best Val target-level F1 Score : {:.3f}".format(max_tar_f1))
    
    if pxlf1 > max_pxl_f1:
        max_pxl_f1 = pxlf1
        torch.save(model.state_dict(), weights_best_pxl)
        print("Current Best Val pixel-level F1 Score : {:.3f}".format(max_pxl_f1))
    
    if avg_f1 > max_avg_f1:
        max_avg_f1 = avg_f1
        torch.save(model.state_dict(), weights_best_avg)
        print("Current Best Val average F1 Score : {:.3f}".format(max_avg_f1))

    if tarRec > max_tar_re :
        max_tar_re = tarRec
        torch.save(model.state_dict(), weights_best_tar_rec)
        print("Current Best Val Target-level Recall Score : {:.3f}".format(max_tar_re))

    if tarPrec > max_tar_prec :
        max_tar_prec = tarPrec
        torch.save(model.state_dict(), weights_best_tar_prec)
        print("Current Best Val Target-level Precision Score : {:.3f}".format(max_tar_prec))

    if tarSpec > max_tar_spec :
        max_tar_spec = tarSpec
        torch.save(model.state_dict(), weights_best_tar_spec)
        print("Current Best Val target-level Specificity Score : {:.3f}".format(max_tar_spec))

    if min_val_loss > val_loss_min :
        min_val_loss = val_loss_min
        weights_best_name_loss    = "Best_RLK-Unet_HF_loss" + str(epoch) + ".pth"   
        weights_best_loss      = os.path.join(weights_path, weights_best_name_loss)
        torch.save(model.state_dict(), weights_best_loss)
        print("Current Best Val loss : {:.5f}".format(min_val_loss))

    all_tar_f1.append(tarf1)
    all_tar_rec.append(tarRec)
    all_tar_prec.append(tarPrec)

    with open(args.output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in zip(all_loss_train, all_loss_valid):
            writer.writerow(row)

    with open(args.metric_csv_path, mode='w', newline='') as file :
        writer2 = csv.writer(file)
        for rows in zip(all_tar_prec, all_tar_rec, all_tar_f1):
            writer2.writerow(rows)
        
    if args.scheduler == True :
        scheduler.step()

print("Training Complete")