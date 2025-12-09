import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler  # ✅ Add GradScaler
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from dataset.medical_few import MedDataset
from CLIP.clip import create_model
from CLIP.tokenizer import tokenize
from CLIP.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_prompt_ensemble
from prompt import REAL_NAME

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Few-shot Adapter Training')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336')
    parser.add_argument('--pretrain', type=str, default='openai')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')  # ✅ New
    args = parser.parse_args()

    setup_seed(args.seed)
    
    # ============================================
    # ✅ CORRECTED: Model Setup
    # ============================================
    
    # Load pretrained CLIP and freeze its behavior
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, 
                              device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()  # ✅ Keep CLIP backbone in eval mode (frozen feature extractor)

    # Wrap with adapters
    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    # ✅ DO NOT call model.eval() here - we need to train adapters!

    # ✅ CORRECTED: Selective gradient enabling
    for name, param in model.named_parameters():
        if 'adapter' in name.lower():  # Only adapters
            param.requires_grad = True
        else:  # Freeze CLIP backbone
            param.requires_grad = False
    
    # Verify what's trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # ============================================
    # ✅ CORRECTED: Optimizers with weight decay
    # ============================================
    
    seg_optimizer = torch.optim.AdamW(  # ✅ AdamW instead of Adam
        model.seg_adapters.parameters(), 
        lr=args.learning_rate, 
        betas=(0.9, 0.999),  # ✅ Standard betas, not (0.5, 0.999)
        weight_decay=1e-4    # ✅ Add weight decay for regularization
    )
    det_optimizer = torch.optim.AdamW(
        model.det_adapters.parameters(), 
        lr=args.learning_rate, 
        betas=(0.9, 0.999), 
        weight_decay=1e-4
    )
    
    # ✅ Add learning rate schedulers
    seg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(seg_optimizer, T_max=args.epoch, eta_min=1e-6)
    det_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(det_optimizer, T_max=args.epoch, eta_min=1e-6)
    
    # ✅ Add gradient scaler for mixed precision
    scaler = GradScaler()

    # ============================================
    # Data Loading (unchanged)
    # ============================================
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # Few-shot augmentation
    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    augment_fewshot_label = torch.cat([torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)

    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)

    # Losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    # Text features (computed once)
    with autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)

    # ============================================
    # ✅ CORRECTED: Training Loop
    # ============================================
    
    best_result = 0
    patience_counter = 0  # ✅ For early stopping
    
    print(f"\n{'='*60}")
    print(f"Starting training: {args.obj} ({args.shot}-shot)")
    print(f"{'='*60}\n")

    for epoch in range(args.epoch):
        # ✅ Set model to TRAINING mode (adapters will have Dropout, BatchNorm active)
        model.train()
        
        print(f'Epoch {epoch}/{args.epoch}:')

        loss_list = []
        for (image, gt, label) in train_loader:
            image = image.to(device)
            
            # ✅ Use autocast and scaler properly
            with autocast():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]
                    
                # Detection loss
                det_loss = 0
                image_label = label.to(device)
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features).unsqueeze(0)    
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    det_loss += loss_bce(anomaly_score, image_label)

                if CLASS_INDEX[args.obj] > 0:
                    # Segmentation loss
                    seg_loss = 0
                    mask = gt.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    for layer in range(len(seg_patch_tokens)):
                        seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features).unsqueeze(0)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)
                    
                    loss = seg_loss + det_loss
                    # ✅ REMOVED: loss.requires_grad_(True) - this is automatic!
                    
                    # ✅ Use scaler for mixed precision
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(seg_optimizer)
                    scaler.step(det_optimizer)
                    scaler.update()

                else:
                    loss = det_loss
                    det_optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(det_optimizer)
                    scaler.update()

                loss_list.append(loss.item())

        avg_loss = np.mean(loss_list)
        print(f"  Loss: {avg_loss:.4f}")
        
        # ✅ Step schedulers
        seg_scheduler.step()
        det_scheduler.step()
        print(f"  LR: {seg_scheduler.get_last_lr()[0]:.6f}")

        # ============================================
        # ✅ Validation (model in eval mode)
        # ============================================
        
        model.eval()  # ✅ Switch to eval mode for testing
        
        # Build memory bank
        seg_features = []
        det_features = []
        for image in support_loader:
            image = image[0].to(device)
            with torch.no_grad():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [p[0].contiguous() for p in seg_patch_tokens]
                det_patch_tokens = [p[0].contiguous() for p in det_patch_tokens]
                seg_features.append(seg_patch_tokens)
                det_features.append(det_patch_tokens)
        seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
        det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]
        
        result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)
        
        # ✅ Early stopping logic
        if result > best_result:
            best_result = result
            patience_counter = 0
            print(f"  ✅ Best result: {best_result:.4f}\n")
            
            if args.save_model == 1:
                os.makedirs(args.save_path, exist_ok=True)
                ckp_path = os.path.join(args.save_path, f'{args.obj}_shot{args.shot}.pth')
                torch.save({
                    'epoch': epoch,
                    'seg_adapters': model.seg_adapters.state_dict(),
                    'det_adapters': model.det_adapters.state_dict(),
                    'seg_optimizer': seg_optimizer.state_dict(),
                    'det_optimizer': det_optimizer.state_dict(),
                    'best_result': best_result
                }, ckp_path)
                print(f"  💾 Saved checkpoint: {ckp_path}\n")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})\n")
            
            if patience_counter >= args.patience:
                print(f"⚠️  Early stopping triggered at epoch {epoch}")
                break
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best result: {best_result:.4f}")
    print(f"{'='*60}\n")


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    """Test function - unchanged but model should be in eval mode when called"""
    gt_list = []
    gt_mask_list = []

    det_image_scores_zero = []
    det_image_scores_few = []
    
    seg_score_map_zero = []
    seg_score_map_few = []

    for (image, y, mask) in tqdm(test_loader, desc='Testing'):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), autocast():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

            if CLASS_INDEX[args.obj] > 0:
                # Few-shot segmentation
                anomaly_maps_few_shot = []
                for idx, p in enumerate(seg_patch_tokens):
                    cos = cos_sim(seg_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                            size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                seg_score_map_few.append(score_map_few)

                # Zero-shot segmentation
                anomaly_maps = []
                for layer in range(len(seg_patch_tokens)):
                    seg_patch_tokens[layer] /= seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features).unsqueeze(0)
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                    anomaly_maps.append(anomaly_map.cpu().numpy())
                score_map_zero = np.sum(anomaly_maps, axis=0)
                seg_score_map_zero.append(score_map_zero)

            else:
                # Few-shot detection
                anomaly_maps_few_shot = []
                for idx, p in enumerate(det_patch_tokens):
                    cos = cos_sim(det_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                            size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                score_few_det = anomaly_map_few_shot.mean()
                det_image_scores_few.append(score_few_det)

                # Zero-shot detection
                anomaly_score = 0
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] /= det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean()
                det_image_scores_zero.append(anomaly_score.cpu().numpy())

            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)

    if CLASS_INDEX[args.obj] > 0:
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)

        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min())
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min())
    
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'  {args.obj} pAUC: {round(seg_roc_auc, 4)}')

        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'  {args.obj} AUC: {round(roc_auc_im, 4)}')

        return seg_roc_auc + roc_auc_im

    else:
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min())
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min())
    
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'  {args.obj} AUC: {round(img_roc_auc_det, 4)}')

        return img_roc_auc_det


if __name__ == '__main__':
    main()
