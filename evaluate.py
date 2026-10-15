import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, return_stats: bool = False):
    """验证一轮。

    return_stats=False（默认）：只返回平均前景 Dice —— 保持原行为
    return_stats=True          ：返回 (dice, stats)，stats 是前景像素级混淆矩阵，
                                 用于算 IoU / 灵敏度 / 特异度 / 精确率
    """
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    tp = fp = fn = tn = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                # squeeze(1)：网络输出 (N,1,H,W)，掩码真值 (N,H,W)，
                # 不 squeeze 的话 dice_coeff 的 assert input.size()==target.size() 会失败
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float().squeeze(1)
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                if return_stats:
                    p, t = mask_pred.bool(), mask_true.bool()
                    tp += int((p & t).sum()); fp += int((p & ~t).sum())
                    fn += int((~p & t).sum()); tn += int((~p & ~t).sum())
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                if return_stats:
                    p, t = mask_pred[:, 1].bool(), mask_true[:, 1].bool()
                    tp += int((p & t).sum()); fp += int((p & ~t).sum())
                    fn += int((~p & t).sum()); tn += int((~p & ~t).sum())

    net.train()
    dice = dice_score / max(num_val_batches, 1)
    if not return_stats:
        return dice

    eps = 1e-8
    stats = {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'iou': tp / (tp + fp + fn + eps),
        'sensitivity': tp / (tp + fn + eps),
        'specificity': tn / (tn + fp + eps),
        'precision': tp / (tp + fp + eps),
    }
    return dice, stats
