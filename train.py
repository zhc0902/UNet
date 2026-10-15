import argparse
import csv
import logging
import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

# 默认数据路径（可用 --data-dir / --images-dir / --masks-dir / --fold-csv 覆盖）
DEFAULT_DATA_DIR = Path('./data/BUSI')
dir_img = DEFAULT_DATA_DIR / 'imgs'
dir_mask = DEFAULT_DATA_DIR / 'masks'
FOLD_CSV = DEFAULT_DATA_DIR / 'folds_5.csv'
dir_checkpoint = Path('./checkpoints/')


def resolve_paths(args):
    """解析数据路径并做存在性检查。

    优先级：命令行指定 > DEFAULT_DATA_DIR/<标准名>。
    这样目录结构换了（比如 data/imgs 改成 data/BUSI/imgs）只需传参数，
    不用改代码；而且路径写错会在启动时立刻报错，不会训到一半才炸。
    """
    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    img_dir = Path(args.images_dir) if args.images_dir else data_dir / 'imgs'
    msk_dir = Path(args.masks_dir) if args.masks_dir else data_dir / 'masks'
    csv_path = Path(args.fold_csv) if args.fold_csv else data_dir / 'folds_5.csv'

    problems = []
    if not img_dir.is_dir():
        problems.append(f'图像目录不存在: {img_dir}')
    if not msk_dir.is_dir():
        problems.append(f'掩膜目录不存在: {msk_dir}')
    if not csv_path.is_file():
        problems.append(f'划分文件不存在: {csv_path}')
    if problems:
        hint = (f'\n\n当前 --data-dir = {data_dir}\n'
                f'该目录下应该有 imgs/、masks/、folds_5.csv 三项。\n'
                f'如果你的结构不是这样，用这些参数指定：\n'
                f'  --images-dir <图像目录>  --masks-dir <掩膜目录>  --fold-csv <划分文件>')
        raise SystemExit('数据路径检查未通过:\n  - ' + '\n  - '.join(problems) + hint)

    n_img = len(list(img_dir.glob('*.png')))
    n_msk = len(list(msk_dir.glob('*.png')))
    logging.info(f'数据路径: {img_dir} ({n_img} 张) | {msk_dir} ({n_msk} 张) | {csv_path}')
    if n_img != n_msk:
        logging.warning(f'图像数({n_img}) 与掩膜数({n_msk}) 不一致，请检查')
    return img_dir, msk_dir, csv_path


# ======================================================================
#  5 折交叉验证的数据集封装
# ======================================================================
class FoldDataset(Subset):
    """给子集起个名字，并确保 .dataset / .indices 可用（DataLoader 需要）。"""

    def __init__(self, dataset, indices, tag=''):
        super().__init__(dataset, indices)
        self.tag = tag


def load_fold_table(csv_path: Path = None):
    """读划分文件。

    文件由 prep_busi.py 生成，一行一张图：
        stem,class,is_empty,fold,n_src_masks
    其中 fold = 0..4 表示该图属于哪一折的验证集；fold = -1 表示空掩膜（normal）。

    为什么用文件而不是随机种子：
      * 划分固定、可核对，任何人拿到这个 CSV 都能复现；
      * stem 是唯一标识，不依赖 dataset 的扫描顺序（更稳）。
    """
    csv_path = Path(csv_path) if csv_path else FOLD_CSV
    if not csv_path.exists():
        raise SystemExit(f'找不到划分文件: {csv_path}')
    table = {}
    with csv_path.open(encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            table[row['stem']] = dict(
                fold=int(row['fold']),
                cls=row['class'],
                is_empty=int(row['is_empty']),
            )
    return table


def build_datasets(dataset, fold: int, n_folds: int, keep_empty_masks: bool,
                   fold_csv: Path = None):
    """按划分文件切出 train / val。

    规则：
      * 验证集 = fold == 当前折 的【含病灶】图（绝不混入空掩膜，否则 Dice 是假的）
      * 训练集 = 其余含病灶图 (+ 全部空掩膜图，若 keep_empty_masks)
    """
    table = load_fold_table(fold_csv)
    ids = getattr(dataset, 'ids', None)
    if ids is None:
        raise SystemExit('数据集没有 .ids 属性，无法按 stem 匹配划分文件')

    train_idx, val_idx, n_empty_in_train = [], [], 0
    unknown = []
    for i, stem in enumerate(ids):
        rec = table.get(stem)
        if rec is None:
            unknown.append(stem)
            continue
        if rec['is_empty']:
            if keep_empty_masks:
                train_idx.append(i)
                n_empty_in_train += 1
            continue
        if rec['fold'] == fold:
            val_idx.append(i)
        else:
            train_idx.append(i)

    if unknown:
        raise SystemExit(f'有 {len(unknown)} 个样本不在划分文件里，例如 {unknown[:5]}')
    if not val_idx:
        raise SystemExit(f'第 {fold} 折的验证集为空，检查 n_folds 是否与划分文件一致')

    info = dict(n_all=len(ids), n_train=len(train_idx), n_val=len(val_idx),
                n_normal_train=n_empty_in_train, fold=fold, n_folds=n_folds)
    return FoldDataset(dataset, train_idx, 'train'), FoldDataset(dataset, val_idx, 'val'), info


# ======================================================================
#  W&B 日志
# ======================================================================
class HistoryCSV:
    """每个 epoch 一行本地 CSV —— 不依赖网络，wandb 挂了数据也在。"""

    FIELDS = ['epoch', 'train_loss', 'val_dice', 'val_iou', 'val_sensitivity',
              'val_specificity', 'val_precision', 'lr', 'sec',
              'tp', 'fp', 'fn', 'tn']

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(','.join(self.FIELDS) + '\n', encoding='utf-8')

    def append(self, row: dict):
        with self.path.open('a', encoding='utf-8') as f:
            f.write(','.join(str(row.get(k, '')) for k in self.FIELDS) + '\n')


def wandb_init(args, device, model, info):
    """初始化 W&B。

    注意原来那行的两个坑：
      anonymous='must' —— 强制匿名，登录了账号也用不上，数据还找不回来
      resume='allow'   —— 同名 run 会被合并，多折实验互相覆盖
    另外 online 失败时自动降级 offline，避免网络问题把训练搞崩。
    """
    if args.wandb_disabled:
        return None
    dir_checkpoint.mkdir(parents=True, exist_ok=True)

    name = args.wandb_run_name or f'fold{info["fold"]}'
    tags = [t for t in (args.wandb_tags or '').split(',') if t.strip()]
    tags.append(f'fold{info["fold"]}')

    kw = dict(project=args.wandb_project, name=name, tags=tags,
              group=args.wandb_group or f'{info["n_folds"]}fold-cv',
              dir=str(dir_checkpoint))
    if args.wandb_entity:
        kw['entity'] = args.wandb_entity

    exp = None
    for mode in ('online', 'offline'):
        try:
            exp = wandb.init(mode=mode, **kw)
            if mode == 'offline':
                logging.warning('W&B online 失败，已降级为 offline（本地记录）')
            break
        except Exception as e:
            logging.warning(f'W&B {mode} 初始化失败: {e}')
    if exp is None:
        logging.warning('W&B 完全不可用，本次只写本地 CSV')
        return None

    exp.config.update(dict(
        fold=info['fold'], n_folds=info['n_folds'],
        n_train=info['n_train'], n_val=info['n_val'],
        n_normal_in_train=info['n_normal_train'], n_all=info['n_all'],
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        amp=args.amp, img_scale=args.scale, optimizer=args.optimizer,
        device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
        n_params=sum(p.numel() for p in model.parameters()),
    ))
    try:
        exp.define_metric('train/loss_batch', step_metric='step')
        exp.define_metric('val/*', step_metric='_step')
        exp.define_metric('train/*', step_metric='_step')
    except Exception:
        pass
    logging.info(f'W&B run: {getattr(exp, "url", None) or exp.dir}')
    return exp



def train_model(
        model,
        device,
        args,
        epochs: int = 300,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-4,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        fold: int = 0,
        n_folds: int = 5,
        keep_empty_masks: bool = True,
        image_size: int = 0,
        optimizer_name: str = 'adamw',
        img_dir: Path = None,
        msk_dir: Path = None,
        fold_csv: Path = None,
):
    # 0. 解析数据路径（不传就用默认的 data/BUSI/）
    img_dir = Path(img_dir) if img_dir else dir_img
    msk_dir = Path(msk_dir) if msk_dir else dir_mask
    fold_csv = Path(fold_csv) if fold_csv else FOLD_CSV

    # 1. Create dataset
    if image_size and image_size > 0:
        try:
            dataset = CarvanaDataset(img_dir, msk_dir, img_scale, image_size=image_size)
        except (AssertionError, RuntimeError, IndexError):
            dataset = BasicDataset(img_dir, msk_dir, img_scale, mask_suffix='_mask',
                                   image_size=image_size)
    else:
        try:
            dataset = CarvanaDataset(img_dir, msk_dir, img_scale)
        except (AssertionError, RuntimeError, IndexError):
            dataset = BasicDataset(img_dir, msk_dir, img_scale)
    # 固定扫描顺序，保证 .ids 与 index 的对应稳定
    dataset.ids = sorted(dataset.ids)

    # 2. 按划分文件切出 train / val（5 折）
    train_set, val_set, info = build_datasets(dataset, fold, n_folds, keep_empty_masks,
                                              fold_csv=fold_csv)
    n_train, n_val = len(train_set), len(val_set)

    # 3. Create data loaders
    # Windows 上 num_workers 用 0 最稳（多进程加载容易报内存错误），Linux 可用 4~8
    n_workers = 0 if os.name == 'nt' else min(8, os.cpu_count() or 4)
    loader_args = dict(batch_size=batch_size, num_workers=n_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=(n_train > batch_size), **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    # (Initialize logging)
    experiment = wandb_init(args, device, model, info)

    def wlog(payload, step=None):
        if experiment is None:
            return
        try:
            experiment.log(payload, step=step)
        except Exception as e:
            logging.warning(f'W&B log 失败（不影响训练）: {e}')

    logging.info(f'''Starting training:
        Fold:            {fold} / {n_folds}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Optimizer:       {optimizer_name}
        Training size:   {n_train}  (含 normal {info['n_normal_train']} 张)
        Validation size: {n_val}    (只含病灶图)
        Image size:      {image_size if image_size else f'scale={img_scale}'}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # 原版是 RMSprop + momentum=0.999，对学习率极敏感（lr=1e-3 容易发散），默认改用 AdamW
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=weight_decay, foreach=True)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay, foreach=True)
    else:
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=learning_rate, weight_decay=weight_decay,
                                  momentum=momentum, foreach=True)
    # patience 按"验证次数"计；每个 epoch 验证一次，所以 15 = 15 个 epoch 没进步才降 lr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=15, factor=0.5, min_lr=1e-6)
    try:
        grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)
    except (AttributeError, TypeError):
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    best_score = 0.0
    last_stats = None
    history = HistoryCSV(dir_checkpoint / f'history_fold{fold}.csv')

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # step 必须用 log(step=...) 传，不能塞进 payload 当普通指标，
                # 否则步数不单调 -> wandb 会刷 "Tried to log to step X < current step Y"
                wlog({'train/loss_batch': loss.item()}, step=global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # ==============================================================
        # 每个 epoch 结束时验证一次（原版是每个 epoch 验 5 次）
        # ==============================================================
        # 原版按 global_step 触发，1 个 epoch 验 5 次；而 ReduceLROnPlateau 的
        # patience 是按"验证次数"计数的，patience=5 就等于"1 个 epoch 没进步就降 lr"，
        # 对 300 epoch 太激进（实测第 2 个 epoch lr 就从 1e-3 掉到 1e-4，几乎学不动）。
        t0 = time.time()
        val_score, vstats = evaluate(model, val_loader, device, amp, return_stats=True)
        scheduler.step(val_score)
        lr_now = optimizer.param_groups[0]['lr']
        train_loss_epoch = epoch_loss / max(global_step, 1)
        logging.info(
            'Epoch {}/{}  train_loss={:.4f}  Dice={:.4f}  IoU={:.4f}  '
            'Sens={:.4f}  Spec={:.4f}  lr={:.2e}'.format(
                epoch, epochs, train_loss_epoch, val_score, vstats['iou'],
                vstats['sensitivity'], vstats['specificity'], lr_now))

        payload = {
            'train/lr': lr_now,
            'train/loss_epoch': train_loss_epoch,
            'train/epoch': epoch,
            'val/dice': val_score,
            'val/iou': vstats['iou'],
            'val/sensitivity': vstats['sensitivity'],
            'val/specificity': vstats['specificity'],
            'val/precision': vstats['precision'],
        }

        # 定期上传一批「原图 / GT / 预测」，肉眼检验分割质量。
        # 注意掩膜是 2 维 (H,W)，而 wandb 新版要求 (H,W,C)，直接传会抛
        #   RuntimeError: input.dim() = 2 is not equal to len(dims) = 3
        # 而且这个异常会把整个训练杀掉，所以既要做形状转换，也要 try 包住。
        if experiment is not None and args.wandb_images and epoch % args.wandb_images_every == 0:
            def _wb_img(t):
                a = t.detach().float().cpu()
                if a.dim() == 3 and a.shape[0] in (1, 3):
                    a = a.permute(1, 2, 0)
                elif a.dim() == 2:
                    a = a.unsqueeze(-1)
                if a.shape[-1] == 1:
                    a = a.repeat(1, 1, 3)
                lo, hi = float(a.min()), float(a.max())
                a = (a - lo) / (hi - lo + 1e-8)
                return wandb.Image((a.numpy() * 255).astype('uint8'))

            try:
                vb = next(iter(val_loader))
                v_img = vb['image'].to(device=device, dtype=torch.float32)
                v_msk = vb['mask'].to(device=device, dtype=torch.long)
                with torch.inference_mode():
                    v_logits = model(v_img)
                payload['samples'] = {
                    'image': _wb_img(v_img[0]),
                    'true': _wb_img(v_msk[0]),
                    'pred': _wb_img(v_logits.argmax(dim=1)[0]),
                }
            except Exception as e:
                logging.warning(f'样本图上传失败（已跳过，不影响训练）: {e}')

        wlog(payload, step=global_step)

        history.append({
            'epoch': epoch,
            'train_loss': f'{train_loss_epoch:.6f}',
            'val_dice': f'{val_score:.6f}',
            'val_iou': f'{vstats["iou"]:.6f}',
            'val_sensitivity': f'{vstats["sensitivity"]:.6f}',
            'val_specificity': f'{vstats["specificity"]:.6f}',
            'val_precision': f'{vstats["precision"]:.6f}',
            'lr': f'{lr_now:.3e}',
            'sec': f'{time.time() - t0:.1f}',
            'tp': vstats['tp'], 'fp': vstats['fp'],
            'fn': vstats['fn'], 'tn': vstats['tn'],
        })

        # 保存最佳权重（换机器也能把模型取回来）
        if val_score > best_score:
            best_score = val_score
            if save_checkpoint:
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / f'checkpoint_best_fold{fold}.pth'))
                logging.info(f'  -> 新的最佳 Dice {best_score:.4f}')

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    logging.info(f'第 {fold} 折训练结束。最佳 Dice = {best_score:.4f}')
    if experiment is not None:
        try:
            experiment.summary['best_val_dice'] = best_score
            experiment.finish()
        except Exception:
            pass
    return best_score


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate（原版 1e-5 配 RMSprop 基本学不动，默认改 1e-3）', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='（已废弃，划分统一由 data/folds_5.csv 决定）')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    # ---- 数据路径 ----
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据根目录（默认 ./data/BUSI，其中应有 imgs/ masks/ folds_5.csv）')
    parser.add_argument('--images-dir', type=str, default=None, help='单独指定图像目录')
    parser.add_argument('--masks-dir', type=str, default=None, help='单独指定掩膜目录')
    parser.add_argument('--fold-csv', type=str, default=None, help='单独指定划分文件路径')

    # ---- 5 折交叉验证 ----
    parser.add_argument('--fold', type=int, default=0,
                        help='当前跑第几折（0..n_folds-1），划分来自 data/BUSI/folds_5.csv')
    parser.add_argument('--n-folds', type=int, default=5, help='折数（与划分文件一致）')
    parser.add_argument('--keep-empty-masks', dest='keep_empty_masks',
                        action='store_true', default=True,
                        help='把 133 张空掩膜的 normal 图加入【训练集】当负样本（默认开）。'
                             '它们永远不进验证集，否则 Dice 计算会白送分')
    parser.add_argument('--no-empty-masks', dest='keep_empty_masks', action='store_false',
                        help='不把 normal 图加入训练')

    # ---- 输入尺寸与优化器 ----
    parser.add_argument('--image-size', type=int, default=320,
                        help='>0 时等比例缩放到长边 N 并居中 padding 成 NxN（不拉伸病灶，'
                             '且保证 batch_size>1 可用）。设 0 用 --scale 缩放')
    parser.add_argument('--no-keep-aspect', dest='keep_aspect', action='store_false', default=True,
                        help='（保留参数）关闭等比例缩放')
    parser.add_argument('--optimizer', choices=['adamw', 'adam', 'rmsprop'], default='adamw',
                        help='原版是 RMSprop+momentum=0.999，对 lr 极敏感易发散，默认 AdamW')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')

    # ---- W&B ----
    parser.add_argument('--wandb-project', type=str, default='BUSI-UNet', help='W&B 项目名')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B 团队/用户名（个人账号留空）')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='run 名（留空自动用 fold{N}）')
    parser.add_argument('--wandb-group', type=str, default=None,
                        help='group（多折放一组，网页上可聚合对比），默认 5fold-cv')
    parser.add_argument('--wandb-tags', type=str, default='', help='逗号分隔的标签')
    parser.add_argument('--wandb-disabled', action='store_true', help='完全关闭 W&B')
    parser.add_argument('--no-wandb-images', dest='wandb_images', action='store_false',
                        default=True, help='不上传样本图（省流量）')
    parser.add_argument('--wandb-images-every', type=int, default=10,
                        help='每多少个 epoch 上传一次样本图')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 启动时就检查数据路径（写错立刻报错，不会训到一半才炸）
    img_dir, msk_dir, fold_csv = resolve_paths(args)

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    call_kw = dict(
        model=model, device=device, args=args,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        img_scale=args.scale, val_percent=args.val / 100, amp=args.amp,
        fold=args.fold, n_folds=args.n_folds,
        keep_empty_masks=args.keep_empty_masks,
        image_size=args.image_size, optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        img_dir=img_dir, msk_dir=msk_dir, fold_csv=fold_csv,
    )
    try:
        train_model(**call_kw)
    except torch.cuda.OutOfMemoryError:
        logging.error('显存不足！启用 checkpointing 会降速。'
                      '也可以调小 --batch-size 或 --image-size，或加 --amp')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(**call_kw)
