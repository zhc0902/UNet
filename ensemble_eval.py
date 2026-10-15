"""集成评估（修正版）。

======================================================================
为什么要重写：前一版有【数据泄漏】，算出来的 0.943 是假的
======================================================================
错误做法：对第 f 折的验证图，用全部 5 个模型平均预测。
    问题：这 5 个模型里有 4 个在训练时见过这些图（它们训练用的正是其余几折）。
          因为模型对训练集几乎完美记忆（train_loss ≈ 0.027），
          这些"背过答案"的模型会给出接近 GT 的预测，平均后接近 1.0。
    表现：单模型 0.717 → "集成" 0.943，涨了 23 个点。
          真实集成增益上限只有 1~3 个点，涨这么多一定是泄漏。

正确做法（本脚本）：
    (a) 单模型 · 无偏       每个 model_f 只在自己那一折的留出集上评估
                            5 折合起来覆盖全部 647 张 → 直接就是 CV 分数
    (b) 集成 · 无偏         只在"训练时没见过该折"的模型之间做平均

======================================================================
用法
======================================================================
  仅复核无偏 CV（秒级，只用已有权重）：
      python ensemble_eval.py
          → 这一步就够写论文了：0.71xx ± 0.0xxx

  想额外做集成（可选，收益通常只有 1~2 个点）：
      需要"训练时没见过测试折"的模型。
      注意：标准 5 折里 model_f 训练用的是"除 fold_f 外的全部数据"，
            所以每个模型都见过别的折，它们之间不能直接互相集成。
      可选做法（任选其一）：
        (1) 跳过集成，只用上面的无偏 CV 分数 —— 推荐，省时且够用；
        (2) 手动把某些折的图从训练集里排除后另外训几个模型，
            再用 --test-fold 指定留出折、--ensemble-glob 指定这些权重。
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset

EPS = 1e-8


# --------------------------------------------------------------- 数据
def load_fold_table(csv_path: Path):
    table = {}
    with csv_path.open(encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            table[row['stem']] = dict(fold=int(row['fold']),
                                      cls=row['class'],
                                      is_empty=int(row['is_empty']))
    return table


def build_folds(dataset, csv_path: Path, n_folds: int):
    table = load_fold_table(csv_path)
    folds = [[] for _ in range(n_folds)]
    for i, stem in enumerate(dataset.ids):
        rec = table.get(stem)
        if rec is None or rec['is_empty']:
            continue                        # 空掩膜图不参与评估
        folds[rec['fold']].append(i)
    return folds


# --------------------------------------------------------------- 模型
def load_model(ckpt: Path, device, bilinear=False, classes=2):
    model = UNet(n_channels=3, n_classes=classes, bilinear=bilinear)
    # weights_only=True：只反序列化张量，不执行 pickle 代码。
    # 新版 PyTorch (>=2.6) 默认已是 True，不显式写会刷一屏 FutureWarning。
    try:
        sd = torch.load(str(ckpt), map_location=device, weights_only=True)
    except TypeError:                       # 老版本 torch 没有这个参数
        sd = torch.load(str(ckpt), map_location=device)
    if isinstance(sd, dict):
        sd.pop('mask_values', None)
    model.load_state_dict(sd)
    model.to(device=device).eval()
    return model


class ConfMat:
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def add(self, pred, gt):
        self.tp += int((pred & gt).sum()); self.fp += int((pred & ~gt).sum())
        self.fn += int((~pred & gt).sum()); self.tn += int((~pred & ~gt).sum())

    def merge(self, o):
        self.tp += o.tp; self.fp += o.fp; self.fn += o.fn; self.tn += o.tn

    def metrics(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        return dict(dice=2 * tp / (2 * tp + fp + fn + EPS),
                    iou=tp / (tp + fp + fn + EPS),
                    sensitivity=tp / (tp + fn + EPS),
                    specificity=tn / (tn + fp + EPS),
                    precision=tp / (tp + fp + EPS))


@torch.inference_mode()
def foreground_prob(models, images, amp):
    """返回 [模型0的前景概率, 模型1的, ...]"""
    out = []
    for m in models:
        with torch.autocast('cuda' if images.is_cuda else 'cpu', enabled=amp):
            logits = m(images)
        out.append(torch.sigmoid(logits[:, 1]))
    return out


def evaluate_split(models, dataset, indices, device, batch_size, threshold, amp,
                   agg='mean'):
    """在给定下标集合上评估。models 为空则报错。
    agg='mean'  -> 概率平均后阈值化（集成）
    agg='single'-> 只用 models[0]
    """
    cm = ConfMat()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)
    with torch.inference_mode():
        for batch in loader:
            img = batch['image'].to(device, dtype=torch.float32)
            gt = batch['mask'].to(device, dtype=torch.long).bool()
            probs = foreground_prob(models, img, amp)
            p = probs[0] if agg == 'single' else torch.stack(probs, 0).mean(0)
            cm.add(p > threshold, gt)
    return cm


def fmt(tag, m):
    return (f'  {tag:<28s} Dice={m["dice"]:.4f}  IoU={m["iou"]:.4f}  '
            f'Sens={m["sensitivity"]:.4f}  Spec={m["specificity"]:.4f}  '
            f'Prec={m["precision"]:.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=str, default='./data/BUSI')
    ap.add_argument('--ckpt-dir', type=str, default='./checkpoints')
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--image-size', type=int, default=320)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--amp', action='store_true', default=True)

    ap.add_argument('--test-fold', type=int, default=None,
                    help='（可选）用哪一折当留出测试集做集成。'
                         '需要配合 --ensemble-glob 指定"训练时没见过该折"的权重')
    ap.add_argument('--ensemble-glob', type=str, default=None,
                    help="集成用的权重通配符，例如 'checkpoint_best_fold0_excl4_*.pth'。"
                         '这些权重的训练数据里必须不包含 --test-fold 那一折')
    ap.add_argument('--out', type=str, default='ensemble_result.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir, ckpt_dir = Path(args.data_dir), Path(args.ckpt_dir)
    csv_path = data_dir / 'folds_5.csv'
    for p in (data_dir / 'imgs', data_dir / 'masks', csv_path):
        if not p.exists():
            raise SystemExit(f'路径不存在: {p}')

    # ---- 数据 ----
    try:
        dataset = CarvanaDataset(data_dir / 'imgs', data_dir / 'masks', 1.0,
                                 image_size=args.image_size)
    except Exception:
        dataset = BasicDataset(data_dir / 'imgs', data_dir / 'masks', 1.0,
                               mask_suffix='_mask', image_size=args.image_size)
    dataset.ids = sorted(dataset.ids)
    folds = build_folds(dataset, csv_path, args.n_folds)
    print(f'各折验证集大小: {[len(f) for f in folds]}   合计 {sum(len(f) for f in folds)}')

    def find_ckpt(fold):
        c = ckpt_dir / f'checkpoint_best_fold{fold}.pth'
        return c if c.exists() else None

    result = dict(n_folds=args.n_folds, threshold=args.threshold,
                  n_eval_images=sum(len(f) for f in folds))

    # ==================================================================
    # (a) 单模型 · 无偏 CV：每个 model_f 只在自己那一折上评估
    # ==================================================================
    print()
    print('=' * 88)
    print('(a) 单模型 · 无偏交叉验证   ——  每个 model_f 只在自己那一折的留出集上评估')
    print('=' * 88)
    total = ConfMat()
    per_fold = []
    for f in range(args.n_folds):
        ck = find_ckpt(f)
        if ck is None:
            print(f'  fold {f}: 缺少 {ck}，跳过')
            per_fold.append(None)
            continue
        m0 = load_model(ck, device)
        cm = evaluate_split([m0], dataset, folds[f], device, args.batch_size,
                            args.threshold, args.amp, agg='single')
        mm = cm.metrics()
        per_fold.append(mm['dice'])
        total.merge(cm)
        print(fmt(f'fold {f} (n={len(folds[f])})', mm))
    d = np.array([x for x in per_fold if x is not None])
    tm = total.metrics()
    print('  ' + '-' * 84)
    print(fmt('全部 647 张（无偏汇总）', tm))
    print(f'  各折 Dice 均值 ± 标准差  : {d.mean():.4f} ± {d.std():.4f}  (n={len(d)})')
    result['single_model_unbiased'] = tm
    result['per_fold_single_dice'] = [None if x is None else float(x) for x in per_fold]
    result['per_fold_mean_std'] = [float(d.mean()), float(d.std())]

    # ==================================================================
    # (b) 集成 · 无偏：必须用"训练时没见过 test-fold"的模型
    # ==================================================================
    if args.test_fold is None:
        print()
        print('=' * 88)
        print('(b) 未指定 --test-fold，跳过集成评估')
        print('    说明：5 折体系里直接拿全部模型平均会【数据泄漏】——')
        print('          model_f 训练时用的是除 fold_f 外的全部数据，')
        print('          所以每个模型都见过别的折，它们之间不能直接互相集成。')
        print('    集成是可选加分项（通常只有 1~2 个点），(a) 的无偏 CV 分数本身已够用。')
        print('=' * 88)
    else:
        tf = args.test_fold
        print()
        print('=' * 88)
        print(f'(b) 集成 · 无偏   ——  只在 fold {tf}（{len(folds[tf])} 张）上评估')
        print('=' * 88)

        # 只认显式指定的 glob —— 因为这些权重的训练数据必须不含 test-fold。
        # 不能自动去抓 checkpoint_best_fold*.pth：它们全部见过 fold_tf。
        cands = []
        if args.ensemble_glob:
            cands = sorted(ckpt_dir.glob(args.ensemble_glob))
            print(f'  用 --ensemble-glob 指定: {len(cands)} 个权重')
            for c in cands:
                print(f'    - {c.name}')

        if len(cands) < 2:
            print()
            print('  !! 需要至少 2 个"训练时未见过 fold '
                  f'{tf}"的权重，当前只有 {len(cands)} 个。')
            print(f'     标准 5 折的 checkpoint_best_fold*.pth 都见过 fold {tf}，不能用于集成。')
            print()
            print('     建议：跳过集成，直接用 (a) 的无偏 CV 分数。')
            print('     如果确实要做集成，需要另外训一批"训练集不含 fold '
                  f'{tf}"的模型，')
            print('     再用 --ensemble-glob 指给本脚本。')
            print()
            print('     或者临时验证集成机制（2 个模型，样本只有 130 张）：')
            print(f'       python ensemble_eval.py --test-fold {tf} '
                  f"--ensemble-glob 'checkpoint_best_fold{{0,1}}.pth'")
        else:
            # 同时给出：单个模型在 fold tf 上的分数（作为对照）
            first = load_model(cands[0], device)
            cm1 = evaluate_split([first], dataset, folds[tf], device, args.batch_size,
                                 args.threshold, args.amp, agg='single')
            print(fmt(f'单模型 ({cands[0].name})', cm1.metrics()))

            models = [load_model(c, device) for c in cands]
            cmE = evaluate_split(models, dataset, folds[tf], device, args.batch_size,
                                 args.threshold, args.amp, agg='mean')
            em = cmE.metrics()
            print(fmt(f'{len(models)} 模型集成', em))
            gain = em['dice'] - cm1.metrics()['dice']
            print()
            print(f'  >>> 集成增益（相对上面那个单模型）: {gain:+.4f} Dice')
            if gain > 0.08:
                print(f'  ⚠️ 增益 >8 个点，偏大。请确认这些权重的训练数据里'
                      f'确实不含 fold {tf}，否则就是数据泄漏。')
            result['ensemble_unbiased'] = em
            result['ensemble_n_models'] = len(models)
            result['ensemble_gain'] = float(gain)
            result['ensemble_test_fold'] = tf

    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                              encoding='utf-8')
    print(f'\n结果已写入: {args.out}')


if __name__ == '__main__':
    main()
