"""Main training script.

Usage:
    python src/train.py
    python src/train.py --debug                      # 500-sample smoke test
    python src/train.py --model ast --epochs 20
    python src/train.py --experiment exp002_ast

All hyperparameters live in Config; CLI flags override them.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent))
from config import Config, NUM_CLASSES
from dataset import build_dataloaders
from augmentation import Augmentation, batch_mixup
from evaluate import evaluate
from models import build_model


def focal_bce_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
    return ((1 - p_t) ** gamma * bce).mean()


def get_warmup_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    return base_lr


def train_one_epoch(
    model,
    bird_loader,
    sc_loader,
    optimizer,
    scaler: GradScaler,
    criterion,
    cfg: Config,
    device,
    epoch: int,
    total_steps_so_far: int,
    warmup_steps: int,
    focal_gamma: float = 0.0,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    steps = 0

    # Interleave bird and soundscape batches according to ratio
    bird_iter = iter(bird_loader)
    sc_iter   = iter(sc_loader)
    use_bird  = True

    # Estimate total batches this epoch
    n_batches = max(len(bird_loader), len(sc_loader))

    for _ in range(n_batches):
        # Alternate between bird and soundscape with the configured ratio
        if use_bird:
            try:
                specs, labels = next(bird_iter)
            except StopIteration:
                bird_iter = iter(bird_loader)
                specs, labels = next(bird_iter)
        else:
            try:
                specs, labels = next(sc_iter)
            except StopIteration:
                sc_iter = iter(sc_loader)
                specs, labels = next(sc_iter)

        # Flip next source based on ratio
        use_bird = (steps % round(1 / (1 - cfg.bird_soundscape_ratio + 1e-9))) != 0

        specs  = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Batch-level Mixup
        specs, labels = batch_mixup(specs, labels, alpha=cfg.mixup_alpha)

        # Warmup LR override
        global_step = total_steps_so_far + steps
        if global_step < warmup_steps:
            lr = get_warmup_lr(global_step, warmup_steps, cfg.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(specs)
            if focal_gamma > 0:
                loss = focal_bce_loss(logits, labels, gamma=focal_gamma)
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1), total_steps_so_far + steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="efficientnet_b3")
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--batch",        type=int,   default=None)
    parser.add_argument("--experiment",   default=None)
    parser.add_argument("--debug",        action="store_true")
    parser.add_argument("--resume",       default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--panns-weights",  default=None, dest="panns_weights",
                        help="Path to local CNN14 .pth file (skips auto-download)")
    parser.add_argument("--unfreeze-epoch", default=None, type=int, dest="unfreeze_epoch",
                        help="Epoch at which to unfreeze all layers (AST phase-2 fine-tuning)")
    parser.add_argument("--pseudo-labels",  default=None, dest="pseudo_labels",
                        help="Path to merged pseudo-labels CSV (output of pseudo_label.py)")
    parser.add_argument("--focal-gamma",   type=float, default=0.0, dest="focal_gamma",
                        help="Focal loss gamma (0 = standard BCE, 2.0 recommended)")
    parser.add_argument("--min-rating",    type=float, default=0.0, dest="min_rating",
                        help="Minimum clip rating to include from train_audio (0 = keep all)")
    args = parser.parse_args()

    cfg = Config()
    cfg.model_name = args.model
    if args.epochs     is not None: cfg.epochs     = args.epochs
    if args.lr         is not None: cfg.learning_rate = args.lr
    if args.batch      is not None: cfg.batch_size  = args.batch
    if args.experiment is not None: cfg.experiment_name = args.experiment
    if args.debug:                  cfg.debug = True
    if args.pseudo_labels:
        from pathlib import Path as _Path
        p = _Path(args.pseudo_labels)
        cfg.soundscape_labels_csv = p if p.is_absolute() else cfg.data_dir / p
        print(f"Using pseudo-labels: {cfg.soundscape_labels_csv}")
    if args.min_rating > 0:
        cfg.min_rating = args.min_rating
        print(f"Rating filter: >= {cfg.min_rating}")

    focal_gamma = args.focal_gamma
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {cfg.model_name}  |  Experiment: {cfg.experiment_name}  |  Focal γ={focal_gamma}")

    augment = Augmentation(cfg)
    bird_loader, sc_loader, val_loader = build_dataloaders(cfg, augment=augment)

    model_kwargs = {}
    if args.panns_weights:
        model_kwargs["weights_path"] = args.panns_weights
    model = build_model(cfg.model_name, NUM_CLASSES, pretrained=True, **model_kwargs).to(device)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Cosine scheduler runs after warmup
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs - cfg.warmup_epochs,
        eta_min=cfg.learning_rate * 0.01,
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 1
    best_cmap = 0.0
    steps_so_far = 0
    warmup_steps = cfg.warmup_epochs * max(len(bird_loader), len(sc_loader))

    ckpt_dir = cfg.checkpoints_dir / cfg.experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_cmap   = ckpt.get("best_cmap", 0.0)
        steps_so_far = ckpt.get("steps_so_far", 0)
        print(f"Resumed from epoch {start_epoch - 1}, best cmAP={best_cmap:.4f}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        # Phase-2 unfreeze (e.g. for AST: unfreeze all layers mid-training at lower LR)
        if args.unfreeze_epoch and epoch == args.unfreeze_epoch:
            if hasattr(model, "unfreeze_all"):
                model.unfreeze_all()
                # Drop LR by 10× so newly unfrozen layers don't overwrite learned features
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["lr"] * 0.1
                print(f"  Epoch {epoch}: phase-2 unfreeze, LR → {optimizer.param_groups[0]['lr']:.2e}")

        t0 = time.time()
        train_loss, steps_so_far = train_one_epoch(
            model, bird_loader, sc_loader,
            optimizer, scaler, criterion, cfg,
            device, epoch, steps_so_far, warmup_steps,
            focal_gamma=focal_gamma,
        )

        if epoch > cfg.warmup_epochs:
            scheduler.step()

        metrics = evaluate(model, val_loader, device)
        cmap = metrics["cmap"]

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"loss={train_loss:.4f}  "
            f"cmAP={cmap:.4f}  "
            f"classes={metrics['classes_with_data']}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"time={elapsed:.0f}s"
        )

        is_best = cmap > best_cmap
        if is_best:
            best_cmap = cmap

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_cmap": best_cmap,
            "steps_so_far": steps_so_far,
            "cfg": cfg,
        }
        torch.save(state, ckpt_dir / "last.pt")
        if is_best:
            torch.save(state, ckpt_dir / "best.pt")
            print(f"  → New best cmAP: {best_cmap:.4f}")

    print(f"\nTraining complete. Best cmAP: {best_cmap:.4f}")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
