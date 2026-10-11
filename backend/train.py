"""Train MobileNetV2 transfer-learning classifier (PyTorch + CUDA).

Compute layout - GPU does the heavy lifting, CPU is (almost) idle:
  * GPU (RTX): model forward/backward (AMP) + ALL augmentation math
    (flip / rotate / zoom / shift / brightness, batched on CUDA) + normalize.
  * CPU: only reads pre-decoded bytes from a memory-mapped cache file
    (built once, shared read-only across loader workers - no copies).

Uses CUDA when available and falls back to classic CPU augmentation otherwise.

Fastest dataset (pre-split, clean lab photos):
    python -m backend.train --from-kaggle plantvillage38 --epochs 15

GPU setup (Windows, RTX 4050 tested):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

Layout: --data may be flat (data/<class>/*.jpg) or pre-split
(data/train/<class>, data/valid/<class> [, data/test/<class>]).
Pre-split is auto-detected and preferred when present.

Output (picked up automatically by the app on restart):
  model/plant_disease_mobilenet.pt, model/labels.json, model/metrics.json,
  model/training_curves.png, model/confusion_matrix.png
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from .datasets import DATASETS

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "model"
PRESETS = {d["id"]: d["kaggle"] for d in DATASETS}
PROGRESS_FILE = MODEL_DIR / "training_progress.json"
_NORM_CACHE = {}


def download_from_kaggle(preset: str, target: Path):
    slug = PRESETS[preset]
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle dataset {slug} -> {target} ...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", slug,
                        "-p", str(target), "--unzip"], check=True)
    except FileNotFoundError:
        sys.exit("Kaggle CLI not found. Run: pip install kaggle "
                 "and place kaggle.json in ~/.kaggle/ (see About page).")
    except subprocess.CalledProcessError:
        sys.exit("Kaggle download failed - check your kaggle.json API key "
                 "and network connection.")
    print("Download complete.")


def resolve_splits(data: Path):
    """Return (train_dir, val_dir, test_dir|None). Auto-detect pre-splits."""
    for train_name in ("train", "Training", "training"):
        t = data / train_name
        if t.is_dir():
            v = next((data / n for n in ("valid", "val", "Validation", "test", "Test")
                      if (data / n).is_dir()), None)
            test = (data / "test") if (data / "test").is_dir() and v is not None \
                and v.name != "test" else None
            return t, v, test
    # some downloads nest one level: data/<dataset>/train/... - only descend
    # into a child that itself contains a train split (never into class folders)
    for child in data.iterdir():
        if child.is_dir() and any(
            (child / n).is_dir() for n in ("train", "Training", "training")
        ):
            return resolve_splits(child)
    return data, None, None


def write_progress(**kw):
    """Persist live training status for GET /api/training-status."""
    try:
        data = {}
        if PROGRESS_FILE.exists():
            data = json.loads(PROGRESS_FILE.read_text())
        data.update(kw)
        PROGRESS_FILE.write_text(json.dumps(data))
    except Exception:
        pass  # monitoring must never break training


def gpu_augment(x):
    """Batched augmentation fully on CUDA: flip, rotate±15°, zoom .85–1.15,
    shift ±10%, brightness ×0.8–1.2. x is float [0,1] on a CUDA device."""
    import torch
    import torch.nn.functional as F
    B = x.size(0)
    dev = x.device
    flip = torch.rand(B, device=dev) < 0.5
    if bool(flip.any()):
        x = x.clone()
        x[flip] = x[flip].flip(-1)
    ang = (torch.rand(B, device=dev) - 0.5) * (math.pi / 6)
    s = 0.85 + torch.rand(B, device=dev) * 0.30
    tx = (torch.rand(B, device=dev) - 0.5) * 0.2
    ty = (torch.rand(B, device=dev) - 0.5) * 0.2
    cos_a, sin_a = torch.cos(ang) * s, torch.sin(ang) * s
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos_a, -sin_a, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin_a, cos_a, ty
    grid = F.affine_grid(theta, list(x.size()), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                      align_corners=False)
    b = 0.8 + torch.rand(B, 1, 1, 1, device=dev) * 0.4
    return (x * b).clamp_(0, 1)


def _norm_tensors(device, mean, std):
    import torch
    key = str(device)
    if key not in _NORM_CACHE:
        _NORM_CACHE[key] = (
            torch.tensor(mean, device=device).view(1, 3, 1, 1),
            torch.tensor(std, device=device).view(1, 3, 1, 1),
        )
    return _NORM_CACHE[key]


class CachedSplit:
    """torch Dataset over a read-only memmap of pre-decoded uint8 images.
    Zero per-epoch CPU decode cost; mmap pages are shared across workers.
    Paths only are pickled - each worker opens its own file view (no copies)."""

    def __init__(self, arr_path, labels_path):
        meta_path = Path(arr_path).with_name(Path(arr_path).stem + ".json")
        meta = json.loads(meta_path.read_text())
        self._arr_path = str(arr_path)
        self._lab_path = str(labels_path)
        self._n = int(meta["n"])
        self._shape = tuple(meta["shape"])
        self.arr = None
        self.labels = None

    def _ensure_open(self):
        if self.arr is None:
            import numpy as np
            self.arr = np.memmap(self._arr_path, dtype=np.uint8,
                                 mode="r", shape=self._shape)
            self.labels = np.load(self._lab_path, mmap_mode="r")

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        import torch
        self._ensure_open()
        return torch.from_numpy(self.arr[i].copy()), int(self.labels[i])


def build_cache(folder: Path, img_size: int, cache_dir: Path, tag: str):
    """Decode+resize a split once into a uint8 memmap. Returns paths+classes."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from torchvision.datasets import ImageFolder
    base = ImageFolder(str(folder))
    n = len(base)
    cache_dir.mkdir(parents=True, exist_ok=True)
    arr_path = cache_dir / f"{tag}_{img_size}.npy"
    lab_path = cache_dir / f"{tag}_{img_size}_labels.npy"
    meta_path = cache_dir / f"{tag}_{img_size}.json"
    if (arr_path.exists() and lab_path.exists() and meta_path.exists()
            and json.loads(meta_path.read_text()).get("n") == n):
        print(f"Cache hit: {tag} ({n} images)")
        return arr_path, lab_path, base.classes
    print(f"Building {tag} cache ({n} images, one-time CPU cost) ...", flush=True)
    arr = np.memmap(str(arr_path), dtype=np.uint8, mode="w+",
                    shape=(n, 3, img_size, img_size))
    labels = np.empty(n, dtype=np.int64)

    def job(i):
        path, y = base.samples[i]
        with Image.open(path) as im:
            arr[i] = np.asarray(
                im.convert("RGB").resize((img_size, img_size))).transpose(2, 0, 1)
        labels[i] = y
        return i

    with ThreadPoolExecutor(max_workers=8) as ex:
        for done, _ in enumerate(ex.map(job, range(n)), 1):
            if done % 10000 == 0 or done == n:
                print(f"  cache {tag}: {done}/{n}", flush=True)
    arr.flush()
    np.save(str(lab_path), labels)
    meta_path.write_text(json.dumps({"n": n, "classes": base.classes,
                                     "shape": [n, 3, img_size, img_size]}))
    return arr_path, lab_path, base.classes


def run_epoch(model, loader, device, criterion, optimizer, scaler, train: bool,
              prog=None, prog_every=25, gpu_aug=False, norm_stats=None):
    import torch
    correct, total, loss_sum = 0, 0, 0.0
    steps = len(loader)
    mean_t, std_t = norm_stats or (None, None)
    for i, (x, y) in enumerate(loader):
        if x.dtype == torch.uint8:
            # cached path: decode-free bytes -> GPU, augment+normalize on GPU
            x = x.to(device, non_blocking=True).float().div_(255.0)
            if train and gpu_aug:
                x = gpu_augment(x)
            x = x.sub_(mean_t).div_(std_t)
        else:
            x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                out = model(x)
                loss = criterion(out, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
        if prog is not None and train and (i + 1) % prog_every == 0:
            write_progress(running=True, stage="train", step=i + 1, steps=steps,
                           loss=round(loss_sum / total, 4),
                           acc=round(correct / total, 4),
                           **{**prog, "updated": time.time()})
    return loss_sum / total, correct / total


def freeze_early_blocks(model, n: int):
    """Freeze first n feature blocks (incl. their BatchNorm stats)."""
    for i, block in enumerate(model.features):
        if i < n:
            block.eval()
            for p in block.parameters():
                p.requires_grad_(False)


def main():
    ap = argparse.ArgumentParser(description="Train AgriVision MobileNetV2 (PyTorch)")
    ap.add_argument("--data", default="./data",
                    help="Folder with class subfolders (flat or pre-split train/valid)")
    ap.add_argument("--from-kaggle", choices=sorted(PRESETS),
                    help="Auto-download preset before training (needs Kaggle CLI)")
    ap.add_argument("--kaggle-target", default="./data_raw",
                    help="Download folder for --from-kaggle")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--split", type=float, default=0.2,
                    help="Validation fraction when --data is flat (no pre-split)")
    ap.add_argument("--workers", type=int, default=2,
                    help="CPU DataLoader workers (read-only cache, near-zero CPU)")
    ap.add_argument("--freeze-blocks", type=int, default=12,
                    help="Freeze first N MobileNetV2 feature blocks (of 19)")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="Mixed precision on CUDA (--no-amp to disable)")
    ap.add_argument("--gpu-aug", action=argparse.BooleanOptionalAction, default=True,
                    help="Augmentation math on GPU (--no-gpu-aug for CPU PIL aug)")
    ap.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                    help="One-time memmap decode cache (--no-cache to decode every epoch)")
    ap.add_argument("--resume", action="store_true",
                    help="Start from model/plant_disease_mobilenet.pt instead of ImageNet")
    args = ap.parse_args()

    data = Path(args.kaggle_target) if args.from_kaggle else Path(args.data)
    if args.from_kaggle:
        download_from_kaggle(args.from_kaggle, data)
    assert data.exists(), f"Data folder not found: {data}"

    import torch
    import torch.nn as nn  # noqa: F401 (kept for clarity / future head edits)
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets as tv_datasets, transforms
    from sklearn.metrics import classification_report, confusion_matrix
    from .model_handler import build_mobilenet, IMAGENET_MEAN, IMAGENET_STD

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu_pipe = args.gpu_aug and args.cache and device.type == "cuda"
    print(f"Device: {device} | gpu-aug: {use_gpu_pipe and 'ON (RTX)' or 'off'}",
          flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | workers: {args.workers} (CPU) | amp: {args.amp}", flush=True)

    train_dir, val_dir, test_dir = resolve_splits(data)
    if val_dir is not None:
        print(f"Pre-split detected: train={train_dir} val={val_dir}"
              + (f" test={test_dir}" if test_dir else ""))
    else:
        print(f"Flat layout - using {args.split:.0%} validation split.")

    if use_gpu_pipe:
        # GPU pipeline: one-time decode cache, uint8 straight to RTX
        cache_dir = data / ".cache"
        if val_dir is None:
            sys.exit("--cache needs a pre-split dataset (train/valid folders)")
        tr_arr, tr_lab, labels = build_cache(train_dir, args.img, cache_dir, "train")
        va_arr, va_lab, _ = build_cache(val_dir, args.img, cache_dir, "valid")
        train_ds, val_ds = CachedSplit(tr_arr, tr_lab), CachedSplit(va_arr, va_lab)
        test_ds = None
        if test_dir is not None:
            te_arr, te_lab, _ = build_cache(test_dir, args.img, cache_dir, "test")
            test_ds = CachedSplit(te_arr, te_lab)
    else:
        # Classic pipeline: CPU decode + PIL augmentation
        train_tf = transforms.Compose([
            transforms.Resize((args.img, args.img)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((args.img, args.img)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        if val_dir is not None:
            train_ds = tv_datasets.ImageFolder(str(train_dir), transform=train_tf)
            val_ds = tv_datasets.ImageFolder(str(val_dir), transform=eval_tf)
            labels = train_ds.classes
            test_ds = (tv_datasets.ImageFolder(str(test_dir), transform=eval_tf)
                       if test_dir is not None else None)
        else:
            full = tv_datasets.ImageFolder(str(train_dir))
            n_val = int(len(full) * args.split)
            gen = torch.Generator().manual_seed(42)
            perm = torch.randperm(len(full), generator=gen).tolist()
            train_ds = Subset(tv_datasets.ImageFolder(str(train_dir), transform=train_tf),
                              perm[n_val:])
            val_ds = Subset(tv_datasets.ImageFolder(str(train_dir), transform=eval_tf),
                            perm[:n_val])
            labels = full.classes
            test_ds = None
    print(f"Classes ({len(labels)}): {labels}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)
    test_loader = (DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=pin)
                   if test_ds is not None else None)

    model = build_mobilenet(len(labels)).to(device)
    freeze_early_blocks(model, args.freeze_blocks)
    best_acc, best_state, bad_epochs = 0.0, None, 0
    resumed_best_acc, resumed_best_epoch = 0.0, 0
    if args.resume:
        ckpt_path = MODEL_DIR / "plant_disease_mobilenet.pt"
        assert ckpt_path.exists(), "Nothing to resume - no .pt weights found."
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        if len(ckpt.get("classes", [])) != len(labels):
            sys.exit("Resume failed: checkpoint class count differs from data.")
        model.load_state_dict(ckpt["state_dict"])
        try:
            best_acc = float(json.loads(
                (MODEL_DIR / "metrics.json").read_text())["val_accuracy"])
        except Exception:
            best_acc = 0.0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"Resumed weights (prev best val_acc={best_acc:.4f})", flush=True)
        resumed_best_acc = best_acc
        try:
            resumed_best_epoch = int(json.loads(
                (MODEL_DIR / "metrics.json").read_text()).get("best_epoch", 0))
        except Exception:
            resumed_best_epoch = 0
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable) / 1e6:.2f}M", flush=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    norm_stats = _norm_tensors(device, IMAGENET_MEAN, IMAGENET_STD)

    hist = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    bad_epochs = 0
    write_progress(running=True, done=False, stage="starting", epoch=0,
                   epochs=args.epochs, step=0, steps=len(train_loader),
                   device=str(device), started=time.time(), updated=time.time())
    epoch_times = []
    for epoch in range(args.epochs):
        t0 = time.time()
        prog = {"epoch": epoch + 1, "epochs": args.epochs,
                "device": str(device), "updated": time.time()}
        model.train()
        freeze_early_blocks(model, args.freeze_blocks)  # keep frozen BNs in eval
        tr_loss, tr_acc = run_epoch(model, train_loader, device,
                                    criterion, optimizer, scaler, True,
                                    prog=prog, gpu_aug=use_gpu_pipe,
                                    norm_stats=norm_stats)
        model.eval()
        write_progress(running=True, stage="validate", **prog)
        va_loss, va_acc = run_epoch(model, val_loader, device,
                                    criterion, optimizer, scaler, False,
                                    gpu_aug=False, norm_stats=norm_stats)
        scheduler.step(va_loss)
        hist["accuracy"].append(tr_acc)
        hist["val_accuracy"].append(va_acc)
        hist["loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        epoch_times.append(time.time() - t0)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        eta = avg_epoch * (args.epochs - epoch - 1)
        print(f"epoch {epoch + 1}/{args.epochs} "
              f"loss={tr_loss:.4f} acc={tr_acc:.4f} "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        write_progress(running=True, stage="epoch-done", step=len(train_loader),
                       steps=len(train_loader), train_loss=round(tr_loss, 4),
                       train_acc=round(tr_acc, 4), val_loss=round(va_loss, 4),
                       val_acc=round(va_acc, 4), best_acc=round(best_acc, 4),
                       epoch_time_s=round(epoch_times[-1], 1),
                       eta_s=round(eta, 0), **prog)
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
            torch.save({"state_dict": best_state, "classes": labels,
                        "img": args.img}, MODEL_DIR / "plant_disease_mobilenet.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= 4:
                print(f"Early stopping (best val_acc={best_acc:.4f})", flush=True)
                break

    # ---- evaluation (test set if available, else validation) ----
    model.load_state_dict(best_state)
    model.eval()
    target_loader = test_loader if test_loader is not None else val_loader
    split_name = "test" if test_loader is not None else "val"
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in target_loader:
            if x.dtype == torch.uint8:
                x = x.to(device).float().div_(255.0)
                mean_t, std_t = norm_stats
                x = x.sub_(mean_t).div_(std_t)
            else:
                x = x.to(device)
            out = model(x)
            y_true.extend(y.tolist())
            y_pred.extend(out.argmax(1).cpu().tolist())
    report = classification_report(y_true, y_pred, target_names=labels,
                                   output_dict=True, zero_division=0)
    test_acc = report["accuracy"]
    improved = best_acc > resumed_best_acc
    if improved:
        final_best_epoch = hist["val_accuracy"].index(max(hist["val_accuracy"])) + 1
    else:
        final_best_epoch = resumed_best_epoch or len(hist["loss"])
    if args.resume and not improved:
        # Keep previous best artifacts untouched: weights, metrics, plots,
        # labels on disk already describe the better model.
        print(f"\nNo improvement over resumed best ({resumed_best_acc:.4f}) - "
              f"kept existing model files.", flush=True)
        write_progress(running=False, done=True, stage="done",
                       test_accuracy=round(test_acc, 4),
                       val_accuracy=round(best_acc, 4), updated=time.time())
        return

    class_metrics = [{
        "class": c,
        "precision": round(report[c]["precision"], 3),
        "recall": round(report[c]["recall"], 3),
        "f1": round(report[c]["f1-score"], 3),
        "support": int(report[c]["support"]),
    } for c in labels]

    metrics = {
        "model": f"MobileNetV2 transfer-learning (PyTorch, {device.type}"
                 + ("+amp" if scaler.is_enabled() else "") + f"), {args.img}x{args.img}, "
                 f"Adam {args.lr}, epochs={len(hist['loss'])}",
        "framework": "pytorch",
        "dataset": str(data),
        "eval_split": split_name,
        "val_accuracy": round(best_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "best_epoch": int(final_best_epoch),
        "class_metrics": class_metrics,
        "macro_avg": {"precision": round(report["macro avg"]["precision"], 4),
                      "recall": round(report["macro avg"]["recall"], 4),
                      "f1": round(report["macro avg"]["f1-score"], 4)},
        "weighted_avg": {"precision": round(report["weighted avg"]["precision"], 4),
                         "recall": round(report["weighted avg"]["recall"], 4),
                         "f1": round(report["weighted avg"]["f1-score"], 4)},
        "note": "Real training metrics from backend/train.py",
    }
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(MODEL_DIR / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist["accuracy"], label="train")
    ax[0].plot(hist["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(hist["loss"], label="train")
    ax[1].plot(hist["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(MODEL_DIR / "training_curves.png", dpi=120)

    cm = confusion_matrix(y_true, y_pred)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.35), max(6, n * 0.3)))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({split_name} acc {test_acc:.3f})")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=120)

    print(f"\nDone. device={device} best_val={best_acc:.4f} {split_name}={test_acc:.4f}")
    print("Saved weights (.pt) + metrics.json + plots in ./model/ - restart the app to use them.")
    write_progress(running=False, done=True, stage="done",
                   test_accuracy=round(test_acc, 4),
                   val_accuracy=round(best_acc, 4), updated=time.time())


if __name__ == "__main__":
    main()
