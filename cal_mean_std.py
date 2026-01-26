import os
import argparse
from collections import Counter

import torch
from torchvision import transforms
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def is_image(p: str) -> bool:
    return p.lower().endswith(IMG_EXT)


def compute_mean_std_and_stats(data_root: str, max_images: int, atol_fake_rgb: float = 1e-4):
    """
    Compute dataset mean/std (on RGB-converted images) and report key dataset stats:
    - PIL mode distribution (original, before convert)
    - fake RGB ratio: images whose R,G,B channels are (almost) identical
    - basic size stats (min/max, and a small histogram)
    """
    to_tensor = transforms.ToTensor()

    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    count = 0

    mode_counter = Counter()
    fake_rgb_count = 0

    # size stats
    min_w, min_h = 10**9, 10**9
    max_w, max_h = 0, 0
    # coarse buckets for quick insight
    # bucket by max(side): <=128, <=256, <=512, <=1024, >1024
    size_buckets = Counter()

    def bucket_size(w: int, h: int) -> str:
        m = max(w, h)
        if m <= 128:
            return "<=128"
        if m <= 256:
            return "<=256"
        if m <= 512:
            return "<=512"
        if m <= 1024:
            return "<=1024"
        return ">1024"

    for root, _, files in os.walk(data_root):
        for fn in files:
            if not is_image(fn):
                continue

            path = os.path.join(root, fn)
            try:
                img = Image.open(path)
            except Exception as e:
                print(f"[warn] failed to read {path}: {e}")
                continue

            # ----- stats BEFORE converting -----
            mode_counter[img.mode] += 1
            w, h = img.size
            min_w, min_h = min(min_w, w), min(min_h, h)
            max_w, max_h = max(max_w, w), max(max_h, h)
            size_buckets[bucket_size(w, h)] += 1

            # fake RGB check (only meaningful when original is RGB)
            if img.mode == "RGB":
                try:
                    x_raw = to_tensor(img)  # [3,H,W]
                    if torch.allclose(x_raw[0], x_raw[1], atol=atol_fake_rgb) and torch.allclose(
                        x_raw[1], x_raw[2], atol=atol_fake_rgb
                    ):
                        fake_rgb_count += 1
                except Exception as e:
                    print(f"[warn] failed fake-rgb check {path}: {e}")

            # ----- mean/std computation on RGB -----
            try:
                img_rgb = img.convert("RGB")
            except Exception as e:
                print(f"[warn] failed to convert RGB {path}: {e}")
                continue

            x = to_tensor(img_rgb)
            sum_ += x.mean(dim=(1, 2))
            sum_sq += (x * x).mean(dim=(1, 2))
            count += 1

            if max_images > 0 and count >= max_images:
                break

        if max_images > 0 and count >= max_images:
            break

    if count == 0:
        raise RuntimeError("No images found.")

    mean = sum_ / count
    std = torch.sqrt(sum_sq / count - mean * mean)

    stats = {
        "mode_counter": mode_counter,
        "fake_rgb_count": fake_rgb_count,
        "processed_count": count,
        "min_size": (min_w, min_h) if max_w > 0 else None,
        "max_size": (max_w, max_h) if max_w > 0 else None,
        "size_buckets": size_buckets,
    }
    return mean, std, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_images", type=int, default=5000, help="-1 means all")
    parser.add_argument("--fake_rgb_atol", type=float, default=1e-4, help="tolerance for fake RGB detection")
    args = parser.parse_args()

    max_images = args.max_images
    if max_images == -1:
        max_images = 0  # treat 0 as "no limit" in our code

    mean, std, stats = compute_mean_std_and_stats(
        args.data_root, max_images=max_images, atol_fake_rgb=args.fake_rgb_atol
    )

    n = stats["processed_count"]
    print(f"Processed images: {n}")
    print("Mean:", mean.tolist())
    print("Std :", std.tolist())

    print("\n[Original PIL modes]")
    mode_counter: Counter = stats["mode_counter"]
    total_modes = sum(mode_counter.values())
    for k, v in mode_counter.most_common():
        pct = 100.0 * v / max(1, total_modes)
        print(f"  {k:>6}: {v:>8}  ({pct:6.2f}%)")

    rgb_total = mode_counter.get("RGB", 0)
    if rgb_total > 0:
        fake_rgb = stats["fake_rgb_count"]
        pct = 100.0 * fake_rgb / rgb_total
        print("\n[Fake RGB among original RGB]")
        print(f"  fake_rgb: {fake_rgb} / {rgb_total} ({pct:.2f}%)")
    else:
        print("\n[Fake RGB among original RGB]")
        print("  (no original RGB images detected)")

    print("\n[Image sizes]")
    print(f"  min (w,h): {stats['min_size']}")
    print(f"  max (w,h): {stats['max_size']}")

    print("\n[Size buckets by max(side)]")
    size_buckets: Counter = stats["size_buckets"]
    for k in ["<=128", "<=256", "<=512", "<=1024", ">1024"]:
        if k in size_buckets:
            pct = 100.0 * size_buckets[k] / max(1, n)
            print(f"  {k:>6}: {size_buckets[k]:>8}  ({pct:6.2f}%)")


if __name__ == "__main__":
    main()
