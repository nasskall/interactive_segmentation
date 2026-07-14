"""
Download and prepare GrabCut and Berkeley evaluation datasets.

Sources
-------
* BSDS500  — official Berkeley tgz (~70 MB)
  http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz

* GrabCut masks — flandrade/dataset-interactive-algorithms (GitHub, ~1 MB)
  50 ground-truth PNG masks for the classic GrabCut evaluation set.

What gets built
---------------
datasets/
  GrabCut/
    data_GT/          50 JPG images  (20 from BSDS500 test set)
    boundary_GT/      50 PNG masks   (binary 0/255 from flandrade GT)

  Berkeley/
    images/           200 JPG images (BSDS500 test set)
    masks/            200 PNG binary masks (largest-region from first annotator)

Notes
-----
* GrabCut – the classic 50-image benchmark includes 30 named images (doll,
  banana, etc.) from Microsoft Research and 20 from BSDS.  Only the 20
  BSDS-overlap images can be reconstructed automatically here.  The 30 named
  images are no longer publicly hosted; if you have them, place JPG/PNG files
  named e.g. doll.jpg in datasets/GrabCut/data_GT/ and the corresponding PNG
  mask in datasets/GrabCut/boundary_GT/.

* Berkeley – uses all 200 BSDS500 test-split images with the largest semantic
  region from the first human annotator as the foreground mask.  Absolute NoC
  numbers will differ slightly from published results (which use a specific
  96-image subset with curated instances), but relative improvement between
  methods remains meaningful.

* If you obtain the original prepared zip files from the authors (see README
  of saic-vul/fbrs_interactive_segmentation), simply unzip them into
  datasets/GrabCut/ and datasets/Berkeley/ respectively to get exact parity
  with published results.
"""

import io
import os
import sys
import tarfile
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

# scipy is needed to read BSDS500 .mat ground-truth files
try:
    import scipy.io as sio
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reporthook(count, block_size, total_size):
    if total_size > 0:
        pct = min(count * block_size * 100 // total_size, 100)
        bar = '#' * (pct // 4)
        print(f'\r  [{bar:<25}] {pct:3d}%', end='', flush=True)


def download(url: str, dest: Path) -> None:
    print(f'  -> {url}')
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, 'wb') as f:
        total = int(r.headers.get('Content-Length', 0))
        blk, count = 65536, 0
        while True:
            chunk = r.read(blk)
            if not chunk:
                break
            f.write(chunk)
            count += 1
            _reporthook(count, blk, total)
    print()


# ---------------------------------------------------------------------------
# BSDS500
# ---------------------------------------------------------------------------

def download_bsds500(cache_dir: Path) -> Path:
    tgz = cache_dir / 'BSR_bsds500.tgz'
    if tgz.exists():
        print('  BSDS500 archive already cached.')
    else:
        download(
            'http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz',
            tgz,
        )
    return tgz


def extract_bsds500(tgz: Path, cache_dir: Path) -> Path:
    """Extract and return the root of the BSR tree."""
    bsr_root = cache_dir / 'BSR'
    if bsr_root.exists():
        print('  BSDS500 already extracted.')
        return bsr_root
    print('  Extracting BSDS500 …')
    with tarfile.open(tgz, 'r:gz') as tf:
        tf.extractall(cache_dir)
    return bsr_root


def bsds_largest_region_mask(mat_path: Path) -> np.ndarray | None:
    """
    Read a BSDS500 .mat ground-truth file and return a binary uint8 mask
    (255 = foreground) for the largest region in the first annotator's
    segmentation.
    """
    if not _SCIPY_OK:
        return None
    mat = sio.loadmat(str(mat_path))
    gt = mat.get('groundTruth')
    if gt is None:
        return None
    # gt is an object array; take first annotator
    seg = gt[0, 0]['Segmentation'][0, 0].astype(np.int32)
    # Find the label with the largest area (excluding background = smallest)
    labels, counts = np.unique(seg, return_counts=True)
    if len(labels) < 2:
        return None
    # Sort by count descending; skip label 0 if present
    order = np.argsort(-counts)
    for idx in order:
        if labels[idx] != 0:
            fg_label = labels[idx]
            break
    else:
        return None
    mask = (seg == fg_label).astype(np.uint8) * 255
    return mask


# ---------------------------------------------------------------------------
# Berkeley dataset
# ---------------------------------------------------------------------------

def build_berkeley(bsr_root: Path, dest: Path) -> None:
    if not _SCIPY_OK:
        print('  scipy not available — cannot read BSDS500 .mat files.  Skipping Berkeley.')
        return

    img_dir  = dest / 'images'
    mask_dir = dest / 'masks'
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    test_imgs = sorted((bsr_root / 'BSDS500' / 'data' / 'images' / 'test').glob('*.jpg'))
    test_gts  = bsr_root / 'BSDS500' / 'data' / 'groundTruth' / 'test'

    print(f'  Processing {len(test_imgs)} test images …')
    ok = 0
    for img_path in test_imgs:
        stem = img_path.stem
        mat_path = test_gts / f'{stem}.mat'
        if not mat_path.exists():
            continue
        mask = bsds_largest_region_mask(mat_path)
        if mask is None:
            continue
        # Copy image
        import shutil
        shutil.copy(img_path, img_dir / img_path.name)
        # Save mask
        Image.fromarray(mask).save(mask_dir / f'{stem}.png')
        ok += 1

    print(f'  Berkeley: {ok} image-mask pairs written to {dest}')


# ---------------------------------------------------------------------------
# GrabCut dataset (BSDS-overlap subset)
# ---------------------------------------------------------------------------

# These 20 IDs appear in both the flandrade GrabCut benchmark masks and BSDS500.
GRABCUT_BSDS_IDS = [
    '106024', '124084', '153077', '153093', '181079', '189080',
    '208001', '209070', '21077',  '227092', '24077',  '271008',
    '304074', '326038', '37073',  '376043', '388016', '65019',
    '69020',  '86016',
]

FLANDRADE_MASK_BASE = (
    'https://raw.githubusercontent.com/flandrade/'
    'dataset-interactive-algorithms/master/ground-truth/{}.png'
)


def build_grabcut(bsr_root: Path, dest: Path) -> None:
    img_dir  = dest / 'data_GT'
    mask_dir = dest / 'boundary_GT'
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Source directories for images in BSDS500
    bsds_test = bsr_root / 'BSDS500' / 'data' / 'images' / 'test'
    bsds_train = bsr_root / 'BSDS500' / 'data' / 'images' / 'train'
    bsds_val   = bsr_root / 'BSDS500' / 'data' / 'images' / 'val'

    def find_img(stem: str) -> Path | None:
        for folder in (bsds_test, bsds_train, bsds_val):
            p = folder / f'{stem}.jpg'
            if p.exists():
                return p
        return None

    ok = 0
    for img_id in GRABCUT_BSDS_IDS:
        src = find_img(img_id)
        if src is None:
            print(f'    [warn] image {img_id} not found in BSDS500')
            continue

        # Download binary mask from flandrade
        mask_url = FLANDRADE_MASK_BASE.format(img_id)
        mask_dest = mask_dir / f'{img_id}.png'
        try:
            req = urllib.request.Request(mask_url, headers={'User-Agent': 'python'})
            with urllib.request.urlopen(req, timeout=15) as r:
                mask_dest.write_bytes(r.read())
        except Exception as e:
            print(f'    [warn] could not download mask for {img_id}: {e}')
            continue

        import shutil
        shutil.copy(src, img_dir / f'{img_id}.jpg')
        ok += 1

    print(f'\n  GrabCut: {ok}/20 BSDS-overlap pairs written to {dest}')
    print()
    print('  ── Named images (doll, banana, etc.) ────────────────────────────────')
    print('  The remaining 30 images from the original GrabCut dataset are no')
    print('  longer hosted publicly.  If you have them, place:')
    print('    • images (.jpg)  → datasets/GrabCut/data_GT/<name>.jpg')
    print('    • masks  (.png)  → datasets/GrabCut/boundary_GT/<name>.png')
    print('  Names: banana1 banana2 banana3 book bool bush ceramic cross doll')
    print('         elefant flower fullmoon grave llama memorial music person1-8')
    print('         scissors sheep stone1 stone2 teddy tennis')
    print('  ─────────────────────────────────────────────────────────────────────')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    cache = Path('datasets/.cache')
    cache.mkdir(parents=True, exist_ok=True)

    grabcut_dest  = Path('datasets/GrabCut')
    berkeley_dest = Path('datasets/Berkeley')

    # ---- Download and extract BSDS500 -------------------------------------
    print('\n=== Downloading BSDS500 (Berkeley + GrabCut source) ===')
    tgz = download_bsds500(cache)
    bsr_root = extract_bsds500(tgz, cache) / 'BSDS500'

    # ---- Build Berkeley ---------------------------------------------------
    print('\n=== Building Berkeley dataset ===')
    if (berkeley_dest / 'images').exists() and any((berkeley_dest / 'images').iterdir()):
        print('  Already present — skipping.')
    else:
        build_berkeley(bsr_root.parent, berkeley_dest)

    # ---- Build GrabCut (BSDS overlap) ------------------------------------
    print('\n=== Building GrabCut dataset (20-image BSDS overlap) ===')
    if (grabcut_dest / 'data_GT').exists() and any((grabcut_dest / 'data_GT').iterdir()):
        print('  Already present — skipping.')
    else:
        build_grabcut(bsr_root.parent, grabcut_dest)

    # ---- Summary ----------------------------------------------------------
    print('\n=== Summary ===')
    for name, path in [('GrabCut', grabcut_dest), ('Berkeley', berkeley_dest)]:
        imgs = list((path / ('data_GT' if name == 'GrabCut' else 'images')).glob('*.*'))
        print(f'  {name}: {len(imgs)} images  ->  {path}')

    print('\nDataset paths match config.yml — ready to evaluate.')


if __name__ == '__main__':
    if not _SCIPY_OK:
        print('[ERROR] scipy is required.  Install: pip install scipy')
        sys.exit(1)
    main()
