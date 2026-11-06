"""
Build dermoscopy test sets that public medical models have not trained on.

IMIS-Net trained on 90% of ISIC 2016/2017/2018 Task 1 and PH2, and MedSAM
pooled dermoscopy with a random split, so both have likely seen the ISIC 2018
test images. This draws instead from IMA++ (Abhishek, Kawahara, Hamarneh;
ISIC Archive images with expert masks, doi:10.5281/zenodo.14201692,
CC BY-NC-ND 4.0) and drops every image that is in an ISIC 2016, 2017 or 2018
challenge set (Task 1-2 and Task 3, i.e. HAM10000), or that shares a
lesion_id or patient_id with one. PH2 is not in the ISIC Archive. Most
images carry no lesion_id or patient_id, so that second check is partial.

Two benchmark folders (the click_benchmark.py layout, split "test"):

  --out-test   --n-test images drawn (seeded) from the clean pool; the mask
               is the STAPLE consensus where IMA++ has several annotators,
               otherwise the one annotator's mask
  --out-multi  every clean image with masks from two or more annotators;
               same reference mask, plus each annotator's own mask under
               masks_annotators/ for the inter-annotator ceiling

Images are fetched from the ISIC Archive API and kept full-size under
<raw>/images; the benchmark copies are shrunk as in build_lesion_benchmark.py.
These are derived copies of NoDerivatives-licensed masks: use them for
evaluation only and never redistribute them.

Run from repo root, with img_metadata.csv, seg_metadata.csv and segs.zip
from the Zenodo record in datasets/imapp/raw:
    python tools/build_clean_isic.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
API = 'https://api.isic-archive.com/api/v2'
EXCLUDED_COLLECTIONS = {   # ISIC Archive collection id -> name
    74: 'Challenge 2016: Training', 61: 'Challenge 2016: Test',
    97: 'Collection for ISBI 2016: 100 Lesion Classification',
    60: 'Challenge 2017: Training', 71: 'Challenge 2017: Validation', 69: 'Challenge 2017: Test',
    63: 'Challenge 2018: Task 1-2: Training', 62: 'Challenge 2018: Task 1-2: Validation',
    64: 'Challenge 2018: Task 1-2: Test',
    66: 'Challenge 2018: Task 3: Training', 73: 'Challenge 2018: Task 3: Validation',
    67: 'Challenge 2018: Task 3: Test',
}


def get_json(url: str, tries: int = 4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def challenge_images(cache: Path) -> dict[str, dict]:
    """isic_id -> {collections, lesion_id, patient_id} for every image in
    EXCLUDED_COLLECTIONS, cached so the list stays fixed once built."""
    if cache.exists():
        return json.loads(cache.read_text())
    out: dict[str, dict] = {}
    for cid in EXCLUDED_COLLECTIONS:
        url = f'{API}/images/search/?collections={cid}&limit=100'
        while url:
            page = get_json(url)
            for im in page['results']:
                cl = im.get('metadata', {}).get('clinical', {})
                e = out.setdefault(im['isic_id'], {'collections': [], 'lesion_id': cl.get('lesion_id'),
                                                   'patient_id': cl.get('patient_id')})
                e['collections'].append(cid)
            url = page.get('next')
        print(f'  {EXCLUDED_COLLECTIONS[cid]}: done', flush=True)
    cache.write_text(json.dumps(out, indent=0))
    return out


def fetch_image(isic_id: str, folder: Path) -> Path:
    path = folder / f'{isic_id}.jpg'
    if not path.exists():
        url = get_json(f'{API}/images/{isic_id}/')['files']['full']['url']
        tmp = path.with_suffix('.part')
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def shrink(img: np.ndarray, max_side: int, interp) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    return img if s == 1.0 else cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp)


def write_set(out: Path, names: list[str], masks_by_image: dict, zf: zipfile.ZipFile,
              raw_images: Path, max_side: int, with_annotators: bool, source: str):
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'masks').mkdir(parents=True, exist_ok=True)
    if with_annotators:
        (out / 'masks_annotators').mkdir(exist_ok=True)
    items = []
    for n, name in enumerate(names, 1):
        bgr = cv2.imread(str(fetch_image(name, raw_images)))
        segs = masks_by_image[name]
        ref = segs.get('ST') or next(v for k, v in segs.items() if k.startswith('A'))
        gt = cv2.imdecode(np.frombuffer(zf.read(ref), np.uint8), cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None or bgr.shape[:2] != gt.shape:
            sys.exit(f'{name}: unreadable image or mask, or their sizes differ')
        small = shrink(bgr, max_side, cv2.INTER_AREA)
        size = (small.shape[1], small.shape[0])
        mask = np.where(cv2.resize(gt, size, interpolation=cv2.INTER_NEAREST) > 127, 255, 0).astype(np.uint8)
        cv2.imwrite(str(out / 'images' / f'{name}.jpg'), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / 'masks' / f'{name}.png'), mask)
        item = {'name': name, 'split': 'test', 'orig_size': [gt.shape[1], gt.shape[0]],
                'size': list(size), 'reference': 'STAPLE' if 'ST' in segs else 'single annotator',
                'lesion_frac': round(float((mask > 0).mean()), 4)}
        if with_annotators:
            item['annotators'] = []
            for ann, f in sorted(segs.items()):
                if not ann.startswith('A'):
                    continue
                m = cv2.imdecode(np.frombuffer(zf.read(f), np.uint8), cv2.IMREAD_GRAYSCALE)
                m = np.where(cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) > 127, 255, 0)
                cv2.imwrite(str(out / 'masks_annotators' / f'{name}__{ann}.png'), m.astype(np.uint8))
                item['annotators'].append(ann)
        items.append(item)
        if n % 100 == 0 or n == len(names):
            print(f'  {out.name}: {n}/{len(names)}', flush=True)
    (out / 'manifest.json').write_text(json.dumps(
        {'source': source, 'max_side': max_side, 'items': items}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--raw', type=Path, default=ROOT / 'datasets' / 'imapp' / 'raw')
    ap.add_argument('--out-test', type=Path, default=ROOT / 'datasets' / 'clean_isic_test')
    ap.add_argument('--out-multi', type=Path, default=ROOT / 'datasets' / 'clean_isic_multi')
    ap.add_argument('--n-test', type=int, default=1000)
    ap.add_argument('--max-side', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    excluded = challenge_images(args.raw / 'excluded_challenge_images.json')
    lesions = {v['lesion_id'] for v in excluded.values() if v['lesion_id']}
    patients = {v['patient_id'] for v in excluded.values() if v['patient_id']}
    meta = list(csv.DictReader(open(args.raw / 'img_metadata.csv', encoding='utf-8')))
    reasons: dict[str, int] = defaultdict(int)
    clean = []
    for r in meta:
        why = ('same image' if r['isic_id'] in excluded else
               'same lesion' if r['lesion_id'] and r['lesion_id'] in lesions else
               'same patient' if r['patient_id'] and r['patient_id'] in patients else None)
        if why:
            reasons[why] += 1
        else:
            clean.append(r['isic_id'])
    clean.sort()

    masks_by_image: dict[str, dict[str, str]] = defaultdict(dict)
    for s in csv.DictReader(open(args.raw / 'seg_metadata.csv', encoding='utf-8')):
        masks_by_image[s['ISIC_id']][s['annotator']] = s['seg_filename']
    multi = [n for n in clean if sum(k.startswith('A') for k in masks_by_image[n]) >= 2]
    order = np.random.default_rng(args.seed).permutation(len(clean))
    test = sorted(clean[i] for i in order[:args.n_test])
    print(f'{len(meta)} IMA++ images; excluded {dict(reasons)}; {len(clean)} clean, '
          f'{len(multi)} with 2+ annotators; test sample {len(test)}', flush=True)

    raw_images = args.raw / 'images'
    raw_images.mkdir(exist_ok=True)
    zf = zipfile.ZipFile(args.raw / 'segs.zip')
    note = ('IMA++ (doi:10.5281/zenodo.14201692, CC BY-NC-ND 4.0) minus ISIC 2016-2018 challenge '
            'images and their lesions/patients; evaluation only, do not redistribute')
    write_set(args.out_test, test, masks_by_image, zf, raw_images, args.max_side, False,
              f'{note}; {len(test)} of {len(clean)} clean images, seed {args.seed}')
    write_set(args.out_multi, multi, masks_by_image, zf, raw_images, args.max_side, True,
              f'{note}; every clean image with 2+ annotators')
    (args.raw / 'clean_pool.json').write_text(json.dumps(
        {'excluded_counts': reasons, 'clean': clean, 'test': test, 'multi': multi}, indent=0))


if __name__ == '__main__':
    main()
