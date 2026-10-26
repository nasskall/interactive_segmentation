"""Project exports: COCO JSON, PNG masks, YOLO and CSV.

Each exporter takes an ``ExportData`` (the project, its classes, the items to
export and their annotations) and writes into an open ``zipfile.ZipFile``,
except CSV, which returns text. Images are copied byte-for-byte from the
item store.

Class ids: COCO and PNG masks number classes 1..N in the project's class
order (0 is background in masks); YOLO numbers them 0..N-1.

Files are named ``<item id>_<item name>`` so names stay unique; change
detection pairs get ``_before`` / ``_after`` suffixes.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from server.labelling import geometry
from server.labelling.storage import ImageStore

FORMATS = ('coco', 'masks', 'yolo', 'csv')

# Which export formats make sense for each task type.
TASK_FORMATS = {
    'classification': ('csv',),
    'detection': ('coco', 'yolo', 'csv'),
    'semantic_segmentation': ('coco', 'masks', 'yolo', 'csv'),
    'instance_segmentation': ('coco', 'masks', 'yolo', 'csv'),
    'change_detection': ('coco', 'masks', 'csv'),
}


@dataclass
class ExportData:
    project: dict
    classes: list[dict]            # in project order
    items: list[dict]              # each with 'files' decoded to {role: filename}
    annotations: dict[int, list[dict]] = field(default_factory=dict)  # item id -> rows, by id
    users: dict[int, str] = field(default_factory=dict)               # user id -> display name

    def class_index(self, start: int) -> dict[int, int]:
        return {c['id']: i + start for i, c in enumerate(self.classes)}


def safe_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('._') or 'item'


def item_stem(item: dict) -> str:
    return f"{item['id']}_{safe_name(item['name'])}"


def image_names(item: dict) -> dict[str, str]:
    """Role -> file name inside the export's ``images/`` folder."""
    stem = item_stem(item)
    return {role: (stem if role == 'image' else f'{stem}_{role}') + Path(f).suffix.lower()
            for role, f in item['files'].items()}


def write_images(zf: zipfile.ZipFile, data: ExportData, store: ImageStore) -> None:
    for item in data.items:
        for role, name in image_names(item).items():
            # Image formats are already compressed.
            zf.write(store.path(item['id'], item['files'][role]), f'images/{name}',
                     compress_type=zipfile.ZIP_STORED)


# ── geometry helpers ──────────────────────────────────────────────────────

def rasterize(geom: dict, width: int, height: int) -> np.ndarray | None:
    """Boolean mask of a box, polygon or mask label; None for tags."""
    kind = geom['type']
    if kind == 'mask':
        return geometry.rle_decode(geom['rle'])
    out = np.zeros((height, width), np.uint8)
    if kind == 'polygon':
        pts = np.round(np.asarray(geom['points'], np.float64)).astype(np.int32)
        cv2.fillPoly(out, [pts], 1)
    elif kind == 'box':
        x, y, w, h = geom['bbox']
        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = int(round(x + w)), int(round(y + h))
        out[y0:y1, x0:x1] = 1
    else:
        return None
    return out.astype(bool)


def polygon_area(points: list[list[float]]) -> float:
    p = np.asarray(points, np.float64)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def bbox_and_area(geom: dict) -> tuple[list[float], float] | None:
    kind = geom['type']
    if kind == 'box':
        x, y, w, h = geom['bbox']
        return [x, y, w, h], float(w * h)
    if kind == 'polygon':
        p = np.asarray(geom['points'], np.float64)
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], polygon_area(geom['points'])
    if kind == 'mask':
        mask = geometry.rle_decode(geom['rle'])
        return [float(v) for v in geometry.mask_to_box(mask)], float(sum(geom['rle']['counts'][1::2]))
    return None


def mask_polygons(mask: np.ndarray) -> list[list[list[int]]]:
    """Outer outline of every region of a mask (YOLO takes one polygon per line)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        c = cv2.approxPolyDP(c, 1.0, True)
        if len(c) >= 3:
            polys.append([[int(x), int(y)] for x, y in c[:, 0]])
    return polys


# ── COCO ──────────────────────────────────────────────────────────────────

def model_fields(a: dict) -> tuple[str, str]:
    """(model, sha256) columns for an annotation: the model type and
    checkpoint that proposed it, members joined with '+' for an ensemble;
    empty for manual annotations."""
    m = a.get('model')
    if not m:
        return '', ''
    m = json.loads(m) if isinstance(m, str) else m
    parts = m.get('members') or [m]
    name = ' + '.join(f"{p['model_type']} {p.get('checkpoint') or ''}".strip() for p in parts)
    if m.get('members'):
        name = f"Ensemble ({m.get('fusion')}): {name}"
    return name, '+'.join(p.get('sha256') or '' for p in parts)


def coco_dict(data: ExportData) -> dict:
    cat_id = data.class_index(1)
    images, annotations = [], []
    for item in data.items:
        names = image_names(item)
        entry = {'id': item['id'], 'width': item['width'], 'height': item['height'],
                 'file_name': names.get('image') or names['after']}
        if item['kind'] == 'pair':
            entry['before_file_name'] = names['before']
        images.append(entry)
        for a in data.annotations.get(item['id'], []):
            geom = a['geometry']
            ba = bbox_and_area(geom)
            if ba is None:
                continue
            bbox, area = ba
            if geom['type'] == 'mask':
                segmentation = geom['rle']
            elif geom['type'] == 'polygon':
                segmentation = [[v for pt in geom['points'] for v in pt]]
            else:
                segmentation = []
            model = a.get('model')
            annotations.append({'id': a['id'], 'image_id': item['id'],
                                'category_id': cat_id[a['class_id']],
                                'segmentation': segmentation, 'bbox': bbox, 'area': area,
                                'iscrowd': 0,
                                # Provenance: who labelled it and, if assisted, with which model.
                                'attributes': {
                                    'source': a['source'],
                                    'labeller': data.users.get(a['labeller_id'], ''),
                                    'model': json.loads(model) if isinstance(model, str) else model,
                                }})
    return {
        'info': {'description': data.project['name'],
                 'task_type': data.project['task_type']},
        'images': images,
        'annotations': annotations,
        'categories': [{'id': cat_id[c['id']], 'name': c['name'], 'supercategory': '',
                        'color': c['color']} for c in data.classes],
    }


def export_coco(zf: zipfile.ZipFile, data: ExportData) -> None:
    zf.writestr('annotations.json', json.dumps(coco_dict(data)))


# ── PNG masks ─────────────────────────────────────────────────────────────

def _hex_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def class_map(data: ExportData, item: dict) -> np.ndarray:
    """uint8 class-index image; later annotations paint over earlier ones."""
    idx = data.class_index(1)
    out = np.zeros((item['height'], item['width']), np.uint8)
    for a in data.annotations.get(item['id'], []):
        m = rasterize(a['geometry'], item['width'], item['height'])
        if m is not None:
            out[m] = idx[a['class_id']]
    return out


def instance_map(data: ExportData, item: dict) -> np.ndarray:
    """uint16 image numbering this item's objects 1..K in annotation order."""
    out = np.zeros((item['height'], item['width']), np.uint16)
    k = 0
    for a in data.annotations.get(item['id'], []):
        m = rasterize(a['geometry'], item['width'], item['height'])
        if m is not None:
            k += 1
            out[m] = k
    return out


def _png(array: np.ndarray, palette: list[int] | None = None) -> bytes:
    buf = io.BytesIO()
    if palette is None:
        ok, enc = cv2.imencode('.png', array)
        return enc.tobytes()
    img = Image.fromarray(array, mode='P')
    img.putpalette(palette)
    img.save(buf, format='PNG')
    return buf.getvalue()


def export_masks(zf: zipfile.ZipFile, data: ExportData) -> None:
    if len(data.classes) > 255:
        raise ValueError('PNG class masks hold at most 255 classes')
    # Palette PNGs keep raw class indices but display in the class colours.
    palette = [0, 0, 0]
    for c in data.classes:
        palette += list(_hex_rgb(c['color']))
    palette += [0, 0, 0] * (256 - len(palette) // 3)
    instances = data.project['task_type'] == 'instance_segmentation'
    for item in data.items:
        stem = item_stem(item)
        zf.writestr(f'masks/{stem}.png', _png(class_map(data, item), palette))
        if instances:
            zf.writestr(f'instances/{stem}.png', _png(instance_map(data, item)))
    rows = ['index,name,color', '0,background,#000000']
    rows += [f"{i},{c['name']},{c['color']}" for i, c in enumerate(data.classes, 1)]
    zf.writestr('classes.csv', '\n'.join(rows) + '\n')


# ── YOLO ──────────────────────────────────────────────────────────────────

def yolo_lines(data: ExportData, item: dict) -> list[str]:
    idx = data.class_index(0)
    w, h = item['width'], item['height']
    fmt = lambda v: f'{min(max(v, 0.0), 1.0):.6f}'  # noqa: E731
    lines = []
    for a in data.annotations.get(item['id'], []):
        geom, c = a['geometry'], idx[a['class_id']]
        if geom['type'] == 'box':
            x, y, bw, bh = geom['bbox']
            lines.append(' '.join([str(c), fmt((x + bw / 2) / w), fmt((y + bh / 2) / h),
                                   fmt(bw / w), fmt(bh / h)]))
            continue
        if geom['type'] == 'polygon':
            polys = [geom['points']]
        elif geom['type'] == 'mask':
            polys = mask_polygons(geometry.rle_decode(geom['rle']))
        else:
            continue
        for poly in polys:
            coords = [fmt(v / s) for x, y in poly for v, s in ((x, w), (y, h))]
            lines.append(' '.join([str(c), *coords]))
    return lines


def export_yolo(zf: zipfile.ZipFile, data: ExportData) -> None:
    for item in data.items:
        lines = yolo_lines(data, item)
        zf.writestr(f'labels/{item_stem(item)}.txt', '\n'.join(lines) + ('\n' if lines else ''))
    names = '\n'.join(f'  {i}: {json.dumps(c["name"])}' for i, c in enumerate(data.classes))
    zf.writestr('data.yaml', f'path: .\ntrain: images\nval: images\nnames:\n{names}\n')


# ── CSV ───────────────────────────────────────────────────────────────────

CSV_COLUMNS = ['item_id', 'item_name', 'files', 'width', 'height', 'status', 'assignee',
               'annotation_id', 'class', 'label_type', 'labeller', 'source',
               'model', 'model_sha256', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'area', 'created_at', 'updated_at']


def export_csv(data: ExportData) -> str:
    """One row per annotation; items without annotations get one row with
    the annotation columns empty."""
    classes = {c['id']: c for c in data.classes}
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator='\n')
    writer.writerow(CSV_COLUMNS)
    for item in data.items:
        base = [item['id'], item['name'], ';'.join(image_names(item).values()),
                item['width'], item['height'], item['status'],
                data.users.get(item['assignee_id'], '')]
        anns = data.annotations.get(item['id'], [])
        if not anns:
            writer.writerow(base + [''] * (len(CSV_COLUMNS) - len(base)))
        for a in anns:
            cls = classes[a['class_id']]
            ba = bbox_and_area(a['geometry'])
            box = [round(v, 2) for v in ba[0]] if ba else [''] * 4
            area = round(ba[1], 2) if ba else ''
            writer.writerow(base + [a['id'], cls['name'], cls['label_type'],
                                    data.users.get(a['labeller_id'], ''), a['source'],
                                    *model_fields(a), *box, area, a['created_at'], a['updated_at']])
    return buf.getvalue()


EXPORTERS = {'coco': export_coco, 'masks': export_masks, 'yolo': export_yolo}


def write_zip(path: Path, fmt: str, data: ExportData, store: ImageStore,
              include_images: bool) -> None:
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
        EXPORTERS[fmt](zf, data)
        if include_images:
            write_images(zf, data, store)
