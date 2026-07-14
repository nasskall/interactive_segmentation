from datetime import timedelta
from pathlib import Path

import torch
import numpy as np
import sys, os
from isegm.data.datasets import GrabCutDataset, BerkeleyDataset, DavisDataset, SBDEvaluationDataset, PascalVocDataset
from isegm.utils.serialization import load_model


def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def read_checkpoint(path):
    """torch.load, but with an error that names the actual problem.

    A failed download saved under a .pth name (an HTTP error body, an LFS
    pointer, a truncated transfer) reaches torch.load as a malformed pickle and
    surfaces as 'Unsupported operand 111' or 'could not find MARK', which tells
    the user nothing. Sniff the header first and say what is really wrong.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {path}')

    size = path.stat().st_size
    with open(path, 'rb') as f:
        head = f.read(256)

    is_zip = head.startswith(b'PK\x03\x04')     # torch >= 1.6 archive format
    is_legacy_pickle = head.startswith(b'\x80')  # older torch.save format
    if not (is_zip or is_legacy_pickle):
        preview = head[:60].decode('utf-8', 'replace').strip()
        raise RuntimeError(
            f'{path.name} is not a PyTorch checkpoint: {size} bytes, starting '
            f'with {preview!r}. This is almost always a failed download saved '
            f'under a .pth name. Re-download the file.'
        )

    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f'Could not read checkpoint {path.name} ({size} bytes): {exc}'
        ) from exc


def load_is_model(checkpoint, device, **kwargs):
    if isinstance(checkpoint, (str, Path)):
        state_dict = read_checkpoint(checkpoint)
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0], device, **kwargs)
        models = [load_single_is_model(x, device, **kwargs) for x in state_dict]

        return model, models
    else:
        return load_single_is_model(state_dict, device, **kwargs)


def load_single_is_model(state_dict, device, **kwargs):
    model = load_model(state_dict['config'], **kwargs)
    model.load_state_dict(state_dict['state_dict'], strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model


# SAM variant name → registry key used by segment_anything
_SAM_VIT_MAP = {
    'SAM - ViT-B': 'vit_b',
    'SAM - ViT-L': 'vit_l',
    'SAM - ViT-H': 'vit_h',
}

# SAM 2 variant label → config filename shipped with the sam2 package.
# The package stores configs under abbreviated names (_t, _s, _b+, _l).
_SAM2_CONFIG_MAP = {
    'SAM2 - Tiny':   'configs/sam2.1/sam2.1_hiera_t.yaml',
    'SAM2 - Small':  'configs/sam2.1/sam2.1_hiera_s.yaml',
    'SAM2 - Base+':  'configs/sam2.1/sam2.1_hiera_b+.yaml',
    'SAM2 - Large':  'configs/sam2.1/sam2.1_hiera_l.yaml',
}


def load_sam_model(checkpoint, model_type_label, device):
    """
    Load a SAM checkpoint and tag the model so the predictor factory
    can route to SAMInteractivePredictor.

    Parameters
    ----------
    checkpoint      : str  path to the SAM .pth file
    model_type_label: str  one of 'SAM - ViT-B', 'SAM - ViT-L', 'SAM - ViT-H'
    device          : str  e.g. 'cpu' or 'cuda:0'
    """
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        raise ImportError(
            "segment_anything is not installed.\n"
            "Install it with:  pip install segment-anything"
        )

    vit_key = _SAM_VIT_MAP.get(model_type_label)
    if vit_key is None:
        raise ValueError(f"Unknown SAM model type: {model_type_label!r}. "
                         f"Choose from {list(_SAM_VIT_MAP)}")

    model = sam_model_registry[vit_key](checkpoint=checkpoint)
    model.to(device=device)
    model.eval()
    model._model_type = 'sam'
    return model


def load_sam2_model(checkpoint, model_type_label, device):
    """
    Load a SAM 2 checkpoint and tag the model so the predictor factory
    can route to SAM2InteractivePredictor.

    Parameters
    ----------
    checkpoint      : str  path to the SAM2 .pt/.pth file
    model_type_label: str  one of 'SAM2 - Tiny/Small/Base+/Large'
    device          : str  e.g. 'cpu' or 'cuda:0'
    """
    try:
        from sam2.build_sam import build_sam2
    except ImportError:
        raise ImportError(
            "sam2 is not installed.\n"
            "Install it from: https://github.com/facebookresearch/sam2\n"
            "  git clone https://github.com/facebookresearch/sam2 && "
            "cd sam2 && pip install -e ."
        )

    config_file = _SAM2_CONFIG_MAP.get(model_type_label)
    if config_file is None:
        raise ValueError(f"Unknown SAM2 model type: {model_type_label!r}. "
                         f"Choose from {list(_SAM2_CONFIG_MAP)}")

    model = build_sam2(config_file, checkpoint, device=device, mode='eval')
    model._model_type = 'sam2'
    return model


def _allowlist_simpleclick_globals():
    """Permit the one non-tensor object SimpleClick checkpoints carry.

    Their saved ``config`` embeds a training-time ``loss_decode`` (a
    ``CrossEntropyLoss`` instance) inside ``head_params``. ``load_is_model``
    reads checkpoints with ``weights_only=True``, which refuses any pickled
    class unless it is allowlisted. Allowlisting just this pair keeps the safe
    loader instead of falling back to a full, arbitrary-code unpickle. The loss
    is unused at inference; only the architecture and weights matter.
    """
    from isegm.model.modeling.transformer_helper.cross_entropy_loss import (
        CrossEntropyLoss, cross_entropy,
    )
    # ``set`` is used by the serialized config for the 'specified' field.
    torch.serialization.add_safe_globals([CrossEntropyLoss, cross_entropy, set])


def load_simpleclick_model(checkpoint, device):
    """
    Load a SimpleClick checkpoint using the same mechanism as RITM.

    SimpleClick model architectures (PlainVitModel, SegformerModel, etc.)
    are now included directly in this repo under ``isegm/model/``, so no
    path manipulation is needed — the standard ``load_is_model`` call works.

    Parameters
    ----------
    checkpoint : str   path to the SimpleClick ``.pth`` file
    device     : str   e.g. 'cpu' or 'cuda:0'
    """
    _allowlist_simpleclick_globals()
    model = load_is_model(checkpoint, device, cpu_dist_maps=True)
    model._model_type = 'simpleclick'
    return model


def load_model_by_type(checkpoint, model_type_label, device):
    """
    Unified entry point: load any supported model given a UI label.

    Supported labels
    ----------------
    'RITM'              – existing RITM checkpoint
    'SimpleClick'       – SimpleClick checkpoint
    'SAM - ViT-B/L/H'
    'SAM2 - Tiny/Small/Base+/Large'
    """
    if model_type_label == 'RITM':
        model = load_is_model(checkpoint, device, cpu_dist_maps=True)
        model._model_type = 'ritm'
        return model
    elif model_type_label == 'SimpleClick':
        return load_simpleclick_model(checkpoint, device)
    elif model_type_label in _SAM_VIT_MAP:
        return load_sam_model(checkpoint, model_type_label, device)
    elif model_type_label in _SAM2_CONFIG_MAP:
        return load_sam2_model(checkpoint, model_type_label, device)
    else:
        raise ValueError(f"Unsupported model type: {model_type_label!r}")


def get_dataset(dataset_name, cfg):
    if dataset_name == 'GrabCut':
        dataset = GrabCutDataset(cfg.GRABCUT_PATH)
    elif dataset_name == 'Berkeley':
        dataset = BerkeleyDataset(cfg.BERKELEY_PATH)
    elif dataset_name == 'DAVIS':
        dataset = DavisDataset(cfg.DAVIS_PATH)
    elif dataset_name == 'SBD':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH)
    elif dataset_name == 'SBD_Train':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH, split='train')
    elif dataset_name == 'PascalVOC':
        dataset = PascalVocDataset(cfg.PASCALVOC_PATH, split='test')
    elif dataset_name == 'COCO_MVal':
        dataset = DavisDataset(cfg.COCO_MVAL_PATH)
    else:
        dataset = None

    return dataset


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array([_get_noc(iou_arr, iou_thr)
                               for iou_arr in all_ious], dtype=np.int)

        score = scores_arr.mean()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        over_max_list.append(over_max)

    return noc_list, over_max_list


def find_checkpoint(weights_folder, checkpoint_name):
    weights_folder = Path(weights_folder)
    if ':' in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(':')
        models_candidates = [x for x in weights_folder.glob(f'{model_name}*') if x.is_dir()]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith('.pth'):
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f'{checkpoint_name}*.pth'))
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]

    return str(checkpoint_path)


def get_results_table(noc_list, over_max_list, brs_type, dataset_name, mean_spc, elapsed_time,
                      n_clicks=20, model_name=None):
    table_header = (f'|{"BRS Type":^13}|{"Dataset":^11}|'
                    f'{"NoC@80%":^9}|{"NoC@85%":^9}|{"NoC@90%":^9}|'
                    f'{">="+str(n_clicks)+"@85%":^9}|{">="+str(n_clicks)+"@90%":^9}|'
                    f'{"SPC,s":^7}|{"Time":^9}|')
    row_width = len(table_header)

    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f'|{brs_type:^13}|{dataset_name:^11}|'
    table_row += f'{noc_list[0]:^9.2f}|'
    table_row += f'{noc_list[1]:^9.2f}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{noc_list[2]:^9.2f}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{over_max_list[1]:^9}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{over_max_list[2]:^9}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{mean_spc:^7.3f}|{eval_time:^9}|'

    return header, table_row


def get_config_path():
    if hasattr(sys, '_MEIPASS'):
        # If running in a PyInstaller bundle, use the _MEIPASS directory
        base_path = sys._MEIPASS
    else:
        # If running as a normal script, use the current directory
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, 'config.yml')


def get_model_path():
    if hasattr(sys, '_MEIPASS'):
        # If running in a PyInstaller bundle, use the _MEIPASS directory
        base_path = sys._MEIPASS
    else:
        # If running as a normal script, use the current directory
        base_path = os.path.abspath(".") + '/model_weights'

    return os.path.join(base_path, 'best_checkpoint_068.pth')