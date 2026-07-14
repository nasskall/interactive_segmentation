from .base import BasePredictor
from .brs import InputBRSPredictor, FeatureBRSPredictor, HRNetFeatureBRSPredictor
from .brs_functors import InputOptimizer, ScaleBiasOptimizer
from .sam_predictor import SAMInteractivePredictor
from .sam2_predictor import SAM2InteractivePredictor
from ..transforms import ZoomIn
from ...model.is_hrnet_model import HRNetModel
from ...model.modeling.pos_embed import interpolate_pos_embed_inference


def get_predictor(net,
                  brs_mode='NoBRS',
                  device='cpu',
                  prob_thresh=0.5,
                  with_flip=True,
                  zoom_in_params=dict(),
                  predictor_params=None,
                  brs_opt_func_params=None,
                  lbfgs_params=None):
    lbfgs_params_ = {
        'm': 20,
        'factr': 0,
        'pgtol': 1e-8,
        'maxfun': 20,
    }

    predictor_params_ = {
        'optimize_after_n_clicks': 1
    }

    if zoom_in_params is not None:
        zoom_in = ZoomIn(**zoom_in_params)
    else:
        zoom_in = None

    if lbfgs_params is not None:
        lbfgs_params_.update(lbfgs_params)
    lbfgs_params_['maxiter'] = 2 * lbfgs_params_['maxfun']

    if brs_opt_func_params is None:
        brs_opt_func_params = dict()

    # ------------------------------------------------------------------ #
    # SAM / SAM2 / SimpleClick dispatch — detected via model type tag    #
    # ------------------------------------------------------------------ #
    model_type = getattr(net, '_model_type', 'ritm')

    if model_type == 'sam':
        return SAMInteractivePredictor(net, device)

    if model_type == 'sam2':
        return SAM2InteractivePredictor(net, device)

    # SimpleClick and RITM share the same predictor pipeline, but SimpleClick's
    # ViT carries a fixed-size positional embedding, so every forward pass must
    # see the same input resolution. Upstream (scripts/evaluate_model.py) always
    # crops to a fixed target -- skip_clicks=-1 applies the zoom from the very
    # first click -- and interpolates pos_embed onto that grid. Without this the
    # backbone dies on a pos_embed/patch-count mismatch for any image whose size
    # is not exactly the checkpoint's img_size.
    if model_type == 'simpleclick':
        target_size = tuple(net.backbone.patch_embed.img_size)
        interpolate_pos_embed_inference(net.backbone, target_size, device)
        zoom_in = ZoomIn(target_size=target_size, skip_clicks=-1)

    if isinstance(net, (list, tuple)):
        assert brs_mode == 'NoBRS', "Multi-stage models support only NoBRS mode."

    if brs_mode == 'NoBRS':
        if predictor_params is not None:
            predictor_params_.update(predictor_params)
        predictor = BasePredictor(net, device, zoom_in=zoom_in, with_flip=with_flip, **predictor_params_)
    elif brs_mode.startswith('f-BRS'):
        predictor_params_.update({
            'net_clicks_limit': 8,
        })
        if predictor_params is not None:
            predictor_params_.update(predictor_params)

        insertion_mode = {
            'f-BRS-A': 'after_c4',
            'f-BRS-B': 'after_aspp',
            'f-BRS-C': 'after_deeplab'
        }[brs_mode]

        opt_functor = ScaleBiasOptimizer(prob_thresh=prob_thresh,
                                         with_flip=with_flip,
                                         optimizer_params=lbfgs_params_,
                                         **brs_opt_func_params)

        if isinstance(net, HRNetModel):
            FeaturePredictor = HRNetFeatureBRSPredictor
            insertion_mode = {'after_c4': 'A', 'after_aspp': 'A', 'after_deeplab': 'C'}[insertion_mode]
        else:
            FeaturePredictor = FeatureBRSPredictor

        predictor = FeaturePredictor(net, device,
                                     opt_functor=opt_functor,
                                     with_flip=with_flip,
                                     insertion_mode=insertion_mode,
                                     zoom_in=zoom_in,
                                     **predictor_params_)
    elif brs_mode == 'RGB-BRS' or brs_mode == 'DistMap-BRS':
        use_dmaps = brs_mode == 'DistMap-BRS'

        predictor_params_.update({
            'net_clicks_limit': 5,
        })
        if predictor_params is not None:
            predictor_params_.update(predictor_params)

        opt_functor = InputOptimizer(prob_thresh=prob_thresh,
                                     with_flip=with_flip,
                                     optimizer_params=lbfgs_params_,
                                     **brs_opt_func_params)

        predictor = InputBRSPredictor(net, device,
                                      optimize_target='dmaps' if use_dmaps else 'rgb',
                                      opt_functor=opt_functor,
                                      with_flip=with_flip,
                                      zoom_in=zoom_in,
                                      **predictor_params_)
    else:
        raise NotImplementedError

    return predictor
