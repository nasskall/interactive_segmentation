import numpy as np

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import torch
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


class SAM2InteractivePredictor:
    """
    Wraps Meta's SAM 2 to conform to the RITM predictor interface
    (set_input_image / get_prediction / get_states / set_states).
    """

    def __init__(self, model, device, **kwargs):
        if not SAM2_AVAILABLE:
            raise ImportError(
                "sam2 is not installed.\n"
                "Install it from: https://github.com/facebookresearch/sam2\n"
                "  git clone https://github.com/facebookresearch/sam2\n"
                "  cd sam2 && pip install -e ."
            )
        self.sam2_predictor = SAM2ImagePredictor(model)
        self.device = device
        self.original_image = None
        self.image_shape = None   # (H, W)
        self.prev_logit = None    # low-res mask for mask-conditioned prediction

    # ------------------------------------------------------------------
    # Interface required by InteractiveController
    # ------------------------------------------------------------------

    def set_input_image(self, image):
        """
        Parameters
        ----------
        image : np.ndarray  H x W x 3, dtype uint8, RGB order.
        """
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        self.original_image = image
        self.image_shape = image.shape[:2]
        with torch.inference_mode():
            self.sam2_predictor.set_image(image)
        self.prev_logit = None

    def get_prediction(self, clicker, prev_mask=None):
        """
        Returns
        -------
        np.ndarray  H x W float32 in [0, 1] — probability map.
        """
        clicks_list = clicker.get_clicks()
        if not clicks_list:
            return np.zeros(self.image_shape, dtype=np.float32)

        # RITM stores clicks as (row, col) = (y, x); SAM 2 expects (x, y).
        point_coords = []
        point_labels = []
        for click in clicks_list:
            y, x = click.coords
            point_coords.append([x, y])
            point_labels.append(1 if click.is_positive else 0)

        point_coords = np.array(point_coords, dtype=np.float32)
        point_labels = np.array(point_labels, dtype=np.int32)

        with torch.inference_mode():
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=self.prev_logit,
                multimask_output=False,
            )

        best_idx = int(np.argmax(scores))
        self.prev_logit = logits[best_idx : best_idx + 1]

        return masks[best_idx].astype(np.float32)

    def get_states(self):
        return {
            'prev_logit': self.prev_logit.copy() if self.prev_logit is not None else None
        }

    def set_states(self, states):
        self.prev_logit = states['prev_logit']