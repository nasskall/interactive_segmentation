import numpy as np

try:
    from segment_anything import SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class SAMInteractivePredictor:
    """
    Wraps Meta's Segment Anything Model (SAM) to conform to the RITM
    predictor interface (set_input_image / get_prediction / get_states /
    set_states).
    """

    def __init__(self, model, device, **kwargs):
        if not SAM_AVAILABLE:
            raise ImportError(
                "segment_anything is not installed.\n"
                "Install it with:  pip install segment-anything"
            )
        self.sam_predictor = SamPredictor(model)
        self.device = device
        self.original_image = None
        self.image_shape = None   # (H, W)
        self.prev_logit = None    # (1, 256, 256) low-res mask for mask-conditioned prediction

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
        self.sam_predictor.set_image(image)
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

        # RITM stores clicks as (row, col) = (y, x); SAM expects (x, y).
        point_coords = []
        point_labels = []
        for click in clicks_list:
            y, x = click.coords
            point_coords.append([x, y])
            point_labels.append(1 if click.is_positive else 0)

        point_coords = np.array(point_coords, dtype=np.float32)
        point_labels = np.array(point_labels, dtype=np.int32)

        masks, scores, logits = self.sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=self.prev_logit,
            multimask_output=False,
        )

        best_idx = int(np.argmax(scores))
        # Save low-res logit for next call (mask conditioning).
        self.prev_logit = logits[best_idx : best_idx + 1]   # (1, 256, 256)

        # Return binary mask as float32; threshold 0.5 in controller matches.
        return masks[best_idx].astype(np.float32)

    def get_states(self):
        return {
            'prev_logit': self.prev_logit.copy() if self.prev_logit is not None else None
        }

    def set_states(self, states):
        self.prev_logit = states['prev_logit']