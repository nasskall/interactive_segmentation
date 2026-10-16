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
        self.box = None           # [x0, y0, x1, y1] box prompt, if any
        # soft=True returns sigmoid probabilities instead of a 0/1 mask (the
        # ensemble needs them to fuse models); last_score is SAM's own IoU
        # estimate for the returned mask.
        self.soft = False
        self.last_score = None

    # ------------------------------------------------------------------
    # Interface required by InteractiveController
    # ------------------------------------------------------------------

    def set_input_image(self, image):
        """
        Parameters
        ----------
        image : np.ndarray  H x W x 3, dtype uint8, RGB order.
        """
        self.source_image = image   # as given, to recognise the same image later
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        self.original_image = image
        self.image_shape = image.shape[:2]
        self.sam_predictor.set_image(image)
        self.prev_logit = None
        self.box = None

    def reset_prompts(self):
        """Forget clicks, box and mask prompt; keep the image embedding."""
        self.prev_logit = None
        self.box = None

    def set_box(self, box):
        """Box prompt [x0, y0, x1, y1] used with the clicks from now on."""
        self.box = [float(v) for v in box]

    def get_prediction(self, clicker, prev_mask=None):
        """
        Returns
        -------
        np.ndarray  H x W float32 in [0, 1] — probability map.
        """
        clicks_list = clicker.get_clicks()
        if not clicks_list and self.box is None:
            return np.zeros(self.image_shape, dtype=np.float32)

        # RITM stores clicks as (row, col) = (y, x); SAM expects (x, y).
        point_coords = []
        point_labels = []
        for click in clicks_list:
            y, x = click.coords
            point_coords.append([x, y])
            point_labels.append(1 if click.is_positive else 0)

        has_points = bool(point_coords)
        point_coords = np.array(point_coords, dtype=np.float32) if has_points else None
        point_labels = np.array(point_labels, dtype=np.int32) if has_points else None
        box = np.array(self.box, dtype=np.float32) if self.box is not None else None

        # A single click is ambiguous (part vs. whole object), so ask for
        # three candidates and keep the best-scoring one; with more clicks,
        # or a box, the prompt is specific enough for a single mask. This
        # follows the segment-anything usage guidance and gives far cleaner
        # first masks.
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=self.prev_logit,
            multimask_output=len(clicks_list) == 1 and box is None,
            return_logits=self.soft,
        )

        best_idx = int(np.argmax(scores))
        # Save low-res logit for next call (mask conditioning).
        self.prev_logit = logits[best_idx : best_idx + 1]   # (1, 256, 256)
        self.last_score = float(scores[best_idx])

        if self.soft:
            return (1.0 / (1.0 + np.exp(-np.clip(masks[best_idx], -50, 50)))).astype(np.float32)
        # Return binary mask as float32; threshold 0.5 in controller matches.
        return masks[best_idx].astype(np.float32)

    def get_states(self):
        return {
            'prev_logit': self.prev_logit.copy() if self.prev_logit is not None else None,
            'box': self.box,
        }

    def set_states(self, states):
        self.prev_logit = states['prev_logit']
        self.box = states.get('box')