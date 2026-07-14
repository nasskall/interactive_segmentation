import os
import time
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from tkinter.ttk import Frame

import cv2
import numpy as np
from PIL import Image, ImageTk

from interactive_demo.canvas import CanvasImage
from interactive_demo.controller import InteractiveController
from interactive_demo.training_panel import FineTuneDialog
from interactive_demo.wrappers import (
    FocusLabelFrame, FocusButton, FocusCheckButton,
    BoundedNumericalEntry, FocusHorizontalScale,
)
from isegm.data.custom_dataset import CustomDataset
from isegm.inference import utils

# ── colour palette ──────────────────────────────────────────────────────────
_BLUE      = '#0078d4'
_BLUE_DARK = '#005a9e'
_BG        = '#f5f5f5'
_PANEL_BG  = '#ffffff'
_GREEN     = '#107c10'
_ORANGE    = '#ca5010'
_BTN_BG    = '#e8e8e8'
_SIDEBAR_W = 310


class InteractiveDemoApp(ttk.Frame):
    """Main application window."""

    def __init__(self, master, limit_longest_size, device, model):
        super().__init__(master)
        self.master = master
        self.device = device
        self.model = model
        self.limit_longest_size = limit_longest_size
        self._loaded_model_name: str = 'pre-loaded'
        self._click_count: int = 0
        self.original_size = None
        self.filename = ''
        self.image_on_canvas = None

        master.title('Interactive Segmentation')
        master.configure(bg=_BG)

        # Size / position — full screen minus taskbar, centred
        master.withdraw()
        master.update_idletasks()
        sw = int(master.winfo_screenwidth()  * 0.98)
        sh = int(master.winfo_screenheight() * 0.92)
        master.geometry(f'{sw}x{sh}+0+0')
        master.deiconify()

        master.protocol('WM_DELETE_WINDOW', self._on_close)

        self.pack(fill='both', expand=True)

        self.brs_modes   = ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']
        self.model_types = [
            'RITM', 'SimpleClick',
            'SAM - ViT-B', 'SAM - ViT-L', 'SAM - ViT-H',
            'SAM2 - Tiny', 'SAM2 - Small', 'SAM2 - Base+', 'SAM2 - Large',
        ]
        self._sam_types = {t for t in self.model_types if t.startswith('SAM')}

        self.controller = InteractiveController(
            model, device,
            predictor_params={'brs_mode': 'NoBRS'},
            update_image_callback=self._update_image,
        )

        self._init_state()
        self._build_layout()
        self._bind_keys()
        self._init_traces()
        self._change_brs_mode()
        self._refresh_status()

    # ──────────────────────────────────────────────────────────────────────
    # State
    # ──────────────────────────────────────────────────────────────────────

    def _init_state(self):
        lim = self.limit_longest_size
        self.state = {
            'zoomin_params': {
                'use_zoom_in':    tk.BooleanVar(value=True),
                'fixed_crop':     tk.BooleanVar(value=True),
                'skip_clicks':    tk.IntVar(value=-1),
                'target_size':    tk.IntVar(value=min(400, lim)),
                'expansion_ratio':tk.DoubleVar(value=1.4),
            },
            'predictor_params': {
                'net_clicks_limit': tk.IntVar(value=8),
            },
            'brs_mode':      tk.StringVar(value='NoBRS'),
            'prob_thresh':   tk.DoubleVar(value=0.5),
            'lbfgs_max_iters': tk.IntVar(value=20),
            'alpha_blend':   tk.DoubleVar(value=0.5),
            'click_radius':  tk.IntVar(value=3),
            'model_type':    tk.StringVar(value='RITM'),
        }

    def _init_traces(self):
        t = self.state
        t['zoomin_params']['skip_clicks'].trace('w', self._reset_predictor)
        t['zoomin_params']['target_size'].trace('w', self._reset_predictor)
        t['zoomin_params']['expansion_ratio'].trace('w', self._reset_predictor)
        t['predictor_params']['net_clicks_limit'].trace('w', self._change_brs_mode)
        t['lbfgs_max_iters'].trace('w', self._change_brs_mode)
        t['model_type'].trace('w', self._on_model_type_changed)

    def _bind_keys(self):
        self.master.bind('<space>', lambda e: self._finish_object_callback())
        self.master.bind('a',      lambda e: self.controller.partially_finish_object())

    # ──────────────────────────────────────────────────────────────────────
    # Layout construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_layout(self):
        """Build the complete window layout."""
        self._build_menubar()
        self._build_statusbar()        # packed BOTTOM before content
        self._build_content()          # canvas + sidebar

    # ── title bar ─────────────────────────────────────────────────────────
    # ── menu bar ──────────────────────────────────────────────────────────
    def _build_menubar(self):
        self.menubar = tk.Frame(self, bg=_BTN_BG, bd=0, relief='flat')
        self.menubar.pack(fill='x', side='top')

        # Separator line under menu
        tk.Frame(self, bg='#cccccc', height=1).pack(fill='x', side='top')

        def btn(text, cmd, fg='black', state=tk.NORMAL):
            b = tk.Button(
                self.menubar, text=text, command=cmd,
                bg=_BTN_BG, fg=fg, activebackground='#d0d0d0',
                relief='flat', padx=12, pady=6,
                font=('Helvetica', 9), cursor='hand2',
                state=state,
            )
            b.pack(side='left')
            b.bind('<Enter>', lambda e: b.config(bg='#d6d6d6') if str(b.cget('state')) != 'disabled' else None)
            b.bind('<Leave>', lambda e: b.config(bg=_BTN_BG))
            return b

        def sep():
            tk.Frame(self.menubar, bg='#cccccc', width=1).pack(side='left', fill='y', pady=4, padx=2)

        self.load_image_btn = btn('Load Image',         self._load_image_callback)
        self.load_model_btn = btn('Load Model',         self._load_model_callback)
        sep()
        self.save_mask_btn  = btn('Save Mask',          self._save_mask_callback,        state=tk.DISABLED)
        sep()
        self.save_for_training_btn = btn(
            'Save for Training', self._save_for_training_callback,
            fg=_BLUE_DARK, state=tk.DISABLED,
        )
        self.fine_tune_btn  = btn('Fine-tune Model',    self._open_fine_tune_dialog, fg=_BLUE_DARK)
        sep()
        self.about_btn      = btn('About',              self._about_callback)

    # ── status bar ────────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self, bg=_BLUE, height=26)
        bar.pack(fill='x', side='bottom')
        bar.pack_propagate(False)

        lbl_kw = dict(bg=_BLUE, fg='white', font=('Helvetica', 8))

        self._status_model  = tk.Label(bar, text='', **lbl_kw)
        self._status_image  = tk.Label(bar, text='', **lbl_kw)
        self._status_clicks = tk.Label(bar, text='', **lbl_kw)
        self._status_device = tk.Label(bar, text=f'Device: {self.device}', **lbl_kw)
        self._status_pairs  = tk.Label(bar, text='', **lbl_kw)

        self._status_device.pack(side='right', padx=12)
        tk.Frame(bar, bg='#5aa0d4', width=1).pack(side='right', fill='y', pady=3)
        self._status_pairs.pack(side='right', padx=12)
        tk.Frame(bar, bg='#5aa0d4', width=1).pack(side='right', fill='y', pady=3)
        self._status_clicks.pack(side='right', padx=12)
        self._status_model.pack(side='left',  padx=12)
        tk.Frame(bar, bg='#5aa0d4', width=1).pack(side='left', fill='y', pady=3)
        self._status_image.pack(side='left',  padx=12)

    # ── main content area ─────────────────────────────────────────────────
    def _build_content(self):
        content = tk.Frame(self, bg=_BG)
        content.pack(fill='both', expand=True)

        # Canvas area — fills all space not taken by the sidebar
        self._build_canvas(content)

        # Sidebar — fixed width, right side
        self._build_sidebar(content)

    def _build_canvas(self, parent):
        self.canvas_frame = FocusLabelFrame(parent, text='Image')
        self.canvas_frame.pack(side='left', fill='both', expand=True, padx=(8, 4), pady=8)
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self.canvas_frame,
            highlightthickness=0,
            cursor='hand1',
            bg='#2b2b2b',
        )
        self.canvas.grid(row=0, column=0, sticky='nsew')

        # Welcome overlay (replaced once an image loads)
        self._canvas_placeholder = self.canvas.create_text(
            300, 250,
            text='Load an image to begin\n\nFile → Load Image',
            font=('Helvetica', 16), fill='#888888',
            justify='center',
        )

    def _build_sidebar(self, parent):
        # Outer container — fixed width, scrollable inside
        outer = tk.Frame(parent, bg=_BG, width=_SIDEBAR_W)
        outer.pack(side='right', fill='y', padx=(4, 8), pady=8)
        outer.pack_propagate(False)

        # Scrollable canvas + scrollbar
        scroll_canvas = tk.Canvas(outer, bg=_BG, highlightthickness=0, width=_SIDEBAR_W - 4)
        vsb = ttk.Scrollbar(outer, orient='vertical', command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side='right', fill='y')
        scroll_canvas.pack(side='left', fill='both', expand=True)

        inner = tk.Frame(scroll_canvas, bg=_BG)
        win_id = scroll_canvas.create_window((0, 0), window=inner, anchor='nw')

        def _on_frame_resize(event):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox('all'))
            scroll_canvas.itemconfig(win_id, width=scroll_canvas.winfo_width())

        inner.bind('<Configure>', _on_frame_resize)
        scroll_canvas.bind('<Configure>', lambda e: scroll_canvas.itemconfig(win_id, width=e.width))
        scroll_canvas.bind('<MouseWheel>', lambda e: scroll_canvas.yview_scroll(-1*(e.delta//120), 'units'))

        master = inner   # shorthand for all sections below

        # ── 1. Keyboard shortcuts ─────────────────────────────────────────
        shortcuts = FocusLabelFrame(master, text='Keyboard Shortcuts')
        shortcuts.pack(fill='x', padx=6, pady=(6, 4))
        shortcuts_text = (
            'Left click    →  Positive click\n'
            'Right click   →  Negative click\n'
            'Scroll wheel  →  Zoom in / out\n'
            'Right drag    →  Pan image\n'
            'Space         →  Finish object\n'
            'A             →  Partial finish'
        )
        tk.Label(
            shortcuts, text=shortcuts_text,
            font=('Courier', 8), justify='left', bg=_PANEL_BG,
            relief='flat',
        ).pack(fill='x')

        # ── 2. Segmentation controls ──────────────────────────────────────
        self.clicks_options_frame = FocusLabelFrame(master, text='Segmentation')
        self.clicks_options_frame.pack(fill='x', padx=6, pady=4)

        btn_frame = tk.Frame(self.clicks_options_frame, bg=_PANEL_BG)
        btn_frame.pack(fill='x')

        def seg_btn(text, cmd, col):
            b = FocusButton(
                btn_frame, text=text, command=cmd,
                bg=_BTN_BG, fg='black', width=9, height=2,
                state=tk.DISABLED, relief='groove',
            )
            b.grid(row=0, column=col, padx=3, pady=2, sticky='ew')
            btn_frame.columnconfigure(col, weight=1)
            return b

        self.finish_object_button = seg_btn('Finish\n(Space)', self._finish_object_callback, 0)
        self.undo_click_button    = seg_btn('Undo\nClick',      self.controller.undo_click,    1)
        self.reset_clicks_button  = seg_btn('Reset\nAll',       self._reset_last_object,        2)

        # Auto-segment button — no clicks needed; uses model_type-specific
        # heuristic (auto-mask-generator for SAM/SAM2, center click otherwise).
        self.auto_segment_button = FocusButton(
            self.clicks_options_frame, text='Auto-segment',
            command=self._auto_segment_callback,
            bg=_BTN_BG, fg=_BLUE_DARK, height=1,
            state=tk.DISABLED, relief='groove',
        )
        self.auto_segment_button.pack(fill='x', pady=(4, 0), padx=3)

        # Click counter label
        self._click_lbl = tk.Label(
            self.clicks_options_frame, text='No clicks yet',
            font=('Helvetica', 8), fg='#666666',
        )
        self._click_lbl.pack(pady=(4, 0))

        # ── 3. Model ──────────────────────────────────────────────────────
        self.model_type_frame = FocusLabelFrame(master, text='Model')
        self.model_type_frame.pack(fill='x', padx=6, pady=4)

        tk.Label(self.model_type_frame, text='Architecture:',
                 font=('Helvetica', 8)).pack(anchor='w')
        self.model_type_combobox = ttk.Combobox(
            self.model_type_frame,
            textvariable=self.state['model_type'],
            values=self.model_types,
            state='readonly', width=24,
        )
        self.model_type_combobox.pack(fill='x', pady=(2, 6))
        self.model_type_combobox.bind('<<ComboboxSelected>>', lambda e: self._on_model_type_changed())

        ttk.Button(
            self.model_type_frame, text='Load Model…', command=self._load_model_callback,
        ).pack(fill='x')

        self._model_lbl = tk.Label(
            self.model_type_frame, text='No model loaded',
            font=('Helvetica', 8), fg='#888888', wraplength=_SIDEBAR_W - 30,
        )
        self._model_lbl.pack(pady=(4, 0))

        # ── 4. ZoomIn ─────────────────────────────────────────────────────
        self.zoomin_options_frame = FocusLabelFrame(master, text='ZoomIn')
        self.zoomin_options_frame.pack(fill='x', padx=6, pady=4)

        zi = self.zoomin_options_frame
        FocusCheckButton(zi, text='Use ZoomIn',  command=self._reset_predictor,
                         variable=self.state['zoomin_params']['use_zoom_in']
                         ).grid(row=0, column=0, padx=6, sticky='w')
        FocusCheckButton(zi, text='Fixed crop',  command=self._reset_predictor,
                         variable=self.state['zoomin_params']['fixed_crop']
                         ).grid(row=1, column=0, padx=6, sticky='w')
        tk.Label(zi, text='Skip clicks').grid(row=0, column=1, sticky='e', pady=1)
        tk.Label(zi, text='Target size').grid(row=1, column=1, sticky='e', pady=1)
        tk.Label(zi, text='Expand ratio').grid(row=2, column=1, sticky='e', pady=1)
        BoundedNumericalEntry(zi, variable=self.state['zoomin_params']['skip_clicks'],
                              min_value=-1, max_value=None, vartype=int,
                              name='zoom_in_skip_clicks'
                              ).grid(row=0, column=2, padx=6, pady=1, sticky='w')
        BoundedNumericalEntry(zi, variable=self.state['zoomin_params']['target_size'],
                              min_value=100, max_value=self.limit_longest_size, vartype=int,
                              name='zoom_in_target_size'
                              ).grid(row=1, column=2, padx=6, pady=1, sticky='w')
        BoundedNumericalEntry(zi, variable=self.state['zoomin_params']['expansion_ratio'],
                              min_value=1.0, max_value=2.0, vartype=float,
                              name='zoom_in_expansion_ratio'
                              ).grid(row=2, column=2, padx=6, pady=1, sticky='w')
        zi.columnconfigure((0, 1, 2), weight=1)

        # ── 5. BRS ────────────────────────────────────────────────────────
        self.brs_options_frame = FocusLabelFrame(master, text='BRS')
        self.brs_options_frame.pack(fill='x', padx=6, pady=4)

        brs = self.brs_options_frame
        self.brs_mode_combobox = ttk.Combobox(
            brs, textvariable=self.state['brs_mode'],
            values=self.brs_modes, state='readonly',
        )
        self.brs_mode_combobox.grid(row=0, rowspan=2, column=0, padx=6, pady=4, sticky='ew')
        self.brs_mode_combobox.bind('<<ComboboxSelected>>', lambda e: self._change_brs_mode())

        self.net_clicks_label = tk.Label(brs, text='Net clicks')
        self.net_clicks_label.grid(row=0, column=1, sticky='e', pady=2)
        self.net_clicks_entry = BoundedNumericalEntry(
            brs, variable=self.state['predictor_params']['net_clicks_limit'],
            min_value=0, max_value=None, vartype=int, allow_inf=True,
            name='net_clicks_limit',
        )
        self.net_clicks_entry.grid(row=0, column=2, padx=6, pady=2, sticky='w')

        self.lbfgs_iters_label = tk.Label(brs, text='L-BFGS iters')
        self.lbfgs_iters_label.grid(row=1, column=1, sticky='e', pady=2)
        self.lbfgs_iters_entry = BoundedNumericalEntry(
            brs, variable=self.state['lbfgs_max_iters'],
            min_value=1, max_value=1000, vartype=int,
            name='lbfgs_max_iters',
        )
        self.lbfgs_iters_entry.grid(row=1, column=2, padx=6, pady=2, sticky='w')
        brs.columnconfigure((0, 1), weight=1)

        # ── 6. Visualisation ──────────────────────────────────────────────
        vis_frame = FocusLabelFrame(master, text='Visualisation')
        vis_frame.pack(fill='x', padx=6, pady=4)

        tk.Label(vis_frame, text='Mask opacity').pack(anchor='w')
        FocusHorizontalScale(
            vis_frame, from_=0.0, to=1.0,
            command=self._update_blend_alpha,
            variable=self.state['alpha_blend'],
        ).pack(fill='x', padx=4)

        tk.Label(vis_frame, text='Probability threshold').pack(anchor='w', pady=(6, 0))
        FocusHorizontalScale(
            vis_frame, from_=0.0, to=1.0,
            command=self._update_prob_thresh,
            variable=self.state['prob_thresh'],
        ).pack(fill='x', padx=4)

        tk.Label(vis_frame, text='Click dot radius').pack(anchor='w', pady=(6, 0))
        FocusHorizontalScale(
            vis_frame, from_=0, to=7, resolution=1,
            command=self._update_click_radius,
            variable=self.state['click_radius'],
        ).pack(fill='x', padx=4)

        # ── 7. Training ───────────────────────────────────────────────────
        train_frame = FocusLabelFrame(master, text='Training Data')
        train_frame.pack(fill='x', padx=6, pady=(4, 8))

        self._training_count_lbl = tk.Label(
            train_frame, text='', font=('Helvetica', 8), fg='#555555',
        )
        self._training_count_lbl.pack(anchor='w', pady=(0, 6))
        self._refresh_training_count()

        ttk.Button(
            train_frame, text='Save for Training',
            command=self._save_for_training_callback,
        ).pack(fill='x', pady=2)
        self._sidebar_save_btn = train_frame.winfo_children()[-1]   # keep ref

        ttk.Button(
            train_frame, text='Fine-tune Model…',
            command=self._open_fine_tune_dialog,
        ).pack(fill='x', pady=2)

        tk.Label(
            train_frame,
            text='After fine-tuning, use Load Model to switch.',
            font=('Helvetica', 7), fg='#888888', wraplength=_SIDEBAR_W - 30,
            justify='left',
        ).pack(anchor='w', pady=(4, 0))

        # ── 8. Domain Adaptation (LoRA) ───────────────────────────────────
        self._build_adaptation_panel(master)

    # ──────────────────────────────────────────────────────────────────────
    # Status helpers
    # ──────────────────────────────────────────────────────────────────────

    def _live_click_count(self) -> int:
        """Clicks currently on the canvas, straight from the clicker."""
        try:
            return len(self.controller.clicker.clicks_list)
        except AttributeError:
            return 0

    def _refresh_status(self):
        self._status_model.config(text=f'Model: {self._loaded_model_name}')
        img_text = f'Image: {self.filename}' if self.filename else 'No image loaded'
        self._status_image.config(text=img_text)
        n = self._live_click_count()
        self._status_clicks.config(text=f'{n} click{"s" if n != 1 else ""}')
        self._refresh_training_count()

    def _refresh_training_count(self):
        try:
            ds = CustomDataset('training_data', augment=False)
            n = len(ds)
        except Exception:
            n = 0
        self._status_pairs.config(text=f'Training pairs: {n}')
        if hasattr(self, '_training_count_lbl'):
            self._training_count_lbl.config(
                text=f'{n} pair{"s" if n != 1 else ""} in training_data/'
            )

    # ──────────────────────────────────────────────────────────────────────
    # Menu callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _load_image_callback(self):
        self.menubar.focus_set()
        if not self._check_entry(self):
            return

        path = filedialog.askopenfilename(
            parent=self.master,
            filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.tiff'), ('All files', '*.*')],
            title='Choose an image',
        )
        if not path:
            return

        self.save_mask_btn.configure(state=tk.DISABLED)
        self.save_for_training_btn.configure(state=tk.DISABLED)

        self.canvas.delete('all')
        self._canvas_placeholder = self.canvas.create_text(
            self.canvas.winfo_width() // 2 or 300,
            self.canvas.winfo_height() // 2 or 250,
            text='Loading…', font=('Helvetica', 18), fill='#888888',
        )
        self.canvas.update()

        original = cv2.imread(path)
        if original is None:
            messagebox.showerror('Error', f'Cannot read image:\n{path}', parent=self.master)
            return
        self.original_size = original.shape[:2]

        image = self._load_and_resize_image(path)

        # Resize canvas to 40% of window width
        w = int(self.master.winfo_width() * 0.62)
        h = int(self.master.winfo_height() * 0.80)
        self.canvas.config(width=w, height=h, scrollregion=(0, 0, w, h))
        self.canvas.delete('all')
        self.image_on_canvas = None

        self.controller.set_image(image)
        self._click_count = 0

        self.filename = os.path.splitext(os.path.basename(path))[0]
        self._loaded_model_name = self._loaded_model_name   # unchanged
        self._refresh_status()

    def _load_and_resize_image(self, filename, max_size=(2048, 2048)):
        image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        ratio = w / h
        if w > h:
            nw = min(max_size[0], w)
            nh = int(nw / ratio)
        else:
            nh = min(max_size[1], h)
            nw = int(nh * ratio)
        return cv2.resize(image, (nw, nh))

    def _finish_object_callback(self):
        if not self.controller.is_incomplete_mask:
            return
        self.controller.finish_object()
        if not self.controller.is_incomplete_mask:
            self.save_mask_btn.configure(state=tk.NORMAL)
            self.save_for_training_btn.configure(state=tk.NORMAL)
        if hasattr(self, '_buffer_lbl'):
            self._refresh_adapter_status()

    def _auto_segment_callback(self):
        self.menubar.focus_set()
        if self.controller.image is None:
            return
        try:
            result = self.controller.auto_segment(mode='auto')
        except Exception as exc:
            messagebox.showerror('Auto-segment failed', str(exc),
                                 parent=self.master)
            return
        if not result['ok']:
            messagebox.showinfo('Auto-segment',
                                f"No mask produced: {result.get('reason')}",
                                parent=self.master)
            return
        # Auto-segment may inject a center click — sync the visible counter.
        self._click_count = len(self.controller.clicker.clicks_list)
        self._refresh_status()
        self.save_mask_btn.configure(state=tk.NORMAL)
        self.save_for_training_btn.configure(state=tk.NORMAL)

    def _save_mask_callback(self):
        self.menubar.focus_set()
        mask = self.controller.result_mask
        if mask is None:
            messagebox.showwarning('Warning', 'No mask to save.', parent=self.master)
            return
        if not self._check_entry(self):
            return

        resized = cv2.resize(
            mask, (self.original_size[1], self.original_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        savename = filedialog.asksaveasfilename(
            parent=self.master,
            initialfile=f'{self.filename}_mask.png',
            filetypes=[('PNG image', '*.png'), ('BMP image', '*.bmp'), ('All files', '*.*')],
            title='Save mask as…',
        )
        if not savename:
            return

        out = resized.astype(np.uint8)
        if out.max() > 0 and out.max() < 256:
            out = (out * (255 // out.max())).astype(np.uint8)
        cv2.imwrite(savename, out)
        self._status_image.config(text=f'Saved → {os.path.basename(savename)}')

    def _load_model_callback(self):
        self.menubar.focus_set()
        model_type = self.state['model_type'].get()
        path = filedialog.askopenfilename(
            parent=self.master,
            filetypes=[('PyTorch model files', '*.pth *.pt'), ('All files', '*.*')],
            title=f'Choose a {model_type} model file',
        )
        if not path:
            return
        try:
            model = utils.load_model_by_type(path, model_type, self.device)
            self.controller.set_net(model)
            self._loaded_model_name = os.path.basename(path)
            self._model_lbl.config(text=self._loaded_model_name, fg=_GREEN)
            self._refresh_status()
            if hasattr(self, '_adapter_status_lbl'):
                self._refresh_adapter_status()
            messagebox.showinfo('Model Loaded',
                                f'{model_type} loaded successfully.\n{path}',
                                parent=self.master)
        except Exception as exc:
            messagebox.showerror('Error', f'Failed to load model:\n{exc}', parent=self.master)

    # ──────────────────────────────────────────────────────────────────────
    # Training callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _save_for_training_callback(self):
        self.menubar.focus_set()
        image = getattr(self.controller, 'image', None)
        mask  = self.controller.result_mask
        if image is None or mask is None:
            messagebox.showwarning('Warning', 'No image or mask available.', parent=self.master)
            return

        resized_mask  = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        binary_mask   = (resized_mask > 0).astype(np.uint8)
        stem          = f'sample_{int(time.time())}'

        try:
            CustomDataset.add_pair('training_data', image, binary_mask, stem)
            self._refresh_training_count()
            messagebox.showinfo('Saved',
                                f'Pair saved to training_data/\n(stem: {stem})',
                                parent=self.master)
        except Exception as exc:
            messagebox.showerror('Error', f'Failed to save:\n{exc}', parent=self.master)

    def _open_fine_tune_dialog(self):
        self.menubar.focus_set()
        FineTuneDialog(self.master, self.device, on_model_saved=self._on_fine_tune_model_saved)

    def _on_fine_tune_model_saved(self, path: str):
        self._refresh_training_count()
        messagebox.showinfo(
            'Model Saved',
            f'Fine-tuned model saved to:\n{path}\n\nUse "Load Model" to switch to it.',
            parent=self.master,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Domain Adaptation panel (Mode A: dialog, Mode B: sidebar controls)
    # ──────────────────────────────────────────────────────────────────────

    def _build_adaptation_panel(self, parent):
        from interactive_demo.wrappers import FocusLabelFrame, FocusCheckButton
        adapt_frame = FocusLabelFrame(parent, text='Domain Adaptation')
        adapt_frame.pack(fill='x', padx=6, pady=(4, 8))

        # Mode A — offline few-shot
        ttk.Button(
            adapt_frame, text='Few-Shot Adapt…',
            command=self._open_few_shot_dialog,
        ).pack(fill='x', pady=2)

        # Mode B — online adaptation toggle + actions
        self._online_var = tk.BooleanVar(value=False)
        FocusCheckButton(
            adapt_frame, text='Online adapt from clicks',
            variable=self._online_var,
            command=self._toggle_online_recording,
        ).pack(anchor='w', pady=(4, 2))

        self._buffer_lbl = tk.Label(
            adapt_frame, text='Buffer: 0',
            font=('Helvetica', 8), fg='#666',
        )
        self._buffer_lbl.pack(anchor='w', padx=4)

        row = tk.Frame(adapt_frame, bg=_PANEL_BG)
        row.pack(fill='x', pady=2)
        ttk.Button(row, text='Adapt now',
                   command=self._adapt_now_callback).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(row, text='Rollback',
                   command=self._rollback_adapter_callback).pack(side='left', expand=True, fill='x', padx=2)

        row2 = tk.Frame(adapt_frame, bg=_PANEL_BG)
        row2.pack(fill='x', pady=2)
        ttk.Button(row2, text='Save…',
                   command=self._save_adapter_callback).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(row2, text='Load…',
                   command=self._load_adapter_callback).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(row2, text='Reset',
                   command=self._reset_adapter_callback).pack(side='left', expand=True, fill='x', padx=2)

        self._adapter_status_lbl = tk.Label(
            adapt_frame, text='no adapter',
            font=('Helvetica', 8), fg='#888', wraplength=_SIDEBAR_W - 30,
        )
        self._adapter_status_lbl.pack(anchor='w', padx=4, pady=(4, 0))

        # Wire controller log to the status bar tail (also visible via prints).
        self.controller.set_adapter_log_cb(self._adapter_log)

    def _adapter_log(self, msg: str):
        # Keep adapter messages out of the modal flow — show in the status bar.
        try:
            self._status_image.config(text=msg[:80])
        except Exception:
            pass
        print(f'[adapter] {msg}')

    def _refresh_adapter_status(self):
        if hasattr(self, '_adapter_status_lbl'):
            self._adapter_status_lbl.config(text=self.controller.adapter_status())
        if hasattr(self, '_buffer_lbl'):
            self._buffer_lbl.config(text=f'Buffer: {len(self.controller.replay_buffer)}')

    def _open_few_shot_dialog(self):
        from interactive_demo.few_shot_panel import FewShotAdaptDialog
        self.menubar.focus_set()
        FewShotAdaptDialog(
            self.master, self.controller,
            on_adapter_applied=self._refresh_adapter_status,
        )

    def _toggle_online_recording(self):
        self.controller.online_recording = bool(self._online_var.get())
        msg = ('Online recording ON — finished objects will be saved to buffer.'
               if self.controller.online_recording
               else 'Online recording OFF.')
        self._adapter_log(msg)

    def _adapt_now_callback(self):
        if len(self.controller.replay_buffer) == 0:
            messagebox.showinfo('Empty buffer',
                                'Enable "Online adapt from clicks" and finish at '
                                'least one object before adapting.',
                                parent=self.master)
            return
        try:
            report = self.controller.adapt_now()
        except Exception as exc:
            messagebox.showerror('Adapt failed', str(exc), parent=self.master)
            return
        self._refresh_adapter_status()
        messagebox.showinfo('Adapted',
                            f"{report['steps']} steps, "
                            f"mean loss = {report['mean_loss']:.4f}",
                            parent=self.master)

    def _rollback_adapter_callback(self):
        if self.controller.rollback_adapter():
            self._refresh_adapter_status()
        else:
            messagebox.showinfo('Nothing to roll back',
                                'No adaptation snapshots are stacked.',
                                parent=self.master)

    def _reset_adapter_callback(self):
        if not messagebox.askyesno(
            'Reset adapter',
            'This re-initialises the LoRA adapter to identity. Continue?',
            parent=self.master,
        ):
            return
        self.controller.reset_adapter()
        self._refresh_adapter_status()

    def _save_adapter_callback(self):
        model_type = self.controller.model_type
        from pathlib import Path
        out_dir = Path('adapters') / model_type
        out_dir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self.master,
            initialdir=str(out_dir),
            initialfile=f'{model_type}_adapter.pt',
            filetypes=[('Adapter', '*.pt'), ('All files', '*.*')],
            title='Save adapter as…',
        )
        if not path:
            return
        try:
            self.controller.save_adapter(path)
        except Exception as exc:
            messagebox.showerror('Save failed', str(exc), parent=self.master)
            return
        self._refresh_adapter_status()
        messagebox.showinfo('Saved', f'Adapter saved to:\n{path}',
                            parent=self.master)

    def _load_adapter_callback(self):
        path = filedialog.askopenfilename(
            parent=self.master,
            filetypes=[('Adapter', '*.pt'), ('All files', '*.*')],
            title='Load adapter…',
        )
        if not path:
            return
        try:
            missing, unexpected = self.controller.load_adapter(path)
        except Exception as exc:
            messagebox.showerror('Load failed', str(exc), parent=self.master)
            return
        self._refresh_adapter_status()
        info = f'Loaded adapter:\n{path}'
        if missing or unexpected:
            info += (f'\n\n{len(missing)} missing, {len(unexpected)} unexpected '
                     f'tensors (model topology differs).')
        messagebox.showinfo('Loaded', info, parent=self.master)

    # ──────────────────────────────────────────────────────────────────────
    # About
    # ──────────────────────────────────────────────────────────────────────

    def _about_callback(self):
        messagebox.showinfo('About',
            'Skin Lesion Interactive Segmentation\n\n'
            'Based on RITM Interactive Segmentation\n'
            'https://github.com/SamsungLabs/ritm_interactive_segmentation\n\n'
            'Licensed under the MIT License.',
            parent=self.master,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Predictor / BRS helpers
    # ──────────────────────────────────────────────────────────────────────

    def _reset_last_object(self):
        self.state['alpha_blend'].set(0.5)
        self.state['prob_thresh'].set(0.5)
        self._click_count = 0
        self.controller.reset_last_object()
        self._refresh_status()

    def _update_prob_thresh(self, value):
        if self.controller.is_incomplete_mask:
            self.controller.prob_thresh = self.state['prob_thresh'].get()
            self._update_image()

    def _update_blend_alpha(self, value):
        self._update_image()

    def _update_click_radius(self, *args):
        if self.image_on_canvas is not None:
            self._update_image()

    def _change_brs_mode(self, *args):
        if self.state['brs_mode'].get() == 'NoBRS':
            self.net_clicks_entry.set('INF')
            self.net_clicks_entry.configure(state=tk.DISABLED)
            self.net_clicks_label.configure(state=tk.DISABLED)
            self.lbfgs_iters_entry.configure(state=tk.DISABLED)
            self.lbfgs_iters_label.configure(state=tk.DISABLED)
        else:
            if self.net_clicks_entry.get() == 'INF':
                self.net_clicks_entry.set(8)
            self.net_clicks_entry.configure(state=tk.NORMAL)
            self.net_clicks_label.configure(state=tk.NORMAL)
            self.lbfgs_iters_entry.configure(state=tk.NORMAL)
            self.lbfgs_iters_label.configure(state=tk.NORMAL)
        self._reset_predictor()

    def _on_model_type_changed(self, *args):
        """Show/hide BRS and ZoomIn panels based on selected model type."""
        if not hasattr(self, 'zoomin_options_frame'):
            return
        model_type = self.state['model_type'].get()
        is_sam     = model_type in self._sam_types
        state      = tk.DISABLED if is_sam else tk.NORMAL
        self.zoomin_options_frame.set_frame_state(state)
        self.brs_options_frame.set_frame_state(state)
        if is_sam:
            self.net_clicks_entry.configure(state=tk.DISABLED)
            self.net_clicks_label.configure(state=tk.DISABLED)
            self.lbfgs_iters_entry.configure(state=tk.DISABLED)
            self.lbfgs_iters_label.configure(state=tk.DISABLED)
        self._reset_predictor()
        if hasattr(self, '_adapter_status_lbl'):
            self._refresh_adapter_status()

    def _reset_predictor(self, *args, **kwargs):
        model_type = self.state['model_type'].get()
        is_sam     = model_type in self._sam_types

        if is_sam:
            predictor_params = {'brs_mode': 'NoBRS'}
        else:
            brs_mode = self.state['brs_mode'].get()
            prob_thresh = self.state['prob_thresh'].get()
            net_clicks_limit = (
                None if brs_mode == 'NoBRS'
                else self.state['predictor_params']['net_clicks_limit'].get()
            )
            if self.state['zoomin_params']['use_zoom_in'].get():
                zp = {
                    'skip_clicks':    self.state['zoomin_params']['skip_clicks'].get(),
                    'target_size':    self.state['zoomin_params']['target_size'].get(),
                    'expansion_ratio':self.state['zoomin_params']['expansion_ratio'].get(),
                }
                if self.state['zoomin_params']['fixed_crop'].get():
                    zp['target_size'] = (zp['target_size'], zp['target_size'])
            else:
                zp = None

            predictor_params = {
                'brs_mode':    brs_mode,
                'prob_thresh': prob_thresh,
                'zoom_in_params': zp,
                'predictor_params': {
                    'net_clicks_limit': net_clicks_limit,
                    'max_size': self.limit_longest_size,
                },
                'brs_opt_func_params': {'min_iou_diff': 1e-3},
                'lbfgs_params':        {'maxfun': self.state['lbfgs_max_iters'].get()},
            }

        self.controller.reset_predictor(predictor_params)

    # ──────────────────────────────────────────────────────────────────────
    # Click / image update
    # ──────────────────────────────────────────────────────────────────────

    def _click_callback(self, is_positive, x, y):
        self.canvas.focus_set()
        if self.image_on_canvas is None:
            messagebox.showwarning('Warning', 'Please load an image first.', parent=self.master)
            return
        if self._check_entry(self):
            self.controller.add_click(x, y, is_positive)
            self._click_count += 1
            self._refresh_status()

    def _update_image(self, reset_canvas=False):
        image = self.controller.get_visualization(
            alpha_blend=self.state['alpha_blend'].get(),
            click_radius=self.state['click_radius'].get(),
        )
        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)
            self.image_on_canvas.register_click_callback(self._click_callback)

        self._set_click_dependent_widgets_state()
        if image is not None:
            self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)

    def _set_click_dependent_widgets_state(self):
        incomplete = self.controller.is_incomplete_mask
        after  = tk.NORMAL  if incomplete else tk.DISABLED
        before = tk.DISABLED if incomplete else tk.NORMAL

        self.finish_object_button.configure(state=after)
        self.undo_click_button.configure(state=after)
        self.reset_clicks_button.configure(state=after)
        # Auto-segment is enabled whenever an image is present, regardless
        # of click state — it implicitly resets the in-progress object.
        self.auto_segment_button.configure(
            state=tk.NORMAL if self.controller.image is not None else tk.DISABLED
        )

        # Read the live count from the clicker, not self._click_count: this runs
        # via controller.add_click()'s redraw callback, which fires *before*
        # _click_callback increments the counter, so the label would sit one
        # click behind (a placed click showing as "0 clicks placed").
        n = self._live_click_count()
        self._click_lbl.config(
            text=f'{n} click{"s" if n != 1 else ""} placed'
            if incomplete else 'Object finished' if self.filename else 'No clicks yet'
        )

        model_type = self.state['model_type'].get()
        is_sam = model_type in self._sam_types
        if not is_sam:
            self.zoomin_options_frame.set_frame_state(before)
            self.brs_options_frame.set_frame_state(before)

        if self.state['brs_mode'].get() == 'NoBRS':
            self.net_clicks_entry.configure(state=tk.DISABLED)
            self.net_clicks_label.configure(state=tk.DISABLED)
            self.lbfgs_iters_entry.configure(state=tk.DISABLED)
            self.lbfgs_iters_label.configure(state=tk.DISABLED)

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    def _check_entry(self, widget):
        ok = True
        for w in widget.winfo_children():
            ok = ok and self._check_entry(w)
        if getattr(widget, '_check_bounds', None) is not None:
            ok = ok and widget._check_bounds(widget.get(), '-1')
        return ok

    def _on_close(self):
        if messagebox.askokcancel('Quit', 'Do you want to quit?', parent=self.master):
            self.master.quit()
