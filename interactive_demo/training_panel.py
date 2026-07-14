"""
FineTuneDialog — Tkinter dialog for fine-tuning a segmentation model.

Opens as a top-level window.  All heavy work runs in a daemon thread;
UI updates arrive via an event queue polled with master.after().
"""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

import torch

from isegm.data.custom_dataset import CustomDataset
from isegm.inference.utils import load_is_model
from isegm.training.fine_tune import FineTuner, FineTuneConfig

TRAINING_DATA_ROOT = 'training_data'
MODEL_WEIGHTS_DIR = 'model_weights'


class FineTuneDialog(tk.Toplevel):
    """
    Parameters
    ----------
    master : tk widget
    device : str
        Torch device string ('cpu' or 'cuda').
    on_model_saved : callable(path: str) | None
        Called after the user clicks "Save Model", passing the saved path.
    """

    def __init__(self, master, device: str, on_model_saved=None):
        super().__init__(master)
        self.device = device
        self.on_model_saved = on_model_saved

        self.title("Fine-Tune Segmentation Model")
        self.resizable(False, False)
        self.grab_set()  # modal

        self._tuner: FineTuner | None = None
        self._thread: threading.Thread | None = None
        self._result_model = None
        self._base_checkpoint_path: str | None = None
        self._import_folder: str | None = None
        self._queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._refresh_model_list()
        self._update_dataset_info()
        self._poll_queue()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = dict(padx=10, pady=5)

        # ── Dataset ──────────────────────────────────────────────────
        ds_frame = ttk.LabelFrame(self, text="Dataset", padding=8)
        ds_frame.grid(row=0, column=0, sticky='ew', **pad)
        ds_frame.columnconfigure(1, weight=1)

        self._ds_source = tk.StringVar(value='collected')
        ttk.Radiobutton(ds_frame, text="Collected sessions:",
                        variable=self._ds_source, value='collected',
                        command=self._update_dataset_info
                        ).grid(row=0, column=0, sticky='w')
        self._collected_lbl = ttk.Label(ds_frame, text="0 pairs")
        self._collected_lbl.grid(row=0, column=1, sticky='w', padx=5)

        ttk.Radiobutton(ds_frame, text="Import folder:",
                        variable=self._ds_source, value='import',
                        command=self._update_dataset_info
                        ).grid(row=1, column=0, sticky='w')
        self._import_lbl = ttk.Label(ds_frame, text="(none selected)")
        self._import_lbl.grid(row=1, column=1, sticky='w', padx=5)
        ttk.Button(ds_frame, text="Browse…", command=self._browse_import
                   ).grid(row=1, column=2, padx=5)

        # ── Base model ───────────────────────────────────────────────
        mdl_frame = ttk.LabelFrame(self, text="Base Model", padding=8)
        mdl_frame.grid(row=1, column=0, sticky='ew', **pad)
        mdl_frame.columnconfigure(0, weight=1)

        self._model_var = tk.StringVar()
        self._model_combo = ttk.Combobox(mdl_frame, textvariable=self._model_var,
                                          state='readonly', width=45)
        self._model_combo.grid(row=0, column=0, sticky='ew')
        ttk.Button(mdl_frame, text="↺", width=3, command=self._refresh_model_list
                   ).grid(row=0, column=1, padx=4)

        # ── Training config ──────────────────────────────────────────
        cfg_frame = ttk.LabelFrame(self, text="Training Config", padding=8)
        cfg_frame.grid(row=2, column=0, sticky='ew', **pad)

        self._epochs_var = tk.IntVar(value=10)
        self._lr_var = tk.StringVar(value='5e-5')
        self._freeze_var = tk.BooleanVar(value=True)
        self._n_clicks_var = tk.IntVar(value=3)

        for col, (lbl, widget_fn) in enumerate([
            ("Epochs (1–100):",   lambda f: ttk.Spinbox(f, from_=1, to=100, width=6,
                                                         textvariable=self._epochs_var)),
            ("Learning rate:",    lambda f: ttk.Entry(f, textvariable=self._lr_var, width=8)),
            ("Clicks/image:",     lambda f: ttk.Spinbox(f, from_=1, to=5, width=4,
                                                         textvariable=self._n_clicks_var)),
        ]):
            ttk.Label(cfg_frame, text=lbl).grid(row=0, column=col * 2, sticky='e', padx=(8, 2))
            widget_fn(cfg_frame).grid(row=0, column=col * 2 + 1, sticky='w', padx=(0, 8))

        ttk.Checkbutton(cfg_frame, text="Freeze backbone (recommended)",
                        variable=self._freeze_var
                        ).grid(row=1, column=0, columnspan=6, sticky='w', padx=8, pady=(4, 0))

        # ── Progress ─────────────────────────────────────────────────
        prog_frame = ttk.LabelFrame(self, text="Progress", padding=8)
        prog_frame.grid(row=3, column=0, sticky='ew', **pad)
        prog_frame.columnconfigure(0, weight=1)

        self._progress = ttk.Progressbar(prog_frame, orient='horizontal',
                                          length=440, mode='determinate')
        self._progress.grid(row=0, column=0, sticky='ew', pady=(0, 4))

        self._progress_lbl = ttk.Label(prog_frame, text="")
        self._progress_lbl.grid(row=1, column=0, sticky='w')

        self._log_text = tk.Text(prog_frame, height=7, width=60, state=tk.DISABLED,
                                  font=('Courier', 9))
        self._log_text.grid(row=2, column=0, sticky='ew', pady=(4, 0))
        sb = ttk.Scrollbar(prog_frame, orient='vertical', command=self._log_text.yview)
        sb.grid(row=2, column=1, sticky='ns')
        self._log_text.configure(yscrollcommand=sb.set)

        # ── Save / buttons ───────────────────────────────────────────
        btn_frame = ttk.Frame(self, padding=8)
        btn_frame.grid(row=4, column=0, sticky='ew', **pad)

        ttk.Label(btn_frame, text="Save as:").grid(row=0, column=0, sticky='e', padx=(0, 4))
        self._save_name_var = tk.StringVar(value='finetuned_model')
        ttk.Entry(btn_frame, textvariable=self._save_name_var, width=22
                  ).grid(row=0, column=1, sticky='w')

        self._start_btn = ttk.Button(btn_frame, text="Start Training",
                                      command=self._start_training)
        self._start_btn.grid(row=0, column=2, padx=8)

        self._stop_btn = ttk.Button(btn_frame, text="Stop",
                                     command=self._stop_training, state=tk.DISABLED)
        self._stop_btn.grid(row=0, column=3, padx=4)

        self._save_btn = ttk.Button(btn_frame, text="Save Model",
                                     command=self._save_model, state=tk.DISABLED)
        self._save_btn.grid(row=0, column=4, padx=4)

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _update_dataset_info(self):
        # Collected pairs
        collected_root = Path(TRAINING_DATA_ROOT)
        try:
            ds = CustomDataset(collected_root, augment=False)
            count = len(ds)
        except Exception:
            count = 0
        self._collected_lbl.config(text=f"{count} pair{'s' if count != 1 else ''}")

        # Import info
        if self._import_folder:
            try:
                ds2 = CustomDataset(self._import_folder, augment=False)
                self._import_lbl.config(text=f"{self._import_folder}  ({len(ds2)} pairs)")
            except Exception as exc:
                self._import_lbl.config(text=f"Error: {exc}")
        else:
            self._import_lbl.config(text="(none selected)")

    def _browse_import(self):
        folder = filedialog.askdirectory(
            parent=self,
            title="Select folder with image/mask pairs",
        )
        if folder:
            self._import_folder = folder
            self._ds_source.set('import')
            self._update_dataset_info()

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _refresh_model_list(self):
        weights_dir = Path(MODEL_WEIGHTS_DIR)
        if not weights_dir.exists():
            self._model_combo['values'] = []
            return
        pths = sorted(p.name for p in weights_dir.glob('*.pth'))
        self._model_combo['values'] = pths
        if pths and not self._model_var.get():
            self._model_var.set(pths[0])

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    def _start_training(self):
        # Validate dataset
        source = self._ds_source.get()
        root = self._import_folder if source == 'import' else TRAINING_DATA_ROOT
        try:
            dataset = CustomDataset(root, augment=True)
        except Exception as exc:
            messagebox.showerror("Error", f"Cannot load dataset:\n{exc}", parent=self)
            return
        if len(dataset) == 0:
            messagebox.showerror("Error",
                "No image/mask pairs found.\n"
                "Use 'Save for Training' in the main window, or import a folder.",
                parent=self)
            return

        # Validate model
        model_name = self._model_var.get()
        if not model_name:
            messagebox.showerror("Error", "Select a base model.", parent=self)
            return
        checkpoint_path = str(Path(MODEL_WEIGHTS_DIR) / model_name)
        self._base_checkpoint_path = checkpoint_path

        try:
            lr = float(self._lr_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid learning rate.", parent=self)
            return

        cfg = FineTuneConfig(
            epochs=self._epochs_var.get(),
            lr=lr,
            freeze_backbone=self._freeze_var.get(),
            n_clicks=self._n_clicks_var.get(),
        )

        # Load model for training
        self._log_append(f"Loading base model: {model_name} …")
        try:
            model = load_is_model(checkpoint_path, self.device, cpu_dist_maps=True)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load model:\n{exc}", parent=self)
            return

        self._progress['maximum'] = cfg.epochs
        self._progress['value'] = 0
        self._progress_lbl.config(text="")
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._save_btn.config(state=tk.DISABLED)

        self._tuner = FineTuner(
            model=model,
            base_checkpoint_path=checkpoint_path,
            dataset=dataset,
            config=cfg,
            progress_cb=lambda ep, tot, loss: self._queue.put(('progress', ep, tot, loss)),
            log_cb=lambda msg: self._queue.put(('log', msg)),
        )
        self._thread = threading.Thread(target=self._run_tuner, daemon=True)
        self._thread.start()

    def _run_tuner(self):
        self._tuner.run()
        self._queue.put(('done',))

    def _stop_training(self):
        if self._tuner:
            self._tuner.stop()
        self._stop_btn.config(state=tk.DISABLED)

    def _save_model(self):
        if self._tuner is None or self._tuner.get_result() is None:
            messagebox.showwarning("Warning", "No trained model to save.", parent=self)
            return

        name = self._save_name_var.get().strip()
        if not name:
            name = 'finetuned_model'
        if not name.endswith('.pth'):
            name += '.pth'

        save_path = str(Path(MODEL_WEIGHTS_DIR) / name)

        try:
            self._tuner.save_checkpoint(save_path)
            messagebox.showinfo("Saved", f"Model saved to:\n{save_path}", parent=self)
            self._refresh_model_list()
            if self.on_model_saved:
                self.on_model_saved(save_path)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save:\n{exc}", parent=self)

    # ------------------------------------------------------------------
    # Queue polling
    # ------------------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                item = self._queue.get_nowait()
                if item[0] == 'log':
                    self._log_append(item[1])
                elif item[0] == 'progress':
                    _, ep, tot, loss = item
                    self._progress['value'] = ep
                    self._progress_lbl.config(
                        text=f"Epoch {ep}/{tot}   loss: {loss:.4f}"
                    )
                elif item[0] == 'done':
                    self._on_training_done()
        except queue.Empty:
            pass
        self.after(200, self._poll_queue)

    def _on_training_done(self):
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        if self._tuner and self._tuner.get_result() is not None:
            self._save_btn.config(state=tk.NORMAL)
            self._log_append("✓ Training finished. Click 'Save Model' to export.")

    # ------------------------------------------------------------------
    # Log helper
    # ------------------------------------------------------------------

    def _log_append(self, msg: str):
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + '\n')
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)
