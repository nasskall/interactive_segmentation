"""
FewShotAdaptDialog — Tkinter dialog for training a LoRA adapter (Mode A).

Operates on the *currently loaded* model held by the controller, so the
user does not need to re-pick a checkpoint. Saves the trained LoRA delta
to ``adapters/<model_type>/<name>.pt`` and applies it live.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from isegm.adaptation import (
    FewShotTrainer, FewShotConfig, DEFAULT_RANKS,
)
from isegm.data.custom_dataset import CustomDataset

TRAINING_DATA_ROOT = 'training_data'
ADAPTERS_DIR = 'adapters'


class FewShotAdaptDialog(tk.Toplevel):
    """
    Parameters
    ----------
    controller : InteractiveController
        Holds the live model + active adapter slot.
    on_adapter_applied : callable() | None
        Invoked after the LoRA adapter has been trained and applied.
    """

    def __init__(self, master, controller, on_adapter_applied=None):
        super().__init__(master)
        self.controller = controller
        self.on_adapter_applied = on_adapter_applied

        self.title("Few-Shot Domain Adaptation (LoRA)")
        self.resizable(False, False)
        self.grab_set()

        self._trainer: FewShotTrainer | None = None
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue = queue.Queue()
        self._import_folder: str | None = None

        self._build_ui()
        self._update_dataset_info()
        self._poll_queue()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = dict(padx=10, pady=5)
        model_type = self.controller.model_type
        default_rank = DEFAULT_RANKS.get(model_type, 4)

        # ── Active model ─────────────────────────────────────────────
        info = ttk.LabelFrame(self, text="Active Model", padding=8)
        info.grid(row=0, column=0, sticky='ew', **pad)
        ttk.Label(info, text=f"model_type = {model_type}").grid(row=0, column=0, sticky='w')
        ttk.Label(info, text=(
            "LoRA will be injected into this model and trained on the "
            "support set below."
        ), wraplength=460, foreground='#555').grid(row=1, column=0, sticky='w', pady=(4, 0))

        # ── Dataset ─────────────────────────────────────────────────
        ds = ttk.LabelFrame(self, text="Support Set", padding=8)
        ds.grid(row=1, column=0, sticky='ew', **pad)
        ds.columnconfigure(1, weight=1)

        self._ds_source = tk.StringVar(value='collected')
        ttk.Radiobutton(ds, text="Collected sessions:",
                        variable=self._ds_source, value='collected',
                        command=self._update_dataset_info
                        ).grid(row=0, column=0, sticky='w')
        self._collected_lbl = ttk.Label(ds, text="0 pairs")
        self._collected_lbl.grid(row=0, column=1, sticky='w', padx=5)

        ttk.Radiobutton(ds, text="Import folder:",
                        variable=self._ds_source, value='import',
                        command=self._update_dataset_info
                        ).grid(row=1, column=0, sticky='w')
        self._import_lbl = ttk.Label(ds, text="(none selected)")
        self._import_lbl.grid(row=1, column=1, sticky='w', padx=5)
        ttk.Button(ds, text="Browse…", command=self._browse_import
                   ).grid(row=1, column=2, padx=5)

        # ── LoRA / training config ─────────────────────────────────
        cfg = ttk.LabelFrame(self, text="LoRA Config", padding=8)
        cfg.grid(row=2, column=0, sticky='ew', **pad)

        self._rank_var    = tk.IntVar(value=default_rank)
        self._epochs_var  = tk.IntVar(value=20)
        self._lr_var      = tk.StringVar(value='1e-3')
        self._clicks_var  = tk.IntVar(value=3)

        for col, (lbl, w) in enumerate([
            ("Rank:",            lambda f: ttk.Spinbox(f, from_=1, to=32, width=5,
                                                        textvariable=self._rank_var)),
            ("Epochs:",          lambda f: ttk.Spinbox(f, from_=1, to=200, width=5,
                                                        textvariable=self._epochs_var)),
            ("LR:",              lambda f: ttk.Entry(f, textvariable=self._lr_var, width=8)),
            ("Clicks/iter:",     lambda f: ttk.Spinbox(f, from_=1, to=5, width=4,
                                                        textvariable=self._clicks_var)),
        ]):
            ttk.Label(cfg, text=lbl).grid(row=0, column=col * 2, sticky='e', padx=(8, 2))
            w(cfg).grid(row=0, column=col * 2 + 1, sticky='w', padx=(0, 8))

        # ── Progress ───────────────────────────────────────────────
        prog = ttk.LabelFrame(self, text="Progress", padding=8)
        prog.grid(row=3, column=0, sticky='ew', **pad)
        prog.columnconfigure(0, weight=1)

        self._progress = ttk.Progressbar(prog, orient='horizontal',
                                          length=440, mode='determinate')
        self._progress.grid(row=0, column=0, sticky='ew', pady=(0, 4))
        self._progress_lbl = ttk.Label(prog, text="")
        self._progress_lbl.grid(row=1, column=0, sticky='w')

        self._log = tk.Text(prog, height=8, width=60, state=tk.DISABLED,
                            font=('Courier', 9))
        self._log.grid(row=2, column=0, sticky='ew', pady=(4, 0))
        sb = ttk.Scrollbar(prog, orient='vertical', command=self._log.yview)
        sb.grid(row=2, column=1, sticky='ns')
        self._log.configure(yscrollcommand=sb.set)

        # ── Save / actions ─────────────────────────────────────────
        btns = ttk.Frame(self, padding=8)
        btns.grid(row=4, column=0, sticky='ew', **pad)

        ttk.Label(btns, text="Save adapter as:").grid(row=0, column=0, sticky='e', padx=(0, 4))
        self._save_name_var = tk.StringVar(
            value=f'{model_type}_adapter')
        ttk.Entry(btns, textvariable=self._save_name_var, width=22
                  ).grid(row=0, column=1, sticky='w')

        self._start_btn = ttk.Button(btns, text="Train Adapter",
                                     command=self._start_training)
        self._start_btn.grid(row=0, column=2, padx=8)

        self._stop_btn = ttk.Button(btns, text="Stop",
                                    command=self._stop_training, state=tk.DISABLED)
        self._stop_btn.grid(row=0, column=3, padx=4)

        self._save_btn = ttk.Button(btns, text="Save Adapter",
                                    command=self._save_adapter, state=tk.DISABLED)
        self._save_btn.grid(row=0, column=4, padx=4)

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _update_dataset_info(self):
        try:
            ds = CustomDataset(TRAINING_DATA_ROOT, augment=False)
            count = len(ds)
        except Exception:
            count = 0
        self._collected_lbl.config(text=f"{count} pair{'s' if count != 1 else ''}")

        if self._import_folder:
            try:
                ds2 = CustomDataset(self._import_folder, augment=False)
                self._import_lbl.config(text=f"{self._import_folder}  ({len(ds2)} pairs)")
            except Exception as exc:
                self._import_lbl.config(text=f"Error: {exc}")
        else:
            self._import_lbl.config(text="(none selected)")

    def _browse_import(self):
        folder = filedialog.askdirectory(parent=self,
                                         title="Select folder with image/mask pairs")
        if folder:
            self._import_folder = folder
            self._ds_source.set('import')
            self._update_dataset_info()

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    def _start_training(self):
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
                                 "Use 'Save for Training' in the main window, "
                                 "or import a folder.", parent=self)
            return

        try:
            lr = float(self._lr_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid learning rate.", parent=self)
            return

        cfg = FewShotConfig(
            rank=int(self._rank_var.get()),
            epochs=int(self._epochs_var.get()),
            lr=lr,
            n_clicks=int(self._clicks_var.get()),
        )

        model_type = self.controller.model_type
        sam_p = self.controller.predictor if model_type == 'sam' else None
        sam2_p = self.controller.predictor if model_type == 'sam2' else None

        self._progress['maximum'] = cfg.epochs
        self._progress['value'] = 0
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._save_btn.config(state=tk.DISABLED)

        self._trainer = FewShotTrainer(
            model=self.controller.net,
            model_type=model_type,
            dataset=dataset,
            config=cfg,
            device=self.controller.device,
            sam_predictor=sam_p,
            sam2_predictor=sam2_p,
            progress_cb=lambda ep, tot, loss: self._queue.put(('progress', ep, tot, loss)),
            log_cb=lambda msg: self._queue.put(('log', msg)),
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _stop_training(self):
        if self._trainer:
            self._trainer.stop()
        self._stop_btn.config(state=tk.DISABLED)

    def _run(self):
        try:
            self._trainer.run()
            self._queue.put(('done', None))
        except Exception as exc:
            self._queue.put(('error', str(exc)))

    def _save_adapter(self):
        if self._trainer is None or self._trainer.get_adapter_state() is None:
            messagebox.showwarning("No adapter", "Train an adapter first.", parent=self)
            return
        model_type = self.controller.model_type
        out_dir = Path(ADAPTERS_DIR) / model_type
        out_dir.mkdir(parents=True, exist_ok=True)
        name = self._save_name_var.get().strip() or f'{model_type}_adapter'
        if not name.endswith('.pt'):
            name += '.pt'
        path = out_dir / name
        try:
            self._trainer.save(str(path))
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)
            return
        # Mirror into controller's adapter slot for live use.
        self.controller._adapter_path = str(path)
        messagebox.showinfo("Saved", f"Adapter saved to:\n{path}", parent=self)

    # ------------------------------------------------------------------
    # Queue plumbing
    # ------------------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                item = self._queue.get_nowait()
                tag = item[0]
                if tag == 'progress':
                    _, ep, tot, loss = item
                    self._progress['value'] = ep
                    self._progress_lbl.config(
                        text=f"Epoch {ep}/{tot} — loss {loss:.4f}")
                elif tag == 'log':
                    self._log_append(item[1])
                elif tag == 'done':
                    self._on_done(success=True)
                elif tag == 'error':
                    self._on_done(success=False, error=item[1])
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)

    def _on_done(self, success: bool, error: str | None = None):
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        if success:
            self._save_btn.config(state=tk.NORMAL)
            self._log_append("Adaptation complete — adapter applied to live model.")
            # The trainer already injected LoRA into controller.net; sync slot.
            self.controller.apply_offline_adapter(self._trainer.get_adapter_state())
            if self.on_adapter_applied:
                self.on_adapter_applied()
        else:
            self._log_append(f"[error] {error}")
            messagebox.showerror("Training failed", error or "Unknown error", parent=self)

    def _log_append(self, text: str):
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, text + "\n")
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)
