"""Tkinter app construction smoke test.

Builds the InteractiveDemoApp inside a withdrawn Tk root, verifies the
sidebar widgets exist, and exercises a few callbacks programmatically
without actually displaying the window. Skips automatically if Tk can
not initialise (headless CI).
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.gui


@pytest.fixture
def app(tk_root, fresh_ritm, device):
    from interactive_demo.app import InteractiveDemoApp
    a = InteractiveDemoApp(tk_root, limit_longest_size=2000,
                           device=device, model=fresh_ritm)
    yield a
    try:
        a.destroy()
    except Exception:
        pass


def test_app_constructs(app):
    assert app.controller is not None
    assert app.controller.model_type == 'ritm'


def test_model_type_combobox_has_expected_values(app):
    expected = {'RITM', 'SimpleClick', 'SAM - ViT-B', 'SAM - ViT-L',
                'SAM - ViT-H', 'SAM2 - Tiny', 'SAM2 - Small',
                'SAM2 - Base+', 'SAM2 - Large'}
    assert expected.issubset(set(app.model_types))


def test_adaptation_panel_is_built(app):
    # Buffer label and adapter status label should exist.
    assert hasattr(app, '_buffer_lbl')
    assert hasattr(app, '_adapter_status_lbl')
    assert app._buffer_lbl.cget('text').startswith('Buffer:')


def test_auto_segment_button_exists_and_disabled_initially(app):
    import tkinter as tk
    assert hasattr(app, 'auto_segment_button')
    # No image loaded yet → button should be disabled after the next
    # widget-state refresh.
    app._set_click_dependent_widgets_state()
    assert str(app.auto_segment_button['state']) == str(tk.DISABLED)


def test_toggle_online_recording_updates_controller(app):
    app._online_var.set(True)
    app._toggle_online_recording()
    assert app.controller.online_recording is True
    app._online_var.set(False)
    app._toggle_online_recording()
    assert app.controller.online_recording is False


def test_refresh_adapter_status_does_not_crash(app):
    app._refresh_adapter_status()
    assert 'no adapter' in app._adapter_status_lbl.cget('text') \
        or 'unsaved' in app._adapter_status_lbl.cget('text')
