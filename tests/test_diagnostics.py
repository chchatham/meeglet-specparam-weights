"""Smoke tests for diagnostic plotting functions."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from meeglet_specparam_weights import (
    meeglet_specparam_reconstruct,
    plot_fit_quality,
    plot_weight_surface,
    plot_decomposition,
    plot_parameter_trajectories,
)


@pytest.fixture(scope="module")
def result():
    """Run pipeline once for all diagnostics tests."""
    rng = np.random.default_rng(42)
    sfreq = 256.0
    n_samples = int(4 * sfreq)
    t = np.arange(n_samples) / sfreq

    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    freqs[0] = 1.0
    white = rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))
    white *= 1.0 / np.sqrt(freqs)
    white[0] = 0
    pink = np.fft.irfft(white, n=n_samples)
    signal = pink + 2.0 * np.sin(2 * np.pi * 10 * t)

    return meeglet_specparam_reconstruct(
        signal, sfreq,
        component="aperiodic",
        foi_start=2.0, foi_end=50.0, bw_oct=0.5,
        fit_stride=50, power_window=200,
        freq_range=[1, 50],
    )


def test_plot_fit_quality(result):
    fig = plot_fit_quality(result)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_plot_weight_surface(result):
    fig = plot_weight_surface(result)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_plot_decomposition(result):
    fig = plot_decomposition(result)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3


def test_plot_decomposition_time_range(result):
    fig = plot_decomposition(result, time_range=(1.0, 3.0))
    assert isinstance(fig, Figure)


def test_plot_parameter_trajectories(result):
    fig = plot_parameter_trajectories(result)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2


def test_plot_fit_quality_with_ax(result):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    returned_fig = plot_fit_quality(result, ax=ax)
    assert returned_fig is fig


def test_plot_weight_surface_with_ax(result):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    returned_fig = plot_weight_surface(result, ax=ax)
    assert returned_fig is fig
