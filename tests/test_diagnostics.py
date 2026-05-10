"""Smoke tests for diagnostic plotting functions."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from tests.conftest import make_pink_noise
from meeglet_specparam_weights import (
    meeglet_specparam_reconstruct,
    plot_fit_quality,
    plot_weight_surface,
    plot_decomposition,
    plot_parameter_trajectories,
    plot_aperiodic_coupling,
)
from meeglet_specparam_weights.wavelet_analysis import wavelet_decompose
from meeglet_specparam_weights.time_resolved_fit import time_resolved_fit
from meeglet_specparam_weights.coupling import compute_aperiodic_csd


@pytest.fixture(scope="module")
def result():
    """Run pipeline once for all diagnostics tests."""
    sfreq = 256.0
    n_samples = int(4 * sfreq)
    t = np.arange(n_samples) / sfreq

    pink = make_pink_noise(n_samples, sfreq, exponent_half=0.5)
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


def test_plot_aperiodic_coupling(result):
    coupling = compute_aperiodic_csd(result.decomposition, result.fit)
    fig = plot_aperiodic_coupling(coupling)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1
