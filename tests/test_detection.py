# -*- coding: utf-8 -*-
"""
Tests for the PSD detection metrics, including the parts that deliberately
diverge from the frozen source.

Everything else in this package reproduces
``reference/stream_analysis_source.py`` bitwise, and ``test_contract.py`` proves
it. The detection scan is the one exception, by decision, because the original
`band_snr` did not measure what its name and docstring claimed. Those behaviors
therefore have no golden. They are pinned here instead, against explicit
properties and against an independent re-derivation of the formula.

The divergences are enumerated in ``tools/cases.py`` under ``DIVERGENCES``.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

METHODS = ("peak_snr", "band_snr", "ratio95")


@pytest.fixture(scope="module")
def spectrum(api, fixtures):
    """The full spectral chain for one moderately long unperturbed stream."""
    phi1, _q, _L = fixtures.track("unperturb", 6)
    dens = api.compute_linearized_density(
        phi1, detrend="poly", smoothing_sigma=1, bin_width=0.25
    )
    psd = api.compute_welch_psd(dens["resid"], dens["w"])
    null = api.compute_sampling_psd_realizations(
        dens["hist_counts"],
        smoothing_sigma_bins=1,
        w=dens["w"],
        n_realizations=1000,
        rng_seed=12345,
    )
    return {
        "freqs": psd["freqs"],
        "psd": psd["psd"],
        "w": dens["w"],
        "psd_all": null["psd_all"],
        "nperseg": null["nperseg"],
    }


def _detect(api, spectrum, **kwargs):
    return api.detect_stream_psd_metrics(
        spectrum["freqs"],
        spectrum["psd"],
        w=spectrum["w"],
        psd_all=spectrum["psd_all"],
        nperseg=spectrum["nperseg"],
        **kwargs,
    )


# ------------------------------------------------------------------ defaults


def test_default_method_is_peak_snr(api):
    import inspect

    sig = inspect.signature(api.detect_stream_psd_metrics)
    assert sig.parameters["method"].default == "peak_snr"
    assert sig.parameters["snr_threshold"].default is None


def test_per_method_thresholds(api, spectrum):
    """None resolves to each method's own default, not to a shared number."""
    from stream_analysis.spectra import _DEFAULT_SNR_THRESHOLD

    assert _DEFAULT_SNR_THRESHOLD == {"peak_snr": 5.0, "band_snr": 3.0, "ratio95": 3.0}

    for method, expected in _DEFAULT_SNR_THRESHOLD.items():
        implicit = _detect(api, spectrum, method=method)
        explicit = _detect(api, spectrum, method=method, snr_threshold=expected)
        assert implicit == explicit, f"{method}: None did not resolve to {expected}"


def test_unknown_method_rejected_before_any_work(api):
    """
    The method name is now validated first, so a typo fails immediately rather
    than after the scalar metrics have been computed.
    """
    with pytest.raises(ValueError, match="peak_snr.*band_snr.*ratio95"):
        api.detect_stream_psd_metrics(np.linspace(0.0, 2.0, 33), np.ones(33), w=0.25, method="chi2")


# ------------------------------------------------------------------ peak_snr


def test_peak_snr_matches_the_formula(api, spectrum):
    """
    min_lambda_band is 1/f at the highest trusted frequency where
    (P_obs - mu_null) / sigma_null clears the threshold.

    Re-derived here from the raw realizations, independently of the
    implementation.
    """
    detail = _detect(api, spectrum, method="peak_snr", detailed=True)

    positive = spectrum["freqs"] > 0
    freqs = spectrum["freqs"][positive]
    obs = spectrum["psd"][positive]
    # The .copy() matters. Boolean-indexing the second axis leaves an
    # F-contiguous result, and np.std reduces it in a different order from the
    # C-contiguous copy the implementation makes, which shifts sigma_null by
    # about one part in 1e15. Same numbers, different summation order.
    mc = spectrum["psd_all"][:, positive].copy()

    mu = np.median(mc, axis=0)
    sigma = np.std(mc, axis=0, ddof=1)
    snr = np.divide(obs - mu, sigma, out=np.zeros_like(obs), where=sigma > 0)

    np.testing.assert_allclose(detail["snr_per_freq"], snr, rtol=0, atol=0)
    np.testing.assert_allclose(detail["psd_null_std"], sigma, rtol=0, atol=0)

    selected = (snr >= 5.0) & detail["trusted_mask"]
    expected = 1.0 / freqs[np.where(selected)[0].max()] if selected.any() else None
    assert detail["min_lambda_band"] == expected


def test_peak_snr_is_monotone_in_threshold(api, spectrum):
    """
    Raising the bar can only lengthen the shortest detected wavelength.

    A weaker statement than exact values, but it is the property that makes the
    metric mean anything.
    """
    previous = 0.0
    for threshold in (2.0, 3.0, 5.0, 8.0, 20.0):
        lam = _detect(api, spectrum, method="peak_snr", snr_threshold=threshold)[-1]
        if lam is None:
            break
        assert lam >= previous, f"threshold {threshold} shortened the wavelength"
        previous = lam


def test_peak_snr_returns_none_when_nothing_clears(api, spectrum):
    assert _detect(api, spectrum, method="peak_snr", snr_threshold=1e9)[-1] is None


def test_peak_snr_needs_the_realizations(api, spectrum):
    """
    sigma_null cannot be recovered from a summary, so peak_snr declines rather
    than guessing.
    """
    result = api.detect_stream_psd_metrics(
        spectrum["freqs"],
        spectrum["psd"],
        w=spectrum["w"],
        psd_95=np.percentile(spectrum["psd_all"], 95.0, axis=0)[spectrum["freqs"] > 0],
        nperseg=spectrum["nperseg"],
        method="peak_snr",
    )
    assert result[-1] is None
    assert np.isfinite(result[0])  # the scalar metrics are still computed

    detail = api.detect_stream_psd_metrics(
        spectrum["freqs"],
        spectrum["psd"],
        w=spectrum["w"],
        nperseg=spectrum["nperseg"],
        method="peak_snr",
        detailed=True,
    )
    assert "warning" in detail and "sigma_null" in detail["warning"]


# ------------------------------------------------------------------ band_snr


def test_band_snr_no_longer_saturates(api, fixtures):
    """
    The defect this fix addresses.

    Before, min_lambda_band came out as exactly 1 / f_top_trusted for every
    stream, because a band spanning the whole trusted range always clears the
    threshold and its upper edge is the top of the range. The column carried no
    per-stream information. It must now vary.
    """
    values, edges = [], []
    for i in range(6):
        phi1, _q, _L = fixtures.track("unperturb", i)
        dens = api.compute_linearized_density(
            phi1, detrend="poly", smoothing_sigma=1, bin_width=0.25
        )
        psd = api.compute_welch_psd(dens["resid"], dens["w"])
        null = api.compute_sampling_psd_realizations(
            dens["hist_counts"],
            smoothing_sigma_bins=1,
            w=dens["w"],
            n_realizations=1000,
            rng_seed=12345,
        )
        detail = api.detect_stream_psd_metrics(
            psd["freqs"],
            psd["psd"],
            w=dens["w"],
            psd_all=null["psd_all"],
            nperseg=null["nperseg"],
            method="band_snr",
            detailed=True,
        )
        positive = psd["freqs"] > 0
        f_top = psd["freqs"][positive][detail["trusted_mask"]].max()
        values.append(detail["min_lambda_band"])
        edges.append(1.0 / f_top)

    assert all(v is not None for v in values)
    at_the_edge = sum(np.isclose(v, e, rtol=1e-9) for v, e in zip(values, edges))
    assert at_the_edge < len(values), (
        "band_snr returned the trusted-band edge for every stream, which is the "
        "saturation this fix was meant to remove"
    )


def test_band_snr_uses_only_narrow_bands(api, spectrum):
    """
    min_lambda_band comes from bands of exactly min_bins frequency bins.

    Re-derived here by scanning only the narrow bands directly.
    """
    detail = _detect(api, spectrum, method="band_snr", detailed=True)

    positive = spectrum["freqs"] > 0
    freqs = spectrum["freqs"][positive]
    obs = spectrum["psd"][positive]
    mc = spectrum["psd_all"][:, positive]
    mu = np.median(mc, axis=0)

    trusted = detail["trusted_mask"]
    f_tr, obs_tr, mu_tr, mc_tr = freqs[trusted], obs[trusted], mu[trusted], mc[:, trusted]

    best = None
    for a in range(f_tr.size - 1):
        band = slice(a, a + 2)  # min_bins == 2
        f_band = f_tr[band]
        i_obs = float(trapezoid(np.clip(obs_tr[band] - mu_tr[band], 0.0, None), f_band))
        i_mc = trapezoid(np.clip(mc_tr[:, band] - mu_tr[band][None, :], 0.0, None), f_band, axis=1)
        sigma = float(np.std(i_mc, ddof=1))
        snr = (i_obs / sigma) if sigma > 0 else (np.inf if i_obs > 0 else 0.0)
        if snr >= 3.0 and f_band[-1] > 0:
            lam = 1.0 / f_band[-1]
            best = lam if best is None else min(best, lam)

    assert detail["min_lambda_band"] == best


def test_band_snr_still_lists_every_qualifying_band(api, spectrum):
    """
    Only the wavelength selection changed. The diagnostic band list is unchanged,
    so wide qualifying bands still appear in it.
    """
    detail = _detect(api, spectrum, method="band_snr", detailed=True)
    bands = detail["bands_above_threshold"]
    assert len(bands) > 0
    widths = {b[1] - b[0] + 1 for b in bands}
    assert max(widths) > 2, "expected wide bands to still be reported"


# ------------------------------------------------------------------- ratio95


def test_ratio95_is_untouched(api, spectrum, sa_ref, fixtures):
    """
    ratio95 was not part of the fix and must still match the frozen source.

    Covered bitwise by test_contract.py as well. Repeated here so that a future
    edit to the detection scan cannot quietly take ratio95 with it.
    """
    phi1, _q, _L = fixtures.track("unperturb", 6)
    dens = sa_ref.compute_linearized_density(
        phi1, detrend="poly", smoothing_sigma=1, bin_width=0.25
    )
    psd = sa_ref.compute_welch_psd(dens["resid"], dens["w"])
    null = sa_ref.compute_sampling_psd_realizations(
        dens["hist_counts"],
        smoothing_sigma_bins=1,
        w=dens["w"],
        n_realizations=1000,
        rng_seed=12345,
    )
    expected = sa_ref.detect_stream_psd_metrics(
        psd["freqs"],
        psd["psd"],
        w=dens["w"],
        psd_all=null["psd_all"],
        nperseg=null["nperseg"],
        method="ratio95",
    )
    got = _detect(api, spectrum, method="ratio95")
    assert got == expected


# -------------------------------------------------------- shared invariants


@pytest.mark.parametrize("method", METHODS)
def test_scalar_metrics_are_method_independent(api, spectrum, method):
    """
    rms_obs, rms_null and excess_power come from the trusted-band integrals and
    do not depend on the detection scan at all.
    """
    baseline = _detect(api, spectrum, method="ratio95")[:3]
    assert _detect(api, spectrum, method=method)[:3] == baseline


@pytest.mark.parametrize("method", METHODS)
def test_detailed_dict_carries_its_own_frequency_grid(api, spectrum, method):
    """
    trusted_mask indexes the DC-stripped frequency vector, which the detailed
    dict now returns alongside it. FINDINGS entry 10.
    """
    detail = _detect(api, spectrum, method=method, detailed=True)
    assert detail["freqs"].shape == detail["trusted_mask"].shape
    assert (detail["freqs"] > 0).all()
    assert detail["freqs"][detail["trusted_mask"]].size > 0


@pytest.mark.parametrize("method", METHODS)
def test_detailed_agrees_with_tuple(api, spectrum, method):
    tup = _detect(api, spectrum, method=method)
    detail = _detect(api, spectrum, method=method, detailed=True)
    assert tup == (
        detail["rms_obs"],
        detail["rms_null"],
        detail["excess_power"],
        detail["min_lambda_band"],
    )
