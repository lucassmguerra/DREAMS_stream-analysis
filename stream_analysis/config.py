# -*- coding: utf-8 -*-
"""
Analysis hyperparameters for *No Stream Left Unscathed* (Arora et al.).

Every value here is the value already hardcoded in the original
``stream_analysis.py``, either as a signature default or as a literal in a
function body or a call site. Nothing has moved. The golden tests prove it.

The point of collecting them is that the paper's numerical choices should be
readable in one place and auditable in one diff, rather than scattered across
forty signatures. Function signatures keep their own literal defaults, so code
written against the old API keeps working with no config object in sight. These
dataclasses are the documented source of those literals and are what
``pipeline.build_metrics_table`` reads.

All dataclasses are frozen. Override a single value with ``replace``:

    from dataclasses import replace
    cfg = replace(DENSITY, bin_width=0.5)
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "GeometryConfig",
    "KinematicsConfig",
    "BinnedConfig",
    "DensityConfig",
    "SpectraConfig",
    "WassersteinNullFit",
    "GEOMETRY",
    "KINEMATICS",
    "BINNED",
    "DENSITY",
    "SPECTRA",
    "WASSERSTEIN_NULL_FIT",
]


@dataclass(frozen=True)
class GeometryConfig:
    """
    Settings for stream length, width and the KDE membership mask.

    Attributes
    ----------
    density_frac : float
        Fraction of particles the length span must enclose. 0.9 in the paper.
    nbins : int
        Number of phi1 bins in the per-stream histogram that approximates the
        KDE. KDE method only.
    smoothing_sigma : float
        Gaussian sigma applied to that histogram, in bin units. KDE method only.
    method : str
        Estimator, "kde" or "quantile".
    """

    density_frac: float = 0.9
    nbins: int = 100
    smoothing_sigma: float = 1.0
    method: str = "kde"


@dataclass(frozen=True)
class KinematicsConfig:
    """
    Settings for the velocity-dispersion measurement.

    Attributes
    ----------
    poly_degree : int
        Degree of the polynomial in phi1 removed from each velocity component
        before the dispersion is taken.
    use_numba : bool
        Whether to prefer the numba worker when numba is importable.
    """

    poly_degree: int = 5
    use_numba: bool = True


@dataclass(frozen=True)
class BinnedConfig:
    """
    Settings for the localized binned-statistics family.

    This is the machinery behind global disturbance, peak disturbance and width
    variation. It is generic in the binned quantity. The paper applies it to
    phi2, and the same machinery applied to v_los yields the kinematic
    analogues.

    Attributes
    ----------
    scheme : str
        "bin_size" for fixed-width bins, "equal_count" for equal-occupancy bins.
    bin_size : float
        Bin width in units of the rescaled along-stream coordinate. phi1 is
        divided by the stream length before binning, so 1/8 gives the eight bins
        used throughout the paper.
    min_particles : int
        Target occupancy per bin under the "equal_count" scheme.
    normal_ref_size : int
        Size of the reference N(0, 1) sample each bin is compared against.
    normal_ref_seed : int
        Seed for that reference sample. Part of the contract.
    quantity_detrend_order : int
        Order of the polynomial in phi1 removed from the binned quantity before
        binning. Applied as ``min(order, n_points - 1)``.
    ddof : int
        Degrees of freedom for the standard deviation of the per-bin standard
        deviations. 1 treats the bin values as a sample.
    wass_flag_threshold : float
        A bin counts as Gaussian when its Wasserstein distance normalized by the
        95% null threshold falls below this.
    ks_alpha : float
        Significance level for the per-bin KS test.
    """

    scheme: str = "bin_size"
    bin_size: float = 1 / 8.0
    min_particles: int = 300
    normal_ref_size: int = 20_000
    normal_ref_seed: int = 12345
    quantity_detrend_order: int = 3
    ddof: int = 1
    wass_flag_threshold: float = 1.00
    ks_alpha: float = 0.05


@dataclass(frozen=True)
class DensityConfig:
    """
    Settings for the linearized along-stream density.

    Attributes
    ----------
    bin_width : float
        Fixed bin width in degrees. Fixed width rather than fixed bin count, so
        that streams of different lengths are compared at the same spatial
        resolution.
    smoothing_sigma : float
        Gaussian sigma in bin units. With ``bin_width`` 0.25 and sigma 1.0 the
        effective KDE bandwidth is 0.25 degrees.
    smoothing_frac : float
        Fallback sigma as a fraction of the bin count, used only when
        ``smoothing_sigma`` is None.
    smoothing_mode : str
        Boundary mode passed to ``gaussian_filter1d``.
    detrend : str
        Trend removed before the residuals are formed. "poly", "lowpass" or
        "none".
    poly_order : int
        Polynomial order for ``detrend="poly"``.
    lowpass_frac : float
        Trend sigma as a fraction of the bin count for ``detrend="lowpass"``,
        floored at 1 bin.
    """

    bin_width: float = 0.25
    smoothing_sigma: float = 1.0
    smoothing_frac: float = 0.02
    smoothing_mode: str = "reflect"
    detrend: str = "poly"
    poly_order: int = 3
    lowpass_frac: float = 0.10


@dataclass(frozen=True)
class SpectraConfig:
    """
    Settings for the Welch power spectra and the detection metrics.

    Attributes
    ----------
    n_realizations : int
        Monte Carlo realizations used to build the sampling noise floor. The
        function's own default is 500. The paper uses 1000, which is the value
        here.
    rng_seed : int
        Seed for those realizations. Part of the contract.
    smoothing_sigma_bins : float
        Gaussian sigma applied to each resampled histogram, matched to the
        smoothing applied to the observed one.
    nperseg_frac : float
        Welch segment length as a fraction of the bin count, floored at 8.
    noverlap_frac : float
        Welch segment overlap as a fraction of the segment length.
    scaling : str
        Welch scaling, "density" or "spectrum".
    detrend : str
        Per-realization detrending, "poly" or "constant".
    poly_order : int
        Polynomial order for that detrending.
    sampling : str
        Resampling model, "multinomial" preserves the total count per
        realization, "poisson" does not.
    null_hypothesis : bool
        When True the realizations are drawn from a uniform density, which makes
        the noise floor a pure sampling floor rather than a floor that already
        contains the observed structure.
    snr_threshold : float
        Detection threshold for "peak_snr", in units of the null scatter. The
        familiar 5-sigma.
    min_bins : int
        Minimum number of frequency bins in a scanned band.
    conservative_nyquist_frac : float
        Fraction of the Nyquist frequency trusted at the top of the band.
    method : str
        Detection method. "peak_snr", the per-frequency signal-to-noise ratio
        ``(P_obs - mu_null) / sigma_null``. This is the default everywhere,
        including in ``build_metrics_table``.
    published_method : str
        The method that produced the values in *No Stream Left Unscathed*,
        "ratio95", with ``published_snr_threshold``. It is a power ratio against
        the 95th percentile of the null rather than an SNR, and was used as a
        proxy for a 5-sigma detection. Pass ``method="ratio95",
        snr_threshold=3.0`` to reproduce the published table.
    published_snr_threshold : float
        Threshold that went with "ratio95" in the published run.
    """

    n_realizations: int = 1000
    rng_seed: int = 12345
    smoothing_sigma_bins: float = 1.0
    nperseg_frac: float = 0.25
    noverlap_frac: float = 0.5
    scaling: str = "density"
    detrend: str = "poly"
    poly_order: int = 3
    sampling: str = "multinomial"
    null_hypothesis: bool = True
    snr_threshold: float = 5.0
    min_bins: int = 2
    conservative_nyquist_frac: float = 0.9
    method: str = "peak_snr"
    published_method: str = "ratio95"
    published_snr_threshold: float = 3.0


@dataclass(frozen=True)
class WassersteinNullFit:
    """
    Precomputed power-law fit to the 95% null Wasserstein distance.

    The null distribution is the Wasserstein distance between a sample of ``n``
    draws from N(0, 1) and a 20000-draw reference sample of the same
    distribution. Its 95th percentile is well described by ``A * n**p + B``.
    These coefficients were fitted once, offline, and are part of the contract.

    Attributes
    ----------
    A, p, B : float
        Coefficients of ``A * n**p + B``.
    reference_size : int
        Size of the reference sample the fit was made against.
    alpha : float
        Significance level. The fit describes the ``1 - alpha`` quantile.
    """

    A: float = 2.3
    p: float = -0.52
    B: float = 0.00590098
    reference_size: int = 20_000
    alpha: float = 0.05


#: Module-level instances at the paper's settings.
GEOMETRY = GeometryConfig()
KINEMATICS = KinematicsConfig()
BINNED = BinnedConfig()
DENSITY = DensityConfig()
SPECTRA = SpectraConfig()
WASSERSTEIN_NULL_FIT = WassersteinNullFit()
