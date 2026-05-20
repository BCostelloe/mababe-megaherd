import math
import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Existing binning utilities
# ---------------------------
def get_bin_counts(tracks, bin_length):
    """
    Count the number of buffalo crossing the count line in each time bin.

    Parameters:
        tracks: iterable of track dictionaries with keys 'track' and 'first_frame'
        bin_length: duration of each bin in seconds
    """
    frames_per_bin = bin_length * 30
    n_bins = math.ceil(368 / bin_length)
    len_last_bin = 368 % bin_length

    counts = []
    tracks_counted = []

    for seg in np.arange(n_bins):
        count = 0
        start_frame = frames_per_bin * seg
        end_frame = frames_per_bin * (seg + 1) - 1
        if seg == (n_bins - 1):
            end_frame = 11040
        for n, t in enumerate(tracks):
            track = t['track']
            yvals = track[:, 0]
            if (yvals > 100).any:
                cross_i = np.min(np.where(yvals > 100))
                cross_frame = cross_i + t['first_frame']
                if (cross_frame <= end_frame) & (cross_frame >= start_frame):
                    count += 1
                    tracks_counted.append(n)
        if seg == (n_bins - 1) and len_last_bin > 0:
            count = int((bin_length * count) / len_last_bin)
        counts.append(count)

    n_tracks_counted = len(tracks_counted)
    if n_tracks_counted != 2972:
        print(f'Track count is {n_tracks_counted} , but should be 2972.')

    return counts



def get_manual_bin_counts(manual_counts, bin_length):
    counts = []
    for n, c in enumerate(manual_counts):
        if n < 18:
            n_newbins = int(20 / bin_length)
            subcount = c / n_newbins
            counts.extend([subcount] * n_newbins)
        if n == 18:
            n_newbins = 8 / bin_length
            n_fullbins = math.floor(8 / bin_length)
            last_binlength = 8 % bin_length
            subcount = c / n_newbins
            if last_binlength == 0:
                counts.extend([subcount] * int(n_newbins))
            else:
                count_fullbins = c / n_newbins
                last_bin_prop = n_newbins - n_fullbins
                count_lastbin = (c * last_bin_prop) / n_newbins
                counts.extend([count_fullbins] * n_fullbins)
                counts.extend([count_lastbin])
    return counts


# ---------------------------
# Overdispersion summaries
# ---------------------------
def summarize_dispersion(counts, label=None):
    """
    Summarize empirical overdispersion in observed counts.

    Returns a one-row DataFrame with mean, variance, variance-to-mean ratio,
    and a simple overdispersion flag.
    """
    y = np.asarray(counts, dtype=float)
    mean_y = float(np.mean(y))
    var_y = float(np.var(y, ddof=1)) if len(y) > 1 else 0.0
    vmr = np.nan if mean_y == 0 else var_y / mean_y

    return pd.DataFrame({
        "label": [label if label is not None else "counts"],
        "n_bins": [len(y)],
        "mean": [mean_y],
        "variance": [var_y],
        "variance_to_mean_ratio": [vmr],
        "overdispersed_relative_to_poisson": [bool(vmr > 1) if not np.isnan(vmr) else False],
    })


# ---------------------------
# Model internals
# ---------------------------
def _gaussian_intensity(time, params):
    A = params["A"]
    mu = params["mu"]
    sigma = params["sigma"]
    return A * pm.math.exp(-0.5 * ((time - mu) / sigma) ** 2)



def _lognormal_intensity(time, params):
    scale = params["scale"]
    mu_log = params["mu_log"]
    sigma_log = params["sigma_log"]
    time = pm.math.maximum(time, 1e-6)
    return scale * (1.0 / (time * sigma_log * np.sqrt(2.0 * np.pi))) * pm.math.exp(
        -0.5 * ((pm.math.log(time) - mu_log) / sigma_log) ** 2
    )



def _build_intensity(intensity_name, time, time_extended):
    if intensity_name == "gaussian":
        A = pm.HalfNormal("A", sigma=200)
        mu = pm.Uniform("mu", lower=0, upper=float(np.max(time_extended)))
        sigma = pm.HalfNormal("sigma", sigma=100)
        params = {"A": A, "mu": mu, "sigma": sigma}
        lambda_obs = _gaussian_intensity(time, params)
        lambda_future = _gaussian_intensity(time_extended, params)
    elif intensity_name == "lognormal":
        scale = pm.HalfNormal("scale", sigma=1000)
        mu_log = pm.Normal("mu_log", mu=np.log(np.median(time_extended)), sigma=1.5)
        sigma_log = pm.HalfNormal("sigma_log", sigma=1.0)
        params = {"scale": scale, "mu_log": mu_log, "sigma_log": sigma_log}
        lambda_obs = _lognormal_intensity(time, params)
        lambda_future = _lognormal_intensity(time_extended, params)
    else:
        raise ValueError("intensity_name must be 'gaussian' or 'lognormal'.")

    return lambda_obs, lambda_future



def _fit_single_model(
    counts,
    bin_width,
    family="poisson",
    intensity="gaussian",
    forecast_seconds=500,
    random_seed=27,
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.95,
):
    """
    Fit one model and return trace, posterior predictive samples, and time vectors.

    family: 'poisson' or 'negative_binomial'
    intensity: 'gaussian' or 'lognormal'
    """
    counts = np.asarray(counts, dtype=int)
    T = len(counts)
    forecast_bins = int(forecast_seconds / bin_width)

    time = np.arange(T, dtype=float) * bin_width + 1.0
    time_extended = np.arange(T + forecast_bins, dtype=float) * bin_width + 1.0

    with pm.Model() as model:
        lambda_obs, lambda_future = _build_intensity(intensity, time, time_extended)

        mu_obs = pm.Deterministic("mu_obs", lambda_obs * bin_width)
        mu_future = pm.Deterministic("mu_future", lambda_future * bin_width)
        lambda_future_det = pm.Deterministic("lambda_future", lambda_future)

        if family == "poisson":
            y_obs = pm.Poisson("y_obs", mu=mu_obs, observed=counts)
        elif family == "negative_binomial":
            alpha = pm.HalfNormal("alpha", sigma=20)
            y_obs = pm.NegativeBinomial("y_obs", mu=mu_obs, alpha=alpha, observed=counts)
        else:
            raise ValueError("family must be 'poisson' or 'negative_binomial'.")

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
        )

        # Define future counts only after fitting so they are generated via
        # posterior predictive sampling rather than sampled inside the MCMC chain.
        if family == "poisson":
            y_future = pm.Poisson("y_future", mu=mu_future, shape=len(time_extended))
        else:
            y_future = pm.NegativeBinomial("y_future", mu=mu_future, alpha=alpha, shape=len(time_extended))

        pred_trace = pm.sample_posterior_predictive(
            trace,
            var_names=["y_obs", "lambda_future", "mu_future", "y_future"],
            random_seed=random_seed,
            progressbar=False,
        )

    return {
        "model": model,
        "trace": trace,
        "pred_trace": pred_trace,
        "counts": counts,
        "time": time,
        "time_extended": time_extended,
        "T": T,
        "forecast_bins": forecast_bins,
        "family": family,
        "intensity": intensity,
        "bin_width": bin_width,
    }



def _extract_forecast_summary(fit_result, speed_factor, real_time_threshold):
    T = fit_result["T"]
    bin_width = fit_result["bin_width"]
    threshold_per_bin = real_time_threshold * (bin_width * speed_factor)

    y_pred = fit_result["pred_trace"].posterior_predictive["y_future"]
    mean_forecast = y_pred.mean(dim=("chain", "draw")).values
    hdi = az.hdi(y_pred, hdi_prob=0.95)
    lower_ci = hdi.sel(hdi="lower")["y_future"].values
    upper_ci = hdi.sel(hdi="higher")["y_future"].values

    future_start = T
    upper_future = upper_ci[future_start:]
    below_threshold = np.where(upper_future < threshold_per_bin)[0]
    threshold_crossed = len(below_threshold) > 0

    if threshold_crossed:
        cutoff_bin = int(below_threshold[0])
        cutoff_time = fit_result["time_extended"][future_start + cutoff_bin]
    else:
        cutoff_bin = len(upper_future)
        cutoff_time = fit_result["time_extended"][-1]

    missed = {
        "Lower bound (95% CI)": lower_ci[future_start:future_start + cutoff_bin].sum(),
        "Posterior mean": mean_forecast[future_start:future_start + cutoff_bin].sum(),
        "Upper bound (95% CI)": upper_ci[future_start:future_start + cutoff_bin].sum(),
    }

    fit_result = dict(fit_result)
    fit_result.update({
        "mean": mean_forecast,
        "lower": lower_ci,
        "upper": upper_ci,
        "cutoff_bin": cutoff_bin,
        "cutoff_time": cutoff_time,
        "threshold_crossed": threshold_crossed,
        "missed_counts": missed,
        "real_time_threshold": real_time_threshold,
    })
    return fit_result


# ---------------------------
# Public model wrappers
# ---------------------------
def fit_and_forecast_poisson_gaussian(
    counts,
    bin_width,
    speed_factor,
    real_time_threshold,
    forecast_seconds=500,
    random_seed=27,
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.95,
):
    fit = _fit_single_model(
        counts=counts,
        bin_width=bin_width,
        family="poisson",
        intensity="gaussian",
        forecast_seconds=forecast_seconds,
        random_seed=random_seed,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
    )
    return _extract_forecast_summary(fit, speed_factor=speed_factor, real_time_threshold=real_time_threshold)



def fit_and_forecast_nb_gaussian(
    counts,
    bin_width,
    speed_factor,
    real_time_threshold,
    forecast_seconds=500,
    random_seed=27,
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.95,
):
    fit = _fit_single_model(
        counts=counts,
        bin_width=bin_width,
        family="negative_binomial",
        intensity="gaussian",
        forecast_seconds=forecast_seconds,
        random_seed=random_seed,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
    )
    return _extract_forecast_summary(fit, speed_factor=speed_factor, real_time_threshold=real_time_threshold)



def fit_and_forecast_poisson_lognormal(
    counts,
    bin_width,
    speed_factor,
    real_time_threshold,
    forecast_seconds=500,
    random_seed=27,
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.95,
):
    fit = _fit_single_model(
        counts=counts,
        bin_width=bin_width,
        family="poisson",
        intensity="lognormal",
        forecast_seconds=forecast_seconds,
        random_seed=random_seed,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
    )
    return _extract_forecast_summary(fit, speed_factor=speed_factor, real_time_threshold=real_time_threshold)


def fit_and_forecast_nb_lognormal(
    counts,
    bin_width,
    speed_factor,
    real_time_threshold,
    forecast_seconds=500,
    random_seed=27,
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.95,
):
    fit = _fit_single_model(
        counts=counts,
        bin_width=bin_width,
        family="negative_binomial",
        intensity="lognormal",
        forecast_seconds=forecast_seconds,
        random_seed=random_seed,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
    )
    return _extract_forecast_summary(fit, speed_factor=speed_factor, real_time_threshold=real_time_threshold)


# Backward-compatible alias for your current primary model
fit_and_forecast = fit_and_forecast_poisson_gaussian


# ---------------------------
# Model comparison tables
# ---------------------------
def compare_models_for_one_series(
    counts,
    bin_width,
    speed_factor,
    real_time_threshold,
    forecast_seconds=500,
    random_seed=27,
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.95,
    include_nb_lognormal=True,
):
    """
    Fit reviewer-relevant models and return a summary table plus fit objects.

    By default this reproduces the original three-way comparison:
    Poisson-Gaussian, NB-Gaussian, and Poisson-Lognormal.
    Set include_nb_lognormal=True to also fit NB-Lognormal.
    """
    fits = {
        "Poisson-Gaussian": fit_and_forecast_poisson_gaussian(
            counts, bin_width, speed_factor, real_time_threshold,
            forecast_seconds=forecast_seconds, random_seed=random_seed,
            draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        ),
        "NB-Gaussian": fit_and_forecast_nb_gaussian(
            counts, bin_width, speed_factor, real_time_threshold,
            forecast_seconds=forecast_seconds, random_seed=random_seed,
            draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        ),
        # "Poisson-Lognormal": fit_and_forecast_poisson_lognormal(
        #     counts, bin_width, speed_factor, real_time_threshold,
        #     forecast_seconds=forecast_seconds, random_seed=random_seed,
        #     draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        # ),
    }

    if include_nb_lognormal:
        fits["NB-Lognormal"] = fit_and_forecast_nb_lognormal(
            counts, bin_width, speed_factor, real_time_threshold,
            forecast_seconds=forecast_seconds, random_seed=random_seed,
            draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        )

    rows = []
    for model_name, fit in fits.items():
        rows.append({
            "model": model_name,
            "bin_width_s": bin_width,
            "threshold_crossed_within_window": bool(fit.get("threshold_crossed", False)),
            "cutoff_time_video_s": float(fit["cutoff_time"]),
            "missed_lower_95": float(fit["missed_counts"]["Lower bound (95% CI)"]),
            "missed_mean": float(fit["missed_counts"]["Posterior mean"]),
            "missed_upper_95": float(fit["missed_counts"]["Upper bound (95% CI)"]),
        })

    return pd.DataFrame(rows), fits


def compare_nb_shapes_for_one_series(
    counts,
    bin_width,
    speed_factor,
    real_time_threshold,
    forecast_seconds=500,
    random_seed=27,
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.99,
):
    """
    Direct comparison requested after reviewer feedback:
    NB-Gaussian versus NB-Lognormal.
    """
    fits = {
        "NB-Gaussian": fit_and_forecast_nb_gaussian(
            counts, bin_width, speed_factor, real_time_threshold,
            forecast_seconds=forecast_seconds, random_seed=random_seed,
            draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        ),
        "NB-Lognormal": fit_and_forecast_nb_lognormal(
            counts, bin_width, speed_factor, real_time_threshold,
            forecast_seconds=forecast_seconds, random_seed=random_seed,
            draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        ),
    }

    rows = []
    for model_name, fit in fits.items():
        ppc = posterior_predictive_check_summary(fit).iloc[0]
        rows.append({
            "model": model_name,
            "bin_width_s": bin_width,
            "threshold_crossed_within_window": bool(fit.get("threshold_crossed", False)),
            "cutoff_time_video_s": float(fit["cutoff_time"]),
            "missed_lower_95": float(fit["missed_counts"]["Lower bound (95% CI)"]),
            "missed_mean": float(fit["missed_counts"]["Posterior mean"]),
            "missed_upper_95": float(fit["missed_counts"]["Upper bound (95% CI)"]),
            "bayesian_p_variance": float(ppc["bayesian_p_variance"]),
            "bayesian_p_max": float(ppc["bayesian_p_max"]),
            "proportion_observed_points_outside_95_ppi": float(ppc["proportion_observed_points_outside_95_ppi"]),
        })

    return pd.DataFrame(rows), fits


# ---------------------------
# Posterior predictive checks
# ---------------------------
def posterior_predictive_check_summary(fit_result):
    """
    Numeric PPC summaries for manuscript/reporting.
    """
    obs = np.asarray(fit_result["counts"], dtype=float)
    y_rep = fit_result["pred_trace"].posterior_predictive["y_obs"]
    y_rep_np = np.asarray(y_rep).reshape(-1, len(obs))

    rep_mean = y_rep_np.mean(axis=1)
    rep_var = y_rep_np.var(axis=1, ddof=1)
    rep_max = y_rep_np.max(axis=1)

    obs_mean = obs.mean()
    obs_var = obs.var(ddof=1) if len(obs) > 1 else 0.0
    obs_max = obs.max()

    # Pointwise predictive interval coverage on observed bins
    hdi = az.hdi(y_rep, hdi_prob=0.95)
    lower = hdi.sel(hdi="lower")["y_obs"].values
    upper = hdi.sel(hdi="higher")["y_obs"].values
    outside_95 = (obs < lower) | (obs > upper)

    return pd.DataFrame({
        "model": [f"{fit_result['family']}-{fit_result['intensity']}"],
        "observed_mean": [obs_mean],
        "ppc_mean_median": [float(np.median(rep_mean))],
        "bayesian_p_mean": [float(np.mean(rep_mean >= obs_mean))],
        "observed_variance": [obs_var],
        "ppc_variance_median": [float(np.median(rep_var))],
        "bayesian_p_variance": [float(np.mean(rep_var >= obs_var))],
        "observed_max": [obs_max],
        "ppc_max_median": [float(np.median(rep_max))],
        "bayesian_p_max": [float(np.mean(rep_max >= obs_max))],
        "n_observed_points_outside_95_ppi": [int(outside_95.sum())],
        "proportion_observed_points_outside_95_ppi": [float(outside_95.mean())],
    })



def plot_posterior_predictive_checks(fit_result, title=None):
    """
    Two-panel PPC figure:
    1) time-series observed vs posterior predictive interval on observed data
    2) histogram PPC for the maximum count
    """
    obs = np.asarray(fit_result["counts"], dtype=float)
    time = fit_result["time"]
    y_rep = fit_result["pred_trace"].posterior_predictive["y_obs"]
    y_rep_np = np.asarray(y_rep).reshape(-1, len(obs))

    mean_rep = y_rep_np.mean(axis=0)
    hdi = az.hdi(y_rep, hdi_prob=0.95)
    lower = hdi.sel(hdi="lower")["y_obs"].values
    upper = hdi.sel(hdi="higher")["y_obs"].values

    rep_max = y_rep_np.max(axis=1)
    obs_max = obs.max()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(time, obs, 'ko', label='Observed counts')
    axes[0].plot(time, mean_rep, color='tab:blue', label='Posterior predictive mean')
    axes[0].fill_between(time, lower, upper, color='tab:blue', alpha=0.3, label='95% predictive interval')
    axes[0].set_xlabel('Video time (s)')
    axes[0].set_ylabel('Buffalo per bin')
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].legend(loc='upper right')

    axes[1].hist(rep_max, bins=30, alpha=0.7)
    axes[1].axvline(obs_max, color='red', linestyle='--', linewidth=2, label='Observed maximum')
    axes[1].set_xlabel('Maximum count across observed bins')
    axes[1].set_ylabel('Posterior predictive draws')
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].legend(loc='upper right')

    if title is None:
        title = f"Posterior predictive checks: {fit_result['family']}-{fit_result['intensity']}"
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------------------------
# Plotting helpers for forecasts
# ---------------------------
def get_cutoff_times_with_crossing(
    pred_trace,
    bin_width,
    speed_factor,
    thresholds,
    forecast_seconds,
    T,
):
    import numpy as np
    import arviz as az

    forecast_bins = int(forecast_seconds / bin_width)
    time_extended = np.arange(T + forecast_bins) * bin_width + 1
    future_start = T

    upper_ci = (
        az.hdi(pred_trace.posterior_predictive["y_future"], hdi_prob=0.95)
        .sel(hdi="higher")["y_future"]
        .values
    )

    upper_future = upper_ci[future_start:]

    cutoffs = {}

    for threshold in thresholds:
        threshold_per_bin = threshold * (bin_width * speed_factor)

        crossed = upper_future < threshold_per_bin

        if crossed.any():
            cutoff_bin = np.argmax(crossed)
            cutoff_time = time_extended[future_start + cutoff_bin]
            threshold_crossed = True
        else:
            cutoff_bin = None
            cutoff_time = None
            threshold_crossed = False

        cutoffs[threshold] = {
            "cutoff_bin": cutoff_bin,
            "cutoff_time": cutoff_time,
            "threshold_crossed": threshold_crossed,
            "threshold_per_bin": threshold_per_bin,
        }

    return cutoffs, time_extended

def add_cutoff_lines(
    ax,
    cutoffs,
    thresholds,
    colors=("red", "orange", "green"),
    base_linewidth=2,
    offset_fraction=0.08,
):
    """
    Add cutoff lines to an axis.

    Thresholds that are not crossed are not plotted.

    If multiple thresholds cross at the same time, lines are slightly offset
    so all remain visible.
    """
    import numpy as np

    dash_patterns = [
        (0, (6, 3)),          # dashed
        (0, (2, 2)),          # dotted-ish
        (0, (8, 2, 2, 2)),    # dash-dot-ish
        (0, (10, 3)),
        (0, (1, 2)),
    ]

    # Keep only crossed thresholds
    crossed = {
        threshold: cutoffs[threshold]["cutoff_time"]
        for threshold in thresholds
        if cutoffs[threshold]["threshold_crossed"]
    }

    if len(crossed) == 0:
        return

    # Group thresholds by identical cutoff time
    time_to_thresholds = {}
    for threshold, cutoff_time in crossed.items():
        time_to_thresholds.setdefault(cutoff_time, []).append(threshold)

    for cutoff_time, grouped_thresholds in time_to_thresholds.items():
        n = len(grouped_thresholds)

        # Small offsets in units of x-axis/video seconds
        # Example for 2s bins: 0.08 * 2 = 0.16 s
        offsets = np.linspace(
            -offset_fraction,
            offset_fraction,
            n
        )

        for j, threshold in enumerate(grouped_thresholds):
            threshold_index = list(thresholds).index(threshold)
            color = colors[threshold_index % len(colors)]
            dash = dash_patterns[threshold_index % len(dash_patterns)]

            # Offset by a fraction of one video bin width only if overlapping
            adjusted_time = cutoff_time + offsets[j]

            ax.axvline(
                adjusted_time,
                color=color,
                linestyle=dash,
                linewidth=base_linewidth,
                label=f"{threshold} buffalo/s cutoff",
            )

def plot_single_forecast_panel(counts, pred_trace, bin_width, speed_factor, thresholds, forecast_seconds=500):
    T = len(counts)
    y_pred = pred_trace.posterior_predictive["y_future"]
    mean_forecast = y_pred.mean(dim=("chain", "draw")).values
    hdi = az.hdi(y_pred, hdi_prob=0.95)
    lower_ci = hdi.sel(hdi="lower")["y_future"].values
    upper_ci = hdi.sel(hdi="higher")["y_future"].values

    cutoffs, time_extended = get_cutoff_times(pred_trace, bin_width, speed_factor, thresholds, forecast_seconds, T)

    plt.figure(figsize=(12, 6))
    time = np.arange(T) * bin_width + 1
    plt.plot(time, counts, 'ko', label='Observed')
    plt.plot(time_extended, mean_forecast, 'b-', label='Forecast mean')
    plt.fill_between(time_extended, lower_ci, upper_ci, color='blue', alpha=0.3, label='95% CI')
    plt.axvline(time[-1], color='gray', linestyle='--', label='End of video')

    colors = ['red', 'orange', 'green']
    for rt, color in zip(thresholds, colors):
        plt.axvline(cutoffs[rt], color=color, linestyle=':', label=f'Cutoff @ {rt} buff/s')

    plt.title("Buffalo Crossing Forecast – 5s Bins")
    plt.xlabel("Video Time (s)")
    plt.ylabel("Buffalo per Bin")
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.tight_layout()
    plt.xlim(-5, 870)
    plt.show()



def plot_multi_bin_forecasts_shared_clean(count_lists, pred_traces, bin_sizes, speed_factor, thresholds, forecast_seconds=500):
    n = len(bin_sizes)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4.5 * n), sharex=True)

    colors = ['red', 'orange', 'green']

    for i, (counts, trace, bw) in enumerate(zip(count_lists, pred_traces, bin_sizes)):
        T = len(counts)
        y_pred = trace.posterior_predictive["y_future"]
        mean_forecast = y_pred.mean(dim=("chain", "draw")).values
        hdi = az.hdi(y_pred, hdi_prob=0.95)
        lower_ci = hdi.sel(hdi="lower")["y_future"].values
        upper_ci = hdi.sel(hdi="higher")["y_future"].values
        cutoffs, time_extended = get_cutoff_times(trace, bw, speed_factor, thresholds, forecast_seconds, T)

        time = np.arange(T) * bw + 1
        ax = axes[i]
        ax.plot(time, counts, 'ko', label='Observed')
        ax.plot(time_extended, mean_forecast, 'b-', label='Forecast mean')
        ax.fill_between(time_extended, lower_ci, upper_ci, color='blue', alpha=0.3, label='95% CI')
        ax.axvline(time[-1], color='gray', linestyle='--', label='End of video')

        for rt, color in zip(thresholds, colors):
            ax.axvline(cutoffs[rt], color=color, linestyle=':', label=f'{rt} buff/s')

        ax.set_title(f"Bin Size: {bw}s")
        ax.set_ylabel("Buffalo per Bin")
        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    axes[-1].set_xlabel("Video Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.xlim(-5, 870)
    plt.show()

def plot_forecast_by_bin_size(
    fits_list,
    counts_list,
    bin_sizes,
    model_name="NB-Gaussian",
    speed_factor=3.5,
    thresholds=(0.2, 0.1, 0.05),
    forecast_seconds=500,
    save=False,
    filepath=None,
    dpi=300,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import arviz as az

    n = len(bin_sizes)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4.2 * n), sharex=True)

    if n == 1:
        axes = [axes]

    threshold_colors = ("red", "orange", "green")

    for ax, fits, counts, bw in zip(axes, fits_list, counts_list, bin_sizes):
        fit = fits[model_name]
        T = len(counts)

        y_pred = fit["pred_trace"].posterior_predictive["y_future"]
        mean_pred = y_pred.mean(dim=("chain", "draw")).values

        hdi = az.hdi(y_pred, hdi_prob=0.95)
        lower = hdi.sel(hdi="lower")["y_future"].values
        upper = hdi.sel(hdi="higher")["y_future"].values

        time = np.arange(T) * bw + 1
        time_extended = fit["time_extended"]

        ax.plot(time, counts, "ko", markersize=4, label="Observed")
        ax.plot(time_extended, mean_pred, color="blue", label="Posterior mean")
        ax.fill_between(time_extended, lower, upper, color="blue", alpha=0.25, label="95% PPI")
        ax.axvline(time[-1], color="gray", linestyle="--", label="End of video")

        cutoffs, _ = get_cutoff_times_with_crossing(
            fit["pred_trace"],
            bin_width=bw,
            speed_factor=speed_factor,
            thresholds=thresholds,
            forecast_seconds=forecast_seconds,
            T=T,
        )

        add_cutoff_lines(
            ax=ax,
            cutoffs=cutoffs,
            thresholds=thresholds,
            colors=threshold_colors,
            offset_fraction=0.08 * bw,
        )

        ax.set_title(f"{model_name}, {bw}s bins")
        ax.set_ylabel("Buffalo per bin")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Video time (s)")

    handles, labels = axes[0].get_legend_handles_labels()

    # Remove duplicate labels while preserving order
    unique = dict(zip(labels, handles))
    axes[0].legend(
        unique.values(),
        unique.keys(),
        loc="upper left",             # anchor corner of legend
        bbox_to_anchor=(0.55, 0.98),  # 🔥 precise position (x, y in axes coords)
        frameon=True,
        framealpha=1,
        facecolor="white",
        edgecolor="black"
    )

    plt.tight_layout()

    if save:
        if filepath is None:
            raise ValueError("Provide filepath when save=True.")
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

    plt.show()
    return fig, axes

def plot_forecast_by_model_5s(
    fits,
    counts,
    bin_width=5,
    model_names=("NB-Gaussian", "NB-Lognormal"),
    speed_factor=3.5,
    thresholds=(0.2, 0.1, 0.05),
    forecast_seconds=500,
    save=False,
    filepath=None,
    dpi=300,
):

    n = len(model_names)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4.2 * n), sharex=True)

    if n == 1:
        axes = [axes]

    threshold_colors = ("red", "orange", "green")
    T = len(counts)
    time = np.arange(T) * bin_width + 1

    for ax, model_name in zip(axes, model_names):
        fit = fits[model_name]

        y_pred = fit["pred_trace"].posterior_predictive["y_future"]
        mean_pred = y_pred.mean(dim=("chain", "draw")).values

        hdi = az.hdi(y_pred, hdi_prob=0.95)
        lower = hdi.sel(hdi="lower")["y_future"].values
        upper = hdi.sel(hdi="higher")["y_future"].values

        time_extended = fit["time_extended"]

        ax.plot(time, counts, "ko", markersize=4, label="Observed")
        ax.plot(time_extended, mean_pred, color="blue", label="Posterior mean")
        ax.fill_between(time_extended, lower, upper, color="blue", alpha=0.25, label="95% PPI")
        ax.axvline(time[-1], color="gray", linestyle="--", label="End of video")

        cutoffs, _ = get_cutoff_times_with_crossing(
            fit["pred_trace"],
            bin_width=bin_width,
            speed_factor=speed_factor,
            thresholds=thresholds,
            forecast_seconds=forecast_seconds,
            T=T,
        )

        add_cutoff_lines(
            ax=ax,
            cutoffs=cutoffs,
            thresholds=thresholds,
            colors=threshold_colors,
            offset_fraction=0.08 * bin_width,
        )

        ax.set_title(model_name)
        ax.set_ylabel("Buffalo per 5s bin")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Video time (s)")

    handles, labels = axes[0].get_legend_handles_labels()

    # Remove duplicate labels while preserving order
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), loc="upper right", frameon=True)

    plt.tight_layout()

    if save:
        if filepath is None:
            raise ValueError("Provide filepath when save=True.")
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

    plt.show()
    #return fig, axes

def plot_ppc_model_comparison(
    fits,
    counts,
    bin_width,
    model_names=("Poisson-Gaussian", "NB-Gaussian", "NB-Lognormal"),
    save=False,
    filepath=None,
    dpi=300,
    n_bins=30,  # control histogram resolution
):
    import numpy as np
    import matplotlib.pyplot as plt
    import arviz as az

    counts = np.asarray(counts)
    T = len(counts)
    time = np.arange(T) * bin_width + 1

    n_models = len(model_names)

    fig, axes = plt.subplots(
        n_models,
        2,
        figsize=(14, 4.2 * n_models),
        sharex="col",
        gridspec_kw={"width_ratios": [2.4, 1]},
    )

    if n_models == 1:
        axes = np.array([axes])

    observed_max = counts.max()

    # 🔹 Step 1: compute all max distributions first
    all_max_rep = []

    for model_name in model_names:
        fit = fits[model_name]
        y_pred = fit["pred_trace"].posterior_predictive["y_future"]
        y_obs_pred = y_pred[..., :T]

        y_rep = y_obs_pred.stack(sample=("chain", "draw")).values
        max_rep = y_rep.max(axis=0)
        all_max_rep.append(max_rep)

    # 🔹 Step 2: define shared bins
    global_min = min(m.min() for m in all_max_rep)
    global_max = max(m.max() for m in all_max_rep)

    bins = np.linspace(global_min, global_max, n_bins)

    # 🔹 Step 3: plotting loop
    for row, (model_name, max_rep) in enumerate(zip(model_names, all_max_rep)):
        fit = fits[model_name]
        y_pred = fit["pred_trace"].posterior_predictive["y_future"]
        y_obs_pred = y_pred[..., :T]

        mean_pred = y_obs_pred.mean(dim=("chain", "draw")).values
        hdi = az.hdi(y_obs_pred, hdi_prob=0.95)
        lower = hdi.sel(hdi="lower")["y_future"].values
        upper = hdi.sel(hdi="higher")["y_future"].values

        # ---- Left panel ----
        ax_ts = axes[row, 0]
        ax_ts.plot(time, counts, "ko", markersize=4, label="Observed")
        ax_ts.plot(time, mean_pred, color="blue", linewidth=2, label="Posterior mean")
        ax_ts.fill_between(time, lower, upper, color="blue", alpha=0.25, label="95% PPI")

        ax_ts.set_title(model_name)
        ax_ts.set_ylabel("Buffalo per bin")
        ax_ts.spines["top"].set_visible(False)
        ax_ts.spines["right"].set_visible(False)

        # ---- Right panel (fixed bins) ----
        ax_hist = axes[row, 1]
        ax_hist.hist(max_rep, bins=bins, alpha=0.75)

        ax_hist.axvline(
            observed_max,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Observed max",
        )

        bayes_p_max = np.mean(max_rep >= observed_max)

        ax_hist.set_title(f"Max count PPC\nBayesian p = {bayes_p_max:.3f}")
        ax_hist.set_xlabel("Maximum count")
        ax_hist.set_ylabel("Frequency")
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)

    axes[-1, 0].set_xlabel("Video time (s)")

    # Legends
    handles_ts, labels_ts = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles_ts, labels_ts, loc="upper right", frameon=True)

    handles_hist, labels_hist = axes[0, 1].get_legend_handles_labels()
    axes[0, 1].legend(handles_hist, labels_hist, loc="upper right", frameon=True)

    plt.tight_layout()

    if save:
        if filepath is None:
            raise ValueError("If save=True, you must provide a filepath.")
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

    plt.show()

    return fig, axes

def plot_ppc_multi_bin_shared(fits_list, counts_list, bin_sizes, model_name="NB-Gaussian", save = False, filepath = None, dpi= 300):
    import matplotlib.pyplot as plt
    import arviz as az
    import numpy as np

    n = len(fits_list)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.5 * n), sharex=True)

    # Ensure axes is iterable if n=1
    if n == 1:
        axes = [axes]

    for i, (fits, counts, bw) in enumerate(zip(fits_list, counts_list, bin_sizes)):
        fit = fits[model_name]

        y_pred = fit["pred_trace"].posterior_predictive["y_future"]

        # Extract posterior predictive summaries
        mean_pred = y_pred.mean(dim=("chain", "draw")).values
        hdi = az.hdi(y_pred, hdi_prob=0.95)
        lower = hdi.sel(hdi="lower")["y_future"].values
        upper = hdi.sel(hdi="higher")["y_future"].values

        T = len(counts)
        time = np.arange(T) * bw + 1

        ax = axes[i]

        # Plot observed
        ax.plot(time, counts, 'ko', label='Observed')

        # Plot predictive mean
        ax.plot(time, mean_pred[:T], color='blue', label='Posterior mean')

        # Plot 95% PPI
        ax.fill_between(time, lower[:T], upper[:T],
                        color='blue', alpha=0.3, label='95% PPI')

        # Styling
        ax.set_ylabel("Buffalo per bin")
        ax.set_title(f"{bw}s bins")
        ax.grid(False)

        # Remove top/right spines for clean look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Shared x-axis label
    axes[-1].set_xlabel("Video time (s)")

    # Single legend (top panel)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='upper right', frameon=True)

    plt.tight_layout()
    if save:
        if filepath is None:
            raise ValueError("If save=True, you must provide a filepath.")
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.show()

def plot_forecast_by_bin_size(
    fits_list,
    counts_list,
    bin_sizes,
    model_name="NB-Gaussian",
    speed_factor=3.5,
    thresholds=(0.2, 0.1, 0.05),
    save=False,
    filepath=None,
    dpi=300,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import arviz as az

    n = len(bin_sizes)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4.2 * n), sharex=True)

    if n == 1:
        axes = [axes]

    threshold_colors = ("red", "orange", "green")
    dash_patterns = [
        (0, (6, 3)),
        (0, (2, 2)),
        (0, (8, 2, 2, 2)),
    ]

    for ax, fits, counts, bw in zip(axes, fits_list, counts_list, bin_sizes):
        fit = fits[model_name]
        T = fit["T"]

        y_pred = fit["pred_trace"].posterior_predictive["y_future"]
        mean_pred = y_pred.mean(dim=("chain", "draw")).values

        hdi = az.hdi(y_pred, hdi_prob=0.95)
        lower = hdi.sel(hdi="lower")["y_future"].values
        upper = hdi.sel(hdi="higher")["y_future"].values

        time = np.arange(len(counts)) * bw + 1
        time_extended = fit["time_extended"]

        ax.plot(time, counts, "ko", markersize=4, label="Observed")
        ax.plot(time_extended, mean_pred, color="blue", label="Posterior mean")
        ax.fill_between(time_extended, lower, upper, color="blue", alpha=0.25, label="95% PPI")
        ax.axvline(time[-1], color="gray", linestyle="--", label="End of video")

        cutoffs = get_cutoff_times_with_crossing_from_fit(
            fit=fit,
            bin_width=bw,
            speed_factor=speed_factor,
            thresholds=thresholds,
        )

        # Group actual crossed thresholds by cutoff time
        crossed_items = [
            (thr, vals["cutoff_time"])
            for thr, vals in cutoffs.items()
            if vals["threshold_crossed"]
        ]

        time_groups = {}
        for thr, cutoff_time in crossed_items:
            time_groups.setdefault(cutoff_time, []).append(thr)

        for cutoff_time, grouped_thresholds in time_groups.items():
            n_overlap = len(grouped_thresholds)

            # Small offsets only if lines overlap exactly
            offsets = np.linspace(-0.08 * bw, 0.08 * bw, n_overlap)

            for offset, thr in zip(offsets, grouped_thresholds):
                idx = list(thresholds).index(thr)

                ax.axvline(
                    cutoff_time + offset,
                    color=threshold_colors[idx],
                    linestyle=dash_patterns[idx],
                    linewidth=2,
                    label=f"{thr} buffalo/s cutoff",
                )

        ax.set_title(f"{model_name}, {bw}s bins")
        ax.set_ylabel("Buffalo per bin")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Video time (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    
    axes[0].legend(
    unique.values(),
    unique.keys(),
    loc="upper left",             # anchor corner of legend
    bbox_to_anchor=(0.55, 0.95),  # 🔥 precise position (x, y in axes coords)
    frameon=True,
    framealpha=1,
    facecolor="white",
    edgecolor="black"
)

    plt.tight_layout()

    if save:
        if filepath is None:
            raise ValueError("Provide filepath when save=True.")
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

    plt.show()

    return fig, axes

def get_cutoff_times_with_crossing_from_fit(
    fit,
    bin_width,
    speed_factor,
    thresholds,
):
    import numpy as np
    import arviz as az

    T = fit["T"]
    time_extended = fit["time_extended"]
    future_start = T

    y_pred = fit["pred_trace"].posterior_predictive["y_future"]

    upper_ci = (
        az.hdi(y_pred, hdi_prob=0.95)
        .sel(hdi="higher")["y_future"]
        .values
    )

    upper_future = upper_ci[future_start:]

    cutoffs = {}

    for threshold in thresholds:
        threshold_per_bin = threshold * (bin_width * speed_factor)

        crossed = upper_future < threshold_per_bin

        if crossed.any():
            cutoff_bin = int(np.argmax(crossed))
            cutoff_time = time_extended[future_start + cutoff_bin]
            threshold_crossed = True
        else:
            cutoff_bin = None
            cutoff_time = None
            threshold_crossed = False

        cutoffs[threshold] = {
            "cutoff_bin": cutoff_bin,
            "cutoff_time": cutoff_time,
            "threshold_crossed": threshold_crossed,
            "threshold_per_bin": threshold_per_bin,
        }

    return cutoffs