import math
import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt

def get_bin_counts(tracks, bin_length):
    """
    Count the number of buffalo crossing the count line in each time bin.

    Parameters:
        - bin_length: the duration of each bin in seconds
    """
    frames_per_bin = bin_length*30 # calculate the number of frames per bin
    n_bins = math.ceil(368/bin_length) # calculate the number of bins in the video
    len_last_bin = 368%bin_length

    counts = [] # initiate empty list to store the counts for each bin
    tracks_counted = [] # initiate an empty list to track which tracks are accounted for. If this doesn't equal 2972, something has gone wrong

    for seg in np.arange(n_bins): # generate the count for each segment/bin
        count = 0 # start count at 0
        start_frame = frames_per_bin * seg # set the first frame of the bin
        end_frame = frames_per_bin * (seg+1)-1 # set the last frame of the bin
        if seg == (n_bins-1):
            end_frame = 11040 # there are only 11040 frames, so if the last bin isn't the full bin length, it needs the end frame set manually
        for n,t in enumerate(tracks): # now go through each track and check when it crosses the count line. If it crosses within this bin, increase count by 1
            track = t['track']
            yvals = track[:,0]
            if (yvals > 100).any:
                cross_i = np.min(np.where(yvals >100))
                cross_frame = cross_i + t['first_frame'] # this is the frame at which the track crosses the count line
                if (cross_frame <= end_frame) & (cross_frame >= start_frame):
                    count += 1
                    tracks_counted.append(n)
        if seg == (n_bins-1): # the last bin may be shorter than the others, so we adjust the count in this bin, assuming that animals cross the line at a constant rate during the bin
            if len_last_bin >0:
                count = int((bin_length*count)/len_last_bin) 
        counts.append(count)

    n_tracks_counted = len(tracks_counted)
    if n_tracks_counted != 2972:
        print('Track count is ' + str(n_tracks_counted) + ' , but should be 2972.')

    return(counts)


def get_manual_bin_counts(manual_counts, bin_length):
    counts = []
    for n,c in enumerate(manual_counts):
        if n <18:
            n_newbins = int(20/bin_length)
            subcount = c/n_newbins
            counts.extend([subcount]*n_newbins)
        if n == 18:
            n_newbins = 8/bin_length
            n_fullbins = math.floor(8/bin_length)
            last_binlength = 8%bin_length
            subcount = c/n_newbins
            if last_binlength == 0:
                counts.extend([subcount]*int(n_newbins))
            else:
                count_fullbins = c/n_newbins
                last_bin_prop = n_newbins - n_fullbins
                count_lastbin = (c*last_bin_prop)/n_newbins
                counts.extend([count_fullbins]*n_fullbins)
                counts.extend([count_lastbin])
    return(counts) 

def fit_and_forecast(counts, bin_width, speed_factor, real_time_threshold, forecast_seconds=500):

    T = len(counts)
    counts = np.array(counts).astype(int)
    time = np.arange(T) * bin_width + 1
    forecast_bins = int(forecast_seconds / bin_width)
    time_extended = np.arange(T + forecast_bins) * bin_width + 1

    bin_duration_real_time = bin_width * speed_factor
    threshold_per_bin = real_time_threshold * bin_duration_real_time

    with pm.Model() as model:
        A = pm.HalfNormal("A", sigma=200)
        mu = pm.Uniform("mu", lower=0, upper=time_extended[-1])
        sigma = pm.HalfNormal("sigma", sigma=100)

        lambda_ = A * pm.math.exp(-0.5 * ((time - mu) / sigma)**2)
        y_obs = pm.Poisson("y_obs", mu=lambda_ * bin_width, observed=counts)

        trace = pm.sample(1000, tune=1000, target_accept=0.95, chains=4, random_seed=27, progressbar=False)

        lambda_future = pm.Deterministic("lambda_future", A * pm.math.exp(
            -0.5 * ((time_extended - mu) / sigma)**2))
        y_future = pm.Poisson("y_future", mu=lambda_future * bin_width, shape=len(time_extended))

        pred_trace = pm.sample_posterior_predictive(trace, var_names=["lambda_future", "y_future"],random_seed=27, progressbar=False)

    # Get predictive values
    future_start = T
    y_pred = pred_trace.posterior_predictive["y_future"]
    mean_forecast = y_pred.mean(dim=("chain", "draw")).values
    hdi = az.hdi(y_pred, hdi_prob=0.95)
    lower_ci = hdi.sel(hdi="lower")["y_future"].values
    upper_ci = hdi.sel(hdi="higher")["y_future"].values

    upper_future = upper_ci[future_start:]
    cutoff_bin = np.argmax(upper_future < threshold_per_bin)
    if cutoff_bin == 0 and upper_future[0] >= threshold_per_bin:
        cutoff_bin = len(upper_future)

    # Compute missed buffalo estimate
    missed = {
        "Lower bound (95% CI)": lower_ci[future_start:future_start + cutoff_bin].sum(),
        "Posterior mean": mean_forecast[future_start:future_start + cutoff_bin].sum(),
        "Upper bound (95% CI)": upper_ci[future_start:future_start + cutoff_bin].sum(),
    }

    # Return everything needed for plotting and tabulation
    return {
        "pred_trace": pred_trace,
        "mean": mean_forecast,
        "lower": lower_ci,
        "upper": upper_ci,
        "cutoff_bin": cutoff_bin,
        "cutoff_time": (future_start + cutoff_bin) * bin_width,
        "missed_counts": missed
    }

def plot_forecast_panels(pred_trace, counts, bin_width, speed_factor, real_time_threshold, label=""):
    T = len(counts)
    forecast_seconds = 500
    forecast_bins = int(forecast_seconds / bin_width)
    time = np.arange(T) * bin_width + 1
    time_extended = np.arange(T + forecast_bins) * bin_width + 1

    future_start = T
    y_pred = pred_trace.posterior_predictive["y_future"]

    mean_forecast = y_pred.mean(dim=("chain", "draw")).values
    hdi = az.hdi(y_pred, hdi_prob=0.95)

    lower_ci = hdi.sel(hdi="lower")["y_future"].values
    upper_ci = hdi.sel(hdi="higher")["y_future"].values

    threshold = real_time_threshold * (bin_width * speed_factor)
    cutoff_bin = np.argmax(upper_ci[future_start:] < threshold)
    if cutoff_bin == 0 and upper_ci[future_start] >= threshold:
        cutoff_bin = forecast_bins
    cutoff_time = time_extended[future_start + cutoff_bin]

    # Plot panel
    plt.plot(time, counts, 'ko', label='Observed counts')
    plt.plot(time_extended, mean_forecast, color='blue', label='Forecast mean')
    plt.fill_between(time_extended, lower_ci, upper_ci, color='blue', alpha=0.3, label='95% CI')
    plt.axvline(x=time[-1], color='gray', linestyle='--', label='End of video')
    plt.axvline(x=cutoff_time, color='red', linestyle=':', label='Forecast cutoff')
    plt.title(f"{label} (Bin: {bin_width}s)")
    plt.xlabel("Time (video seconds)")
    plt.ylabel("Buffalo per bin")
    plt.xlim(-5,870)
    plt.legend()


def plot_combined_panels(traces, bin_sizes, count_lists, real_time_threshold, speed_factor=3.5):
    plt.figure(figsize=(16, 5 * len(bin_sizes)))
    for i, (trace, bw, counts) in enumerate(zip(traces, bin_sizes, count_lists), start=1):
        plt.subplot(len(bin_sizes), 1, i)
        plot_forecast_panels(trace, counts, bin_width=bw, speed_factor=speed_factor,
                             real_time_threshold=real_time_threshold,
                             label=f"Forecast for Bin Width = {bw}s")
    plt.tight_layout()
    plt.show()

def get_cutoff_times(pred_trace, bin_width, speed_factor, thresholds, forecast_seconds, T):

    forecast_bins = int(forecast_seconds / bin_width)
    time_extended = np.arange(T + forecast_bins) * bin_width + 1
    future_start = T
    upper_ci = az.hdi(pred_trace.posterior_predictive["y_future"], hdi_prob=0.95).sel(hdi="higher")["y_future"].values

    cutoffs = {}
    for rt in thresholds:
        threshold_per_bin = rt * (bin_width * speed_factor)
        upper_future = upper_ci[future_start:]
        cutoff_bin = np.argmax(upper_future < threshold_per_bin)
        if cutoff_bin == 0 and upper_future[0] >= threshold_per_bin:
            cutoff_bin = forecast_bins
        cutoffs[rt] = time_extended[future_start + cutoff_bin]
    return cutoffs, time_extended

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
    plt.legend(loc = 'upper right')
    plt.grid(False)
    plt.tight_layout()
    plt.xlim(-5,870)
    #plt.savefig('file_path')
    plt.show()

def plot_multi_bin_forecasts_shared_clean(count_lists, pred_traces, bin_sizes, speed_factor, thresholds, forecast_seconds=500):
    import matplotlib.pyplot as plt
    import arviz as az
    import numpy as np

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

        # Remove right and top spines for clean look
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # X label only on bottom
    axes[-1].set_xlabel("Video Time (s)")

    # One legend in top-right of first panel
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='upper right', frameon=True)

    plt.tight_layout()
    plt.xlim(-5,870)
    #plt.savefig('file_path')
    plt.show()
