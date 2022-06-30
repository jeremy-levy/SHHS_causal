import numpy as np
from pobm.obm.burden import HypoxicBurdenMeasures
from pobm.obm.complex import ComplexityMeasures
from pobm.obm.desat import DesaturationsMeasures
from pobm.obm.general import OverallGeneralMeasures
from pobm.obm.periodicity import PRSAMeasures, PSDMeasures
from pobm.prep import resamp_spo2, set_range, median_spo2
from dataclasses import asdict
from scipy.stats import entropy, kurtosis, skew, median_absolute_deviation, moment
import pandas as pd
from scipy.signal import welch
from numpy.random import uniform


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def linear_interpolation(signal):
    signal = np.array(signal)

    nans, x = nan_helper(signal)
    signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])

    return signal


def extract_relative_threshold(signal, max_length_desat, ODI):
    desat_class_relative = DesaturationsMeasures(desat_max_length=max_length_desat, ODI_Threshold=ODI,
                                                 desat_min_length=0, min_dist_meta_event=0)
    results_desat_relative = desat_class_relative.compute(signal)

    hypoxic_class_relative = HypoxicBurdenMeasures(results_desat_relative.begin, results_desat_relative.end)
    results_hypoxic_relative = asdict(hypoxic_class_relative.compute(signal))
    results_hypoxic_relative = dict(
        (key + "_relative_" + str(ODI), value) for (key, value) in results_hypoxic_relative.items())

    results_desat_relative = asdict(results_desat_relative)
    results_desat_relative.pop('begin')
    results_desat_relative.pop('end')
    results_desat_relative = dict(
        (key + "_relative_" + str(ODI), value) for (key, value) in results_desat_relative.items())

    return results_hypoxic_relative, results_desat_relative


def extract_hard_desat_features(signal, hard_threshold):
    desat_class_hard = DesaturationsMeasures(relative=False, hard_threshold=hard_threshold, min_dist_meta_event=10,
                                             desat_min_length=20)
    results_desat_hard = desat_class_hard.compute(signal)

    hypoxic_class_hard = HypoxicBurdenMeasures(results_desat_hard.begin, results_desat_hard.end)
    results_hypoxic_hard = asdict(hypoxic_class_hard.compute(signal))
    results_hypoxic_hard = dict(
        (key + "_hard_" + str(hard_threshold), value) for (key, value) in results_hypoxic_hard.items())

    results_desat_hard = asdict(results_desat_hard)
    results_desat_hard.pop('begin')
    results_desat_hard.pop('end')
    results_desat_hard = dict(
        (key + "_hard_" + str(hard_threshold), value) for (key, value) in results_desat_hard.items())

    return results_hypoxic_hard, results_desat_hard


def extract_hard_personalized_desat(signal, quantile):
    desat_class_hard = DesaturationsMeasures(relative=False, hard_threshold=np.quantile(signal, quantile),
                                             min_dist_meta_event=10, desat_min_length=20)
    results_desat_hard = desat_class_hard.compute(signal)

    hypoxic_class_hard = HypoxicBurdenMeasures(results_desat_hard.begin, results_desat_hard.end)
    results_hypoxic_hard = asdict(hypoxic_class_hard.compute(signal))
    results_hypoxic_hard = dict(
        (key + "_personalized_" + str(quantile), value) for (key, value) in results_hypoxic_hard.items())

    results_desat_hard = asdict(results_desat_hard)
    results_desat_hard.pop('begin')
    results_desat_hard.pop('end')
    results_desat_hard = dict(
        (key + "_personalized_" + str(quantile), value) for (key, value) in results_desat_hard.items())

    return results_hypoxic_hard, results_desat_hard


def extract_all_new_features(signal):
    signal = signal[~np.isnan(signal)]
    kurtosis_sig, skew_sig, mad = kurtosis(signal), skew(signal), median_absolute_deviation(signal)

    pd_series = pd.Series(signal)
    counts = pd_series.value_counts()
    entropy_sig = entropy(counts)

    dict_features = {
        'entropy': entropy_sig,
        'kurtosis': kurtosis_sig,
        'skew': skew_sig,
        'MAD': mad
    }
    return dict_features


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (entropy(p, m) + entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def compute_features_from_psd(psd_signal, to_add=''):
    second_moment = moment(psd_signal, moment=2)
    third_moment = moment(psd_signal, moment=3)
    fourth_moment = moment(psd_signal, moment=4)

    median_frequency = np.median(psd_signal)
    spectral_entropy = entropy(psd_signal)

    uniform_dist = uniform(low=np.min(psd_signal), high=np.max(psd_signal), size=len(psd_signal))
    jensen_distance = jensen_shannon_distance(psd_signal, uniform_dist)

    final_features = {
        'second_moment' + to_add: second_moment,
        'third_moment' + to_add: third_moment,
        'fourth_moment' + to_add: fourth_moment,
        'median_frequency' + to_add: median_frequency,
        'spectral_entropy' + to_add: spectral_entropy,
        'jensen_distance' + to_add: jensen_distance
    }

    return final_features


def compute_psd_features(signal):
    freq, psd_transform = welch(signal, fs=1, window="hamming")
    psd_transform = psd_transform / len(signal)

    final_features_all_psd = compute_features_from_psd(psd_transform)

    lower_f, higher_f = 0.014, 0.033
    amplitude_bp = psd_transform[lower_f < freq]
    freq_bp = freq[lower_f < freq]

    amplitude_bp = amplitude_bp[freq_bp < higher_f]
    final_features_bp = compute_features_from_psd(amplitude_bp, to_add='_bp')

    all_features = {**final_features_all_psd, **final_features_bp}
    return all_features
