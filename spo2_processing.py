from pobm.obm.complex import ComplexityMeasures
from pobm.obm.general import OverallGeneralMeasures
from pobm.obm.periodicity import PRSAMeasures, PSDMeasures
from pobm.prep import resamp_spo2, set_range, median_spo2
from dataclasses import asdict
import pandas as pd

from utils_func import linear_interpolation, extract_relative_threshold, extract_hard_desat_features, \
    extract_hard_personalized_desat, extract_all_new_features, compute_psd_features


def preprocess_spo2(spo2_signal, original_freq):
    """
    Preprocess spo2 signal, including resampling to 1 Hz, remove non-physiological value, smoothing of the signal and
    then filling removed values with linear interpolation

    :param spo2_signal: Numpy array of shape (len, 1)
    :param original_freq: Int, original frequency of the signal
    :return: Preprocessed signal
    """

    if original_freq != 1:
        spo2_signal = resamp_spo2(spo2_signal, original_freq)

    spo2_signal = set_range(spo2_signal)
    spo2_signal = median_spo2(spo2_signal)
    spo2_signal = linear_interpolation(spo2_signal)

    return spo2_signal


def extract_biomarkers_per_signal(signal, patient, max_length_desat=500, compute_complexity=False):
    metadata = {
        "patient": patient,
    }

    if compute_complexity is True:
        complexity_class = ComplexityMeasures()
        results_complexity = asdict(complexity_class.compute(signal))
        results_complexity["LZ_4"] = complexity_class.comp_lz(signal, dual_quantization=False)
    else:
        results_complexity = {}

    results_hypoxic_relative_3, results_desat_relative_3 = extract_relative_threshold(signal, max_length_desat, ODI=3)
    results_hypoxic_relative_5, results_desat_relative_5 = extract_relative_threshold(signal, max_length_desat, ODI=5)

    results_desat_hard_93, results_hypoxic_hard_93 = extract_hard_desat_features(signal, hard_threshold=93)
    results_desat_hard_90, results_hypoxic_hard_90 = extract_hard_desat_features(signal, hard_threshold=90)
    results_desat_hard_88, results_hypoxic_hard_88 = extract_hard_desat_features(signal, hard_threshold=88)
    results_desat_hard_85, results_hypoxic_hard_85 = extract_hard_desat_features(signal, hard_threshold=85)

    results_desat_personalized_05, results_hypoxic_personalized_05 = extract_hard_personalized_desat(signal,
                                                                                                     quantile=0.05)
    results_desat_personalized_10, results_hypoxic_personalized_10 = extract_hard_personalized_desat(signal,
                                                                                                     quantile=0.10)
    results_desat_personalized_20, results_hypoxic_personalized_20 = extract_hard_personalized_desat(signal,
                                                                                                     quantile=0.20)

    statistics_class = OverallGeneralMeasures()
    results_overall = asdict(statistics_class.compute(signal))

    prsa_class = PRSAMeasures()
    results_PRSA = asdict(prsa_class.compute(signal))

    prsa_class_long = PRSAMeasures(PRSA_Window=20)
    results_PRSA_long = asdict(prsa_class_long.compute(signal))
    results_PRSA_long = dict((key + "_long", value) for (key, value) in results_PRSA_long.items())

    psd_class = PSDMeasures()
    results_PSD = asdict(psd_class.compute(signal))

    dict_new_features = extract_all_new_features(signal)
    new_psd_features = compute_psd_features(signal)

    all_features = {**metadata, **results_hypoxic_relative_3, **results_desat_relative_3, **results_hypoxic_relative_5,
                    **results_desat_relative_5, **results_desat_hard_90, **results_hypoxic_hard_90,
                    **results_desat_hard_88, **results_hypoxic_hard_88, **results_desat_hard_85,
                    **results_hypoxic_hard_85, **results_desat_hard_93, **results_hypoxic_hard_93,
                    **results_overall, **results_PRSA, **results_PRSA_long, **results_PSD,
                    **results_complexity,
                    **dict_new_features, **new_psd_features,

                    **results_desat_personalized_05, **results_hypoxic_personalized_05,
                    **results_desat_personalized_10, **results_hypoxic_personalized_10,
                    **results_desat_personalized_20, **results_hypoxic_personalized_20,
                    }
    biomarkers = pd.DataFrame(all_features, index=[0])

    return biomarkers
