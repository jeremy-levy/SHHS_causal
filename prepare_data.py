import os

import numpy as np
import pandas as pd
import pyedflib
from pobm.prep import set_range, median_spo2
from tqdm import tqdm
from joblib import Parallel, delayed

from spo2_processing import extract_biomarkers_per_signal
from utils_func import linear_interpolation, load_sleep_stages
from xml_processing import extract_events_xml


def read_csv_metadata(database_path):
    metadata = pd.read_csv(os.path.join(database_path, 'shhs1-harmonized-dataset-0.18.0.csv'))

    metadata['smoking_status'] = metadata['nsrr_current_smoker'] + metadata['nsrr_ever_smoker']

    metadata['smoking_status'].replace({'nono': 'No smoker'}, inplace=True)
    metadata['smoking_status'].replace({'noyes': 'Ex smoker'}, inplace=True)
    metadata['smoking_status'].replace({'yesno': 'Bug'}, inplace=True)
    metadata['smoking_status'].replace({'yesyes': 'Smoker'}, inplace=True)
    metadata['smoking_status'].replace({'not reportednot reported': 'Smoker'}, inplace=True)

    return metadata


def prepare_csv_one_patient(arg):
    database_path, metadata, patient, window_length = arg

    biomarker_patient = pd.DataFrame()
    metadata_patient = metadata.loc[metadata['nsrrid'] == int(patient[6:])]

    try:
        edf = pyedflib.EdfReader(os.path.join(database_path, 'edf', patient + '.edf'))
        i_position = np.where(np.array(edf.getSignalLabels()) == 'SaO2')[0][0]
        position = edf.readSignal(i_position)
        spo2_signal = np.array(position).astype(float)
        edf.close()
    except OSError:
        return

    # Preprocess SpO2 signal. Does not change the shape of the signal
    spo2_signal = set_range(spo2_signal)
    spo2_signal = median_spo2(spo2_signal)
    spo2_signal = linear_interpolation(spo2_signal)

    events_name, events_start, events_duration = extract_events_xml(os.path.join(database_path, 'xml',
                                                                                 patient + '-nsrr.xml'))

    length_epoch = 30
    try:
        sleep_stages = load_sleep_stages(os.path.join(database_path, 'profusion', patient + '-profusion.xml'))
    except FileNotFoundError:
        return
    sleep_stages = np.repeat(sleep_stages, length_epoch)

    previous_events_end = 0
    for event_name, event_start, event_duration in zip(events_name, events_start, events_duration):
        event_start = int(float(event_start))
        event_duration = int(float(event_duration))

        if event_start - previous_events_end >= 2 * window_length:
            window_begin = previous_events_end + window_length / 2
            window_end = window_begin + window_length

            spo2_window_no_event = spo2_signal[int(float(window_begin)): int(float(window_end))]
            sleep_stage_window = [1]

            biomarkers_window = save_window(spo2_window_no_event, window_length, sleep_stage_window, -1,  -1, patient,
                                            '', metadata_patient, outcome=0)
            biomarker_patient = pd.concat([biomarker_patient, biomarkers_window])

        spo2_window = spo2_signal[int(float(event_start)) - window_length: int(float(event_start))]
        sleep_stage_window = sleep_stages[
                             int(float(event_start)) - 1: int(float(event_start)) + int(float(event_duration)) + 1]
        biomarkers_window = save_window(spo2_window, window_length, sleep_stage_window, event_start, event_duration,
                                        patient, event_name, metadata_patient, outcome=1)
        biomarker_patient = pd.concat([biomarker_patient, biomarkers_window])

        previous_events_end = event_start + event_duration
    biomarker_patient.to_csv(os.path.join('data_csv', patient + '.csv'), index=False)


def save_window(spo2_window, window_length, sleep_stage_window, event_start, event_duration, patient, event_name,
                metadata_patient, outcome):
    # Filtering events with no Spo2 available
    if len(spo2_window) < window_length:
        return pd.DataFrame()

    # Filtering events occurring while patient is awake
    if 0 in sleep_stage_window:
        return pd.DataFrame()

    biomarkers_window = extract_biomarkers_per_signal(spo2_window, patient=patient)

    biomarkers_window['event_name'] = event_name
    biomarkers_window['event_start'] = event_start
    biomarkers_window['event_duration'] = event_duration

    biomarkers_window['outcome'] = outcome

    biomarkers_window['age'] = metadata_patient['nsrr_age']
    biomarkers_window['sex'] = metadata_patient['nsrr_sex']
    biomarkers_window['race'] = metadata_patient['nsrr_race']
    biomarkers_window['ethnicity'] = metadata_patient['nsrr_ethnicity']
    biomarkers_window['bmi'] = metadata_patient['nsrr_bmi']
    biomarkers_window['smoking_status'] = metadata_patient['smoking_status']

    return biomarkers_window


def union_csv():
    data_df = pd.DataFrame()
    for csv_file in tqdm(os.listdir('data_csv')):
        df_current = pd.read_csv(os.path.join('data_csv', csv_file))
        data_df = pd.concat([data_df, df_current])

    print('data_df', data_df.shape)
    data_df.to_csv('data_SHHS1.csv', index=False)


def main(database_path):
    window_length = 300

    metadata = read_csv_metadata(database_path)

    list_patients = [file_name[0:-4] for file_name in os.listdir(os.path.join(database_path, 'edf'))]
    args = [(database_path, metadata, patient, window_length) for patient in list_patients]
    Parallel(n_jobs=-1)(delayed(prepare_csv_one_patient)(arg) for arg in tqdm(args))


if __name__ == "__main__":
    os.makedirs('data_csv', exist_ok=True)

    main(database_path='/MLAIM/AIMLab/Jeremy/SHHS1/')
    # main(database_path='/MLAIM/AIMLab/Jeremy/shhs2/')

    union_csv()
