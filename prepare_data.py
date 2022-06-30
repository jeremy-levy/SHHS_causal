import os

import numpy as np
import pandas as pd
import pyedflib
from pobm.prep import set_range, median_spo2
from tqdm import tqdm
from joblib import Parallel, delayed

from spo2_processing import extract_biomarkers_per_signal
from utils_func import linear_interpolation
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


def prepare_csv_one_patient(database_path, metadata, patient, window_length=300):
    biomarker_patient = pd.DataFrame()
    metadata_patient = metadata.loc[metadata['nsrrid'] == int(patient[6:])]
    edf = pyedflib.EdfReader(os.path.join(database_path, 'edf', patient + '.edf'))

    i_position = np.where(np.array(edf.getSignalLabels()) == 'SaO2')[0][0]
    position = edf.readSignal(i_position)
    spo2_signal = np.array(position).astype(float)
    edf.close()

    # Preprocess SpO2 signal. Does not change the shape of the signal
    spo2_signal = set_range(spo2_signal)
    spo2_signal = median_spo2(spo2_signal)
    spo2_signal = linear_interpolation(spo2_signal)

    events_name, events_start, events_duration = extract_events_xml(os.path.join(database_path, 'xml',
                                                                                 patient + '-nsrr.xml'))

    for event_name, event_start, event_duration in zip(events_name, events_start, events_duration):
        spo2_window = spo2_signal[int(float(event_start)) - window_length: int(float(event_start))]

        if len(spo2_window) < window_length:
            continue

        biomarkers_window = extract_biomarkers_per_signal(spo2_window, patient=patient)

        biomarkers_window['event_name'] = event_name
        biomarkers_window['event_start'] = event_start
        biomarkers_window['event_duration'] = event_duration

        biomarkers_window['outcome'] = 1

        biomarkers_window['age'] = metadata_patient['nsrr_age']
        biomarkers_window['sex'] = metadata_patient['nsrr_sex']
        biomarkers_window['race'] = metadata_patient['nsrr_race']
        biomarkers_window['ethnicity'] = metadata_patient['nsrr_ethnicity']
        biomarkers_window['bmi'] = metadata_patient['nsrr_bmi']
        biomarkers_window['smoking_status'] = metadata_patient['smoking_status']

        biomarker_patient = pd.concat([biomarker_patient, biomarkers_window])
    biomarker_patient.to_csv(os.path.join('data_csv', patient + '.csv'), index=False)


def main(database_path):
    # database_path = 'C:\\Users\\jeremy\\Documents\\ServerHTTP\\SHHS_causal\\data'
    metadata = read_csv_metadata(database_path)

    list_patients = [file_name[0:-4] for file_name in os.listdir(os.path.join(database_path, 'edf'))]
    for patient in tqdm(list_patients):
        prepare_csv_one_patient(database_path, metadata, patient)

    args = [(database_path, metadata, patient) for patient in list_patients]
    Parallel(n_jobs=-1)(delayed(prepare_csv_one_patient)(arg) for arg in args)


if __name__ == "__main__":
    main(database_path='/MLAIM/AIMLab/Jeremy/SHHS1/')
    main(database_path='/MLAIM/AIMLab/Jeremy/shhs2/')
