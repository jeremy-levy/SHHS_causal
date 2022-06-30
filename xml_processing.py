import os
import xml.dom.minidom as xdm

import numpy as np
from tqdm import tqdm

ahi_events_name_shhs = ['Obstructive Apnea', 'Hypopnea', 'Obstructive Hypopnea', 'Central Apnea', 'Mixed Apnea',
                        'Obstructive apnea|Obstructive Apnea', 'Hypopnea|Hypopnea', 'Central apnea|Central Apnea',
                        'Mixed Apnea|Mixed Apnea', 'Obstructive Hypopnea|Obstructive Hypopnea']


# TODO: Check that number of event correspond to AHI in csv-guidelines
def extract_events_xml(xml_path):
    """
    Receives as parameter path to specific xml file.
    Returns 3 np.array: name of events, start time, and duration.
    3 arrays have same shape.

    """

    doc = xdm.parse(open(xml_path, 'r'))
    events = doc.getElementsByTagName("ScoredEvents")[0]

    event_duration = events.getElementsByTagName("Duration")
    event_duration = np.array([int(float(event_duration[i].firstChild.data)) for i in range(len(event_duration))])

    event_start_time = events.getElementsByTagName("Start")
    event_start_time = np.array([event_start_time[i].firstChild.data for i in np.arange(len(event_start_time))])

    event_name = events.getElementsByTagName("EventConcept")
    event_name = np.array([event_name[i].firstChild.data for i in range(len(event_name))])
    ahi_events = [event in ahi_events_name_shhs for event in event_name]

    event_duration = event_duration[ahi_events]
    event_start_time = event_start_time[ahi_events]
    event_name = event_name[ahi_events]

    assert event_name.shape == event_start_time.shape
    assert event_duration.shape == event_start_time.shape

    return event_name, event_start_time, event_duration


def main():
    xml_directory_path = os.path.join('data', 'xml')

    for xml_file in tqdm(os.listdir(xml_directory_path)):
        event_name, event_start_time, event_duration = extract_events_xml(xml_path=os.path.join('data\\xml', xml_file))
        print(len(event_duration), len(event_start_time), len(event_name))

