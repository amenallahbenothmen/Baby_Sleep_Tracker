import mediapipe as mp
import cv2
import os
import numpy as np
import csv
from utils.landmark_extractor import image_landmark_extractor, capture_landmark_monitor
import argparse

parser = argparse.ArgumentParser(description="Collect media data and save landmarks.")
parser.add_argument('--data_path', type=str, required=False, help="Path to the data source.")
parser.add_argument('--state', type=str, required=True, choices=['Asleep', 'Awake'], help="State to be used in CSV (e.g., 'Asleep', 'Awake').")
parser.add_argument('--output_csv', type=str, required=True, help="Path to save the CSV file.")
parser.add_argument('--source', type=str, choices=['camera', 'video', 'image'], required=True, help="Media source (e.g., 'camera', 'video', 'image').")

args = parser.parse_args()
data_path = args.data_path
state = args.state.capitalize()
output_csv = args.output_csv
source = args.source.lower()

if not os.path.exists(output_csv):
    coord_len = 501
    landmarks = ['state'] + [f'{axis}{i}' for i in range(1, coord_len + 1) for axis in ['x', 'y', 'z', 'v']]
    with open(output_csv, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)
    print(f"CSV header written to {output_csv}.")
else:
    print(f"!! {output_csv} already exists. Header will not be overwritten. !!")

print(f"!! Collecting from '{data_path}' for state '{state}' outputted to '{output_csv}' !! ")

if source in ['camera', 'video']:
    capture_landmark_monitor(data_path=data_path, output_csv=output_csv, source=source, state=state)
elif source == 'image':
    image_landmark_extractor(data_path=data_path, output_csv=output_csv, state=state)
else:
    raise ValueError("Invalid source. Please choose 'video', 'camera', or 'image'.")
