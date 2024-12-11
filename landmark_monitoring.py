import numpy as np
import csv
import mediapipe as mp
import cv2
import os
import numpy as np
import csv
from utils.landmark_extractor import capture_landmark_monitor, image_landmark_monitor
import argparse

parser = argparse.ArgumentParser(description="Collect media data and save landmarks.")
parser.add_argument('--data_path', type=str, required=False, help="Path to the data source.")
parser.add_argument('--source', type=str, choices=['camera', 'video', 'image'], required=True, help="Media source.")
parser.add_argument('--model_path',type=str,required=True,help="Path to the trained model.")

args = parser.parse_args()
data_path = args.data_path
source = args.source.lower()
model_path=args.model_path
import pickle
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f"!! Monitoring with {source} !! ")
if source in ['camera', 'video']:
    capture_landmark_monitor(data_path,model,source)
elif source == 'image':
     image_landmark_monitor(data_path,model)
else:
    raise NameError("Invalid source. Please choose 'video', 'camera', or 'image'.") 

parser = argparse.ArgumentParser(description="Collect media data and save landmarks.")
parser.add_argument('--data_path', type=str, required=False, help="Path to the data source.")
parser.add_argument('--source', type=str, choices=['camera', 'video', 'image'], required=True, help="Media source.")
parser.add_argument('--model_path',type=str,required=True,help="Path to the trained model.")

args = parser.parse_args()
data_path = args.data_path
source = args.source.lower()
model_path=args.model_path
import pickle
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f"!! Monitoring with {source} !! ")
if source in ['camera', 'video']:
   capture_landmark_monitor(data_path,model,source)
elif source == 'image':
     image_landmark_monitor(data_path,model)
else:
    raise NameError("Invalid source. Please choose 'video', 'camera', or 'image'.")
