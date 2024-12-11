
# **Baby_Sleep_Tracker Video, Webcam, Images, Mediapipe, Keras, and OpenCV**

## **Overview**
The **Baby_Sleep_Tracker** is a framework designed to classify states like "Awake" or "Asleep" based on body pose data collected from videos, webcam feeds, or images. It leverages **MediaPipe**, **OpenCV**, and **Keras** to extract and analyze pose landmarks, train models, and perform real-time monitoring.  

### **Current Application**  
This project is currently specifically used for **baby awakeness monitoring**, offering insights into whether a baby is awake or asleep through non-invasive observation.  

---

## **Features**
1. **Real-Time State Monitoring:**
   - Analyze and classify states  inputs from **webcam feeds**, **video files**, or **static images**.
   - Outputs predictions based on pre-trained models.

2. **Comprehensive Data Collection:**
   - Captures pose landmarks from various media sources and organizes them into structured CSV files.
   - Seamlessly integrates with live cameras, video files, and image datasets for flexible data collection.

3. **Robust Preprocessing Pipeline:**
   - Implements advanced preprocessing steps, including outlier removal, data balancing, and feature standardization.
   - Adapts to diverse datasets, ensuring optimized data preparation for model training..

4. **Lightweight Deep Learning Model:**
   - Builds efficient binary classifiers using **Keras**.
   - Evaluates performance with metrics such as **accuracy** and **F1-score** **accuracy** for classification performance.

---


## **Directory Structure**
```
BABY_SLEEP_TRACKER/
├── utils/
│   ├── data_preprocessing.py           
│   ├── landmark_extractor.py       
│   ├── landmark_monitor.py           
├── landmark_extractor.py                  
├── landmark_monitoring.py      
├── train_model.py                  
└── README.md                   
```
---

## **Dependencies**
This project is built on **Python 3.8.18** 
Install dependencies using:
```bash
conda create -n StateEnv python=3.8.18
conda activate StateEnv
pip install -r requirements.txt
```

---

## **How to Use**

### **1. Data Collection**
Collect pose landmarks from a camera, video, or images.  
Run:
```bash
python landmark_extractor.py --source [camera/video/image] --data_path [path/to/media] --state [Asleep/Awake] --output_csv [path/to/output.csv]
```

### **2. Train the Model**
Train a deep learning model using preprocessed data.  
Run:
```bash
python train_model.py --data_path [path/to/dataset.csv] --output_model [path/to/save/model.pkl] --epochs 100 --batch_size 32 --test_size 0.3 --state_column state
```

### **3. Monitor in Real-Time**
Use the trained model to classify states from media sources in real time.  
Run:
```bash
python landmark_monitoring.py --source [camera/video/image] --data_path [path/to/media] --model_path [path/to/model.pkl]
```
---

## **Acknowledgments**
This project uses cutting-edge technologies like **MediaPipe**, **OpenCV**, and **Keras** to enable lightweight and effective body pose classification. It is an evolving solution aimed at improving **baby monitoring systems** and related applications.