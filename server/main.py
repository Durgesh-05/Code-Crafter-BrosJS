import cv2
import numpy as np
import os
import json
import time
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, TimeDistributed, Bidirectional, Flatten
from tensorflow.keras.applications import MobileNetV2
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import argparse

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["NonViolence", "Violence"]

# Flask app setup
app = Flask(__name__)
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*")

# Global Variables
model = None
processing_thread = None
is_processing = False

# Dataset
Global_dataset = [
    {"video_id": "V_11.mp4", "path": "dataset/V_11.mp4", "address":"Parvati angan apartment badlapur, Mumbai"},
    {"video_id": "V_100.mp4", "path": "dataset/V_100.mp4", "address":"BarhalGanj Gorakhpur, Uttarpradesh"},
    {"video_id": "V_101.mp4", "path": "dataset/V_101.mp4", "address": "12, MG Road, Bangalore, Karnataka"},
    {"video_id": "V_102.mp4", "path": "dataset/V_102.mp4", "address": "45, Park Street, Kolkata, West Bengal"},
    {"video_id": "V_103.mp4", "path": "dataset/V_103.mp4", "address": "78, Anna Salai, Chennai, Tamil Nadu"},
    {"video_id": "V_104.mp4", "path": "dataset/V_104.mp4", "address": "101, Banjara Hills, Hyderabad, Telangana"},
    {"video_id": "V_105.mp4", "path": "dataset/V_105.mp4", "address": "22, Sector 17, Chandigarh, Punjab"},
    {"video_id": "V_106.mp4", "path": "dataset/V_106.mp4", "address": "6, Ashram Road, Ahmedabad, Gujarat"},
    {"video_id": "V_107.mp4", "path": "dataset/V_107.mp4", "address": "99, Civil Lines, Jaipur, Rajasthan"},
    {"video_id": "V_108.mp4", "path": "dataset/V_108.mp4", "address": "33, Hazratganj, Lucknow, Uttar Pradesh"},
    {"video_id": "V_109.mp4", "path": "dataset/V_109.mp4", "address": "56, Ernakulam South, Kochi, Kerala"},
    {"video_id": "V_110.mp4", "path": "dataset/V_110.mp4", "address": "8, Shivaji Nagar, Pune, Maharashtra"},
    {"video_id": "V_1000.mp4", "path": "dataset/V_1000.mp4", "address": "25, Lal Chowk, Srinagar, Jammu and Kashmir"}
]

# Global_dataset = [
#     {"video_id": "V_11.mp4", "path": "dataset/V_11.mp4"},
#     {"video_id": "V_100.mp4", "path": "dataset/V_100.mp4"},
#     {"video_id": "V_101.mp4", "path": "dataset/V_101.mp4"},
#     {"video_id": "V_102.mp4", "path": "dataset/V_102.mp4"},
#     {"video_id": "V_103.mp4", "path": "dataset/V_103.mp4"},
#     {"video_id": "V_104.mp4", "path": "dataset/V_104.mp4"},
#     {"video_id": "V_105.mp4", "path": "dataset/V_105.mp4"},
#     {"video_id": "V_106.mp4", "path": "dataset/V_106.mp4"},
#     {"video_id": "V_107.mp4", "path": "dataset/V_107.mp4"},
#     {"video_id": "V_108.mp4", "path": "dataset/V_108.mp4"},
#     {"video_id": "V_109.mp4", "path": "dataset/V_109.mp4"},
#     {"video_id": "V_110.mp4", "path": "dataset/V_110.mp4"},
#     {"video_id": "V_1000.mp4", "path": "dataset/V_1000.mp4"},
# ]


@app.route('/')
def hello_world():
    return 'Violence Detection API is running'

@socketio.on("connect")
def connected():
    print("Client connected")
    emit("server_message", {"status": "connected", "message": "Connected to server"})

@socketio.on("disconnect")
def disconnected(reason):
    print(f"Client disconnected. Reason: {reason}")

@socketio.on("start_processing")
def handle_start_processing():
    global processing_thread, is_processing

    if is_processing:
        emit("server_message", {"status": "error", "message": "Processing already in progress"})
        return

    emit("server_message", {"status": "started", "message": "Starting video processing"})

    is_processing = True
    processing_thread = threading.Thread(target=process_videos_from_json)
    processing_thread.daemon = True
    processing_thread.start()

@socketio.on("stop_processing")
def handle_stop_processing():
    global is_processing
    is_processing = False
    emit("server_message", {"status": "stopped", "message": "Processing stopped by user"})

def create_model():
    """Create MoBiLSTM model"""
    mobilenet = MobileNetV2(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), include_top=False, weights='imagenet', pooling='avg')
    mobilenet.trainable = False
     
    lstm_fw = LSTM(units=32)
    lstm_bw = LSTM(units=32, go_backwards=True)  

    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        TimeDistributed(mobilenet),
        Dropout(0.25),
        TimeDistributed(Flatten()),
        Bidirectional(lstm_fw, backward_layer=lstm_bw),
        Dropout(0.25),
        Dense(512, activation='relu'),
        Dropout(0.25),
        Dense(256, activation='relu'),
        Dropout(0.25),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dropout(0.25),
        Dense(32, activation='relu'),
        Dropout(0.25),
        Dense(16, activation='relu'),
        Dropout(0.25),
        Dense(len(CLASSES_LIST), activation='sigmoid')
    ])

    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=["accuracy"])
    
    return model

def process_videos_from_json():
    """Process all videos in Global_dataset"""
    global is_processing

    socketio.emit("server_message", {"status": "info", "message": f"Processing {len(Global_dataset)} videos"})

    for idx, video_info in enumerate(Global_dataset):
        if not is_processing:
            break

        video_id = video_info["video_id"]
        video_path = video_info["path"]
        video_address = video_info["address"]

        if not os.path.exists(video_path):
            socketio.emit("video_result", {"video_id": video_id, "error": "File not found", "status": "error"})
            continue

        socketio.emit("processing_update", {"status": "processing", "current": idx + 1, "total": len(Global_dataset), "video_id": video_id})

        try:
            result = process_single_video(video_path)
            socketio.emit("video_result", {
                "video_id": video_id,
                "path": video_path,
                "address": video_address,
                "result": result["class"],
                "processing_time": result["processing_time"]
            })
        except Exception as e:
            socketio.emit("video_result", {"video_id": video_id, "error": str(e), "status": "error"})

    socketio.emit("server_message", {"status": "completed", "message": "All videos processed"})
    is_processing = False

def process_single_video(video_path):
    """Process a single video"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_reader = cv2.VideoCapture(video_path)
    if not video_reader.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    violence_frames = 0
    non_violence_frames = 0
    total_processed_frames = 0
    confidence_sum = 0
    start_time = time.time()

    while video_reader.isOpened() and is_processing:
        ok, frame = video_reader.read()
        if not ok:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        normalized_frame = rgb_frame / 255.0
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            input_frames = np.expand_dims(np.array(frames_queue), axis=0)
            prediction = model.predict(input_frames, verbose=0)[0]
            predicted_label = np.argmax(prediction)
            predicted_class = CLASSES_LIST[predicted_label]
            confidence = float(prediction[predicted_label])

            if predicted_class == "Violence":
                violence_frames += 1
            else:
                non_violence_frames += 1

            confidence_sum += confidence
            total_processed_frames += 1

    video_reader.release()
    processing_time = time.time() - start_time

    if total_processed_frames > 0:
        violence_ratio = violence_frames / total_processed_frames
        avg_confidence = confidence_sum / total_processed_frames

        result_class = "Violence" if violence_ratio > 0.3 else "NonViolence"
        result_confidence = round(violence_ratio * 100 if result_class == "Violence" else (1 - violence_ratio) * 100, 2)
    else:
        result_class = "Unknown"
        result_confidence = 0

    return {
        "class": result_class,
        "confidence": result_confidence,
        "processing_time": round(processing_time, 2)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./my_model_updated.keras')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    print("Loading model...")
    model = create_model()
    print(f"Starting server on {args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=True)