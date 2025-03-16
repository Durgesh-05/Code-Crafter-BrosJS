import cv2
import numpy as np
import os
import json
import time
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import argparse

IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["NonViolence", "Violence"]

app = Flask(__name__)
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*")

model = None
processing_thread = None
is_processing = False

Global_dataset = [
    {"video_id": "V_11.mp4", "path": "dataset/V_11.mp4", "address":"Parvati angan apartment badlapur, Mumbai"},
    {"video_id": "V_100.mp4", "path": "dataset/V_100.mp4", "address":"BarhalGanj Gorakhpur, Uttarpradesh"},
    {"video_id": "V_101.mp4", "path": "dataset/V_101.mp4", "address": "12, MG Road, Bangalore, Karnataka"},
    {"video_id": "V_102.mp4", "path": "dataset/V_102.mp4", "address": "45, Park Street, Kolkata, West Bengal"},
    {"video_id": "NV.mp4", "path": "dataset/NV.mp4", "address": "45, Park Street, Kolkata, West Bengal"},
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

def create_cnn_model():
    model = Sequential()
    model.add(Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def process_videos_from_json():
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
                "confidence": result["confidence"],
                "processing_time": result["processing_time"],
                "model_accuracy": result["model_accuracy"]
            })
        except Exception as e:
            socketio.emit("video_result", {"video_id": video_id, "error": str(e), "status": "error"})

    socketio.emit("server_message", {"status": "completed", "message": "All videos processed"})
    is_processing = False

def extract_frames(video_path, skip_frames=7):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
        
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % skip_frames == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame.astype('float32') / 255.0
            frames.append(normalized_frame)
        frame_count += 1
    cap.release()
    return frames

def process_single_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    start_time = time.time()
    
    frames = extract_frames(video_path)
    if not frames:
        raise ValueError(f"No frames could be extracted from the video: {video_path}")
    
    predictions = []
    for frame in frames:
        frame_expanded = np.expand_dims(frame, axis=0)
        prediction = model.predict(frame_expanded, verbose=0)[0][0]
        predictions.append(prediction)
    
    predictions = np.array(predictions)
    violence_count = np.sum(predictions > 0.5)
    non_violence_count = len(predictions) - violence_count
    
    if len(predictions) > 0:
        violence_ratio = violence_count / len(predictions)
        avg_confidence = np.mean(predictions)
        
        result_class = "Violence" if violence_ratio > 0.3 else "NonViolence"
        result_confidence = float(round(avg_confidence * 100 if result_class == "Violence" else (1 - avg_confidence) * 100, 2))
        
        _, model_accuracy = model.evaluate(np.array(frames), 
                                          np.ones(len(frames)) if result_class == "Violence" else np.zeros(len(frames)), 
                                          verbose=0)
        model_accuracy = float(model_accuracy)

    else:
        result_class = "Unknown"
        result_confidence = 0
        model_accuracy = 0
    
    processing_time = time.time() - start_time
    
    return {
        "class": result_class,
        "confidence": result_confidence,
        "processing_time": round(processing_time, 2),
        "model_accuracy": round(model_accuracy * 100, 2)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='public/violence_detection_model.h5')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    print("Creating CNN model...")
    model = create_cnn_model()
    
    if os.path.exists(args.model):
        print(f"Loading model weights from {args.model}")
        model.load_weights(args.model)
    else:
        print(f"Model file {args.model} not found. Using untrained model.")
    
    print(f"Starting server on {args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=True)