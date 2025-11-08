"""
Dataset Collection Module for Draw in Air Project
Author: Keerthi
Purpose: Collect hand gesture data for various drawing patterns
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import warnings
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime

# ... rest of code

class DatasetCollector:
    def __init__(self, save_dir='gesture_dataset'):
        """
        Initialize dataset collector with MediaPipe hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Gesture categories to collect
        self.gesture_types = [
            'line_horizontal',
            'line_vertical',
            'circle_clockwise',
            'circle_counterclockwise',
            'zigzag',
            'wave',
            'spiral',
            'rectangle',
            'triangle',
            'star'
        ]
        
        self.current_gesture = 0
        self.recording = False
        self.gesture_data = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def collect_data(self):
        """
        Main data collection loop
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Dataset Collection Started!")
        print("Controls:")
        print("  SPACE: Start/Stop recording")
        print("  N: Next gesture type")
        print("  S: Save current session")
        print("  Q: Quit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand landmarks
            results = self.hands.process(rgb_frame)
            
            # Display current gesture type
            gesture_name = self.gesture_types[self.current_gesture]
            status = "RECORDING" if self.recording else "IDLE"
            
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Status: {status}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (0, 0, 255) if self.recording else (255, 255, 255), 2)
            cv2.putText(frame, f"Samples: {len(self.gesture_data)}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract index finger tip (landmark 8)
                    index_tip = hand_landmarks.landmark[8]
                    x, y = int(index_tip.x * w), int(index_tip.y * h)
                    
                    # Draw tracking point
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                    
                    # Record data if recording
                    if self.recording:
                        landmark_data = []
                        for lm in hand_landmarks.landmark:
                            landmark_data.extend([lm.x, lm.y, lm.z])
                        
                        self.gesture_data.append({
                            'gesture_type': gesture_name,
                            'landmarks': landmark_data,
                            'index_tip': [index_tip.x, index_tip.y, index_tip.z],
                            'timestamp': datetime.now().isoformat()
                        })
            
            cv2.imshow('Dataset Collection', frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Toggle recording
                self.recording = not self.recording
                print(f"Recording {'started' if self.recording else 'stopped'}")
                
            elif key == ord('n'):  # Next gesture
                self.current_gesture = (self.current_gesture + 1) % len(self.gesture_types)
                print(f"Switched to: {self.gesture_types[self.current_gesture]}")
                
            elif key == ord('s'):  # Save session
                self.save_session()
                
            elif key == ord('q'):  # Quit
                self.save_session()
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_session(self):
        """
        Save collected data to JSON file
        """
        if not self.gesture_data:
            print("No data to save!")
            return
        
        filename = f"{self.save_dir}/session_{self.session_id}.json"
        
        metadata = {
            'session_id': self.session_id,
            'total_samples': len(self.gesture_data),
            'gesture_types': self.gesture_types,
            'data': self.gesture_data
        }
        
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(self.gesture_data)} samples to {filename}")
        
        # Generate statistics
        gesture_counts = {}
        for data in self.gesture_data:
            gt = data['gesture_type']
            gesture_counts[gt] = gesture_counts.get(gt, 0) + 1
        
        print("\nGesture Distribution:")
        for gesture, count in gesture_counts.items():
            print(f"  {gesture}: {count} samples")

# Usage
if __name__ == "__main__":
    collector = DatasetCollector()
    collector.collect_data()