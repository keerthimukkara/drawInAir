"""
Complete User Interface for Draw in Air Project
Author: Madhavi
Purpose: Intuitive interface for real-time air drawing
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import cv2
import os
import mediapipe as mp
from PIL import Image, ImageTk
import numpy as np
from collections import deque
import json
from datetime import datetime
from gesture_recognition import TemplateMatching, DynamicTimeWarping
import pickle
import glob
import threading
import subprocess
import sys
import time
import random
from tkinter import ttk


class DrawInAirUI:
    """
    Main application window with complete functionality
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Draw in Air - Computer Vision Project")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Drawing state
        self.drawing_mode = False
        self.drawing_points = deque(maxlen=1000)
        self.canvas_points = []
        
        # Camera
        self.cap = None
        self.camera_running = False
        
        # Colors and settings
        self.current_color = (0, 255, 0)
        self.brush_size = 3
        self.smoothing_enabled = True
        self.show_hand_landmarks = True
        
        # Smoothing buffer
        self.smooth_buffer_x = deque(maxlen=5)
        self.smooth_buffer_y = deque(maxlen=5)
        
        # Recognition mode
        self.recognition_mode = False
        self.gesture_buffer = []
        # Gesture control debounce
        self.gesture_hold_count = 0
        self.gesture_last_state = None
        self.gesture_frames_required = 6  # number of consecutive frames to confirm a gesture

        # Create UI
        self.create_widgets()
        # Create recognizers and load synthetic templates
        try:
            self.template_recognizer = TemplateMatching()
            self.dtw_recognizer = DynamicTimeWarping()
            self._create_synthetic_templates()
            # Try to load dataset templates (if available)
            try:
                self._load_templates_from_dataset('gesture_dataset')
            except Exception:
                pass
            # Try to load trained RandomForest recognizer (prefer focused model if present)
            try:
                focused_path = os.path.join('models', 'focused_random_forest.pkl')
                default_path = os.path.join('models', 'random_forest_recognizer.pkl')
                self.rf_recognizer = None
                if os.path.exists(focused_path):
                    with open(focused_path, 'rb') as f:
                        self.rf_recognizer = pickle.load(f)
                        print(f'Loaded RandomForest recognizer from {focused_path}')
                elif os.path.exists(default_path):
                    with open(default_path, 'rb') as f:
                        self.rf_recognizer = pickle.load(f)
                        print(f'Loaded RandomForest recognizer from {default_path}')
                else:
                    self.rf_recognizer = None
            except Exception:
                self.rf_recognizer = None
        except Exception:
            self.template_recognizer = None
            self.dtw_recognizer = None
        
        # Start camera
        self.start_camera()
        
        # Update loop
        self.update_frame()
        # Keybindings
        self.root.bind('<s>', lambda e: self.save_drawing(from_key=True))
        self.root.bind('<S>', lambda e: self.save_drawing(from_key=True))
    
    def create_widgets(self):
        """Build the complete user interface"""
        
        # ============ TITLE BAR ============
        title_frame = tk.Frame(self.root, bg='#34495e', height=80)
        title_frame.pack(fill=tk.X, side=tk.TOP)
        
        title_label = tk.Label(
            title_frame,
            text="✨ Draw in Air ✨",
            font=('Arial', 28, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        title_label.pack(pady=20)
        
        subtitle = tk.Label(
            title_frame,
            text="Computer Vision Gesture Recognition System",
            font=('Arial', 12),
            bg='#34495e',
            fg='#bdc3c7'
        )
        subtitle.pack()
        
        # ============ MAIN CONTAINER ============
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel - Controls (scrollable)
        left_container = tk.Frame(main_container, bg='#34495e', width=320)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        left_canvas = tk.Canvas(left_container, bg='#34495e', width=320, highlightthickness=0)
        left_scrollbar = tk.Scrollbar(left_container, orient=tk.VERTICAL, command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        left_panel = tk.Frame(left_canvas, bg='#34495e')
        left_canvas.create_window((0, 0), window=left_panel, anchor='nw')

        def _on_left_panel_config(event):
            left_canvas.configure(scrollregion=left_canvas.bbox('all'))

        left_panel.bind('<Configure>', _on_left_panel_config)
        
        # Right Panel - Video Feed
        right_panel = tk.Frame(main_container, bg='#2c3e50')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ============ LEFT PANEL CONTENTS ============
        
        # Section 1: Drawing Controls
        controls_frame = tk.LabelFrame(
            left_panel,
            text="Drawing Controls",
            font=('Arial', 12, 'bold'),
            bg='#34495e',
            fg='#ecf0f1',
            padx=10,
            pady=10
        )
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Start/Stop Drawing Button
        self.draw_button = tk.Button(
            controls_frame,
            text="🎨 Start Drawing",
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            command=self.toggle_drawing,
            height=2,
            relief=tk.RAISED,
            bd=3
        )
        self.draw_button.pack(fill=tk.X, pady=5)
        
        # Clear Canvas Button
        clear_button = tk.Button(
            controls_frame,
            text="🗑️ Clear Canvas",
            font=('Arial', 11),
            bg='#e74c3c',
            fg='white',
            command=self.clear_canvas
        )
        clear_button.pack(fill=tk.X, pady=5)
        
        # Section 2: Color Selection
        color_frame = tk.LabelFrame(
            left_panel,
            text="Color Palette",
            font=('Arial', 12, 'bold'),
            bg='#34495e',
            fg='#ecf0f1',
            padx=10,
            pady=10
        )
        color_frame.pack(fill=tk.X, padx=10, pady=10)
        
        colors = [
            ('Green', '#27ae60', (0, 255, 0)),
            ('Blue', '#3498db', (255, 0, 0)),
            ('Red', '#e74c3c', (0, 0, 255)),
            ('Yellow', '#f39c12', (0, 255, 255)),
            ('Purple', '#9b59b6', (255, 0, 255)),
            ('White', '#ecf0f1', (255, 255, 255))
        ]
        
        color_buttons_frame = tk.Frame(color_frame, bg='#34495e')
        color_buttons_frame.pack()
        
        for i, (name, hex_color, bgr_color) in enumerate(colors):
            btn = tk.Button(
                color_buttons_frame,
                text=name,
                bg=hex_color,
                fg='white' if i < 5 else 'black',
                font=('Arial', 9, 'bold'),
                width=8,
                command=lambda c=bgr_color: self.set_color(c)
            )
            btn.grid(row=i//3, column=i%3, padx=2, pady=2)
        
        # Section 3: Brush Settings
        brush_frame = tk.LabelFrame(
            left_panel,
            text="Brush Settings",
            font=('Arial', 12, 'bold'),
            bg='#34495e',
            fg='#ecf0f1',
            padx=10,
            pady=10
        )
        brush_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            brush_frame,
            text="Brush Size:",
            bg='#34495e',
            fg='#ecf0f1',
            font=('Arial', 10)
        ).pack(anchor=tk.W)
        
        self.brush_slider = tk.Scale(
            brush_frame,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            bg='#34495e',
            fg='#ecf0f1',
            highlightthickness=0,
            command=self.update_brush_size
        )
        self.brush_slider.set(3)
        self.brush_slider.pack(fill=tk.X)
        
        # Section 4: Advanced Options
        options_frame = tk.LabelFrame(
            left_panel,
            text="Options",
            font=('Arial', 12, 'bold'),
            bg='#34495e',
            fg='#ecf0f1',
            padx=10,
            pady=10
        )
        options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.smoothing_var = tk.BooleanVar(value=True)
        smoothing_check = tk.Checkbutton(
            options_frame,
            text="Enable Smoothing",
            variable=self.smoothing_var,
            bg='#34495e',
            fg='#ecf0f1',
            selectcolor='#2c3e50',
            font=('Arial', 10),
            command=self.toggle_smoothing
        )
        smoothing_check.pack(anchor=tk.W)
        
        self.landmarks_var = tk.BooleanVar(value=True)
        landmarks_check = tk.Checkbutton(
            options_frame,
            text="Show Hand Landmarks",
            variable=self.landmarks_var,
            bg='#34495e',
            fg='#ecf0f1',
            selectcolor='#2c3e50',
            font=('Arial', 10),
            command=self.toggle_landmarks
        )
        landmarks_check.pack(anchor=tk.W)
        
        # Section 5: Gesture Recognition
        gesture_frame = tk.LabelFrame(
            left_panel,
            text="Gesture Recognition",
            font=('Arial', 12, 'bold'),
            bg='#34495e',
            fg='#ecf0f1',
            padx=10,
            pady=10
        )
        gesture_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.recognize_button = tk.Button(
            gesture_frame,
            text="🔍 Recognize Gesture",
            font=('Arial', 11),
            bg='#3498db',
            fg='white',
            command=self.start_recognition
        )
        self.recognize_button.pack(fill=tk.X, pady=5)
        
        self.gesture_label = tk.Label(
            gesture_frame,
            text="Draw a gesture...",
            bg='#34495e',
            fg='#ecf0f1',
            font=('Arial', 10),
            wraplength=250
        )
        self.gesture_label.pack(pady=5)
        
        # Section 6: File Operations
        file_frame = tk.LabelFrame(
            left_panel,
            text="File Operations",
            font=('Arial', 12, 'bold'),
            bg='#34495e',
            fg='#ecf0f1',
            padx=10,
            pady=10
        )
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        save_button = tk.Button(
            file_frame,
            text="💾 Save Drawing",
            font=('Arial', 10),
            bg='#16a085',
            fg='white',
            command=self.save_drawing
        )
        save_button.pack(fill=tk.X, pady=2)
        
        export_button = tk.Button(
            file_frame,
            text="📤 Export Data",
            font=('Arial', 10),
            bg='#8e44ad',
            fg='white',
            command=self.export_data
        )
        export_button.pack(fill=tk.X, pady=2)
        # Record labeled gesture button
        record_button = tk.Button(
            file_frame,
            text="🎙️ Record Labeled Gesture",
            font=('Arial', 10),
            bg='#f39c12',
            fg='white',
            command=self.start_label_recording
        )
        record_button.pack(fill=tk.X, pady=2)

        save_label_button = tk.Button(
            file_frame,
            text="🔴 Save Labeled Sample",
            font=('Arial', 10),
            bg='#c0392b',
            fg='white',
            command=self.save_labeled_sample
        )
        save_label_button.pack(fill=tk.X, pady=2)

        retrain_button = tk.Button(
            file_frame,
            text="🔁 Retrain Model",
            font=('Arial', 10),
            bg='#2d98da',
            fg='white',
            command=self.retrain_model
        )
        retrain_button.pack(fill=tk.X, pady=2)

        # Progress bar and percentage label for retraining
        self.retrain_progress = ttk.Progressbar(file_frame, orient='horizontal', mode='determinate', maximum=100)
        self.retrain_progress.pack(fill=tk.X, pady=(6, 2))
        self.retrain_progress_label = tk.Label(file_frame, text='Idle', bg='#34495e', fg='#ecf0f1', font=('Arial', 9))
        self.retrain_progress_label.pack(pady=(0, 6))
        # Last saved thumbnail
        self.saved_thumb_label = tk.Label(
            file_frame,
            text="No saved image",
            bg='#34495e',
            fg='#ecf0f1',
            font=('Arial', 9),
            width=28,
            height=6,
            anchor=tk.CENTER
        )
        self.saved_thumb_label.pack(pady=6)
        
        # ============ RIGHT PANEL - VIDEO FEED ============
        
        # Status Bar
        status_bar = tk.Frame(right_panel, bg='#34495e', height=40)
        status_bar.pack(fill=tk.X, side=tk.TOP)
        
        self.status_label = tk.Label(
            status_bar,
            text="🎥 Camera: Active | Mode: Idle",
            bg='#34495e',
            fg='#2ecc71',
            font=('Arial', 11),
            anchor=tk.W,
            padx=10
        )
        self.status_label.pack(fill=tk.BOTH, expand=True)
        
        # Video Canvas
        self.video_canvas = tk.Canvas(
            right_panel,
            bg='black',
            highlightthickness=2,
            highlightbackground='#7f8c8d'
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Instructions Label
        instructions = tk.Label(
            right_panel,
            text="👆 Point with your index finger | ✋ Open palm to draw | ✊ Closed fist to stop",
            bg='#2c3e50',
            fg='#95a5a6',
            font=('Arial', 10, 'italic'),
            pady=10
        )
        instructions.pack(side=tk.BOTTOM)
    
    def start_camera(self):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera_running = True
    
    def update_frame(self):
        """Main update loop for video processing"""
        if not self.camera_running or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Create canvas for drawing
        drawing_canvas = np.zeros_like(frame)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Process hand landmarks (use handedness for correct thumb detection)
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks if enabled
                if self.show_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                
                # Get index finger tip (landmark 8)
                index_tip = hand_landmarks.landmark[8]
                x, y = int(index_tip.x * w), int(index_tip.y * h)

                # Count fingers raised (0/1 for each finger)
                hand_label = handedness.classification[0].label if handedness is not None else 'Right'
                fingers = self.count_fingers(hand_landmarks, hand_label)

                # Gesture control (open palm to start, fist to stop) with debounce
                if fingers == [1, 1, 1, 1, 1]:
                    gesture_state = 'open'
                elif fingers == [0, 0, 0, 0, 0]:
                    gesture_state = 'fist'
                else:
                    gesture_state = 'other'

                if gesture_state == self.gesture_last_state:
                    self.gesture_hold_count += 1
                else:
                    self.gesture_hold_count = 1
                    self.gesture_last_state = gesture_state

                if self.gesture_hold_count >= self.gesture_frames_required:
                    if gesture_state == 'open' and not self.drawing_mode:
                        # Start drawing
                        self.drawing_mode = True
                        self.draw_button.config(text="⏸️ Stop Drawing", bg='#e74c3c')
                        self.status_label.config(text="🎥 Camera: Active | Mode: Drawing", fg='#e74c3c')
                        self.drawing_points.clear()
                    elif gesture_state == 'fist' and self.drawing_mode:
                        # Stop drawing
                        self.drawing_mode = False
                        self.draw_button.config(text="🎨 Start Drawing", bg='#27ae60')
                        self.status_label.config(text="🎥 Camera: Active | Mode: Idle", fg='#2ecc71')
                
                # Apply smoothing if enabled
                if self.smoothing_enabled:
                    x, y = self.apply_smoothing(x, y)
                
                # Draw tracking circle
                cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)
                
                # Drawing logic
                if self.drawing_mode:
                    self.drawing_points.append((x, y))
                    
                    # Draw on canvas
                    if len(self.drawing_points) > 1:
                        points = list(self.drawing_points)
                        for i in range(len(points) - 1):
                            cv2.line(
                                drawing_canvas,
                                points[i],
                                points[i + 1],
                                self.current_color,
                                self.brush_size
                            )
        
        # Overlay drawing on frame
        mask = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        frame = cv2.add(frame, drawing_canvas)
        
        # Add mode indicator
        mode_text = "DRAWING" if self.drawing_mode else "IDLE"
        mode_color = (0, 255, 0) if self.drawing_mode else (200, 200, 200)
        cv2.putText(frame, mode_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, mode_color, 3)
        
        # Convert to PhotoImage and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Resize to fit canvas
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img = img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.video_canvas.imgtk = imgtk
        
        # Schedule next update
        self.root.after(10, self.update_frame)

    def count_fingers(self, hand_landmarks, hand_label):
        """Return a list of 5 elements (thumb,index,middle,ring,pinky) as 0/1 for raised fingers."""
        landmarks = hand_landmarks.landmark

        # Helper: tip and pip indices
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]

        fingers = [0, 0, 0, 0, 0]

        # Thumb: compare x depending on handedness
        try:
            if hand_label.lower() == 'right':
                fingers[0] = 1 if landmarks[tips[0]].x < landmarks[pips[0]].x else 0
            else:
                fingers[0] = 1 if landmarks[tips[0]].x > landmarks[pips[0]].x else 0
        except Exception:
            fingers[0] = 0

        # Other fingers: tip y < pip y means finger is up (y=0 at top)
        for i in range(1, 5):
            try:
                fingers[i] = 1 if landmarks[tips[i]].y < landmarks[pips[i]].y else 0
            except Exception:
                fingers[i] = 0

        return fingers
    
    def apply_smoothing(self, x, y):
        """Apply moving average smoothing"""
        self.smooth_buffer_x.append(x)
        self.smooth_buffer_y.append(y)
        
        return int(np.mean(self.smooth_buffer_x)), int(np.mean(self.smooth_buffer_y))
    
    def toggle_drawing(self):
        """Start/stop drawing mode"""
        self.drawing_mode = not self.drawing_mode
        
        if self.drawing_mode:
            self.draw_button.config(
                text="⏸️ Stop Drawing",
                bg='#e74c3c'
            )
            self.status_label.config(
                text="🎥 Camera: Active | Mode: Drawing",
                fg='#e74c3c'
            )
            self.drawing_points.clear()
        else:
            self.draw_button.config(
                text="🎨 Start Drawing",
                bg='#27ae60'
            )
            self.status_label.config(
                text="🎥 Camera: Active | Mode: Idle",
                fg='#2ecc71'
            )
    
    def clear_canvas(self):
        """Clear all drawings"""
        self.drawing_points.clear()
        self.canvas_points.clear()
        messagebox.showinfo("Canvas Cleared", "All drawings have been cleared!")
    
    def set_color(self, color):
        """Set drawing color"""
        self.current_color = color
    
    def update_brush_size(self, value):
        """Update brush size"""
        self.brush_size = int(value)
    
    def toggle_smoothing(self):
        """Toggle smoothing on/off"""
        self.smoothing_enabled = self.smoothing_var.get()
    
    def toggle_landmarks(self):
        """Toggle hand landmarks display"""
        self.show_hand_landmarks = self.landmarks_var.get()
    
    def start_recognition(self):
        """Start gesture recognition mode"""
        if len(self.drawing_points) < 10:
            messagebox.showwarning(
                "Insufficient Data",
                "Please draw a gesture first!"
            )
            return
        # Preprocess drawing: convert to numpy, resample, normalize
        pts = np.array(list(self.drawing_points))
        # resample to 128
        n = 128
        if len(pts) < 2:
            messagebox.showwarning("Insufficient Data", "Please draw a longer gesture!")
            return

        old_idx = np.linspace(0, len(pts) - 1, len(pts))
        new_idx = np.linspace(0, len(pts) - 1, n)
        resampled = np.zeros((n, pts.shape[1]))
        for dim in range(pts.shape[1]):
            resampled[:, dim] = np.interp(new_idx, old_idx, pts[:, dim])

        # center and scale
        centroid = np.mean(resampled, axis=0)
        resampled = resampled - centroid
        max_dist = np.max(np.linalg.norm(resampled, axis=1))
        if max_dist > 0:
            resampled = resampled / max_dist

        best_label = None
        best_conf = 0.0

        # Template matching
        if getattr(self, 'template_recognizer', None) is not None:
            label, conf = self.template_recognizer.recognize(resampled)
            if label and conf > best_conf:
                best_conf = conf
                best_label = label

        # DTW
        if getattr(self, 'dtw_recognizer', None) is not None:
            label2, conf2 = self.dtw_recognizer.recognize(resampled)
            if label2 and conf2 > best_conf:
                best_conf = conf2
                best_label = label2

        # RandomForest recognizer (if available)
        if getattr(self, 'rf_recognizer', None) is not None:
            try:
                label3, conf3 = self.rf_recognizer.recognize(resampled)
                if label3 and conf3 > best_conf:
                    best_conf = conf3
                    best_label = label3
            except Exception:
                # ignore RF errors and continue with other recognizers
                pass

        if best_label is None:
            messagebox.showinfo("Gesture Recognized", "Could not confidently recognize the gesture.")
            self.gesture_label.config(text="Detected: - \nConfidence: 0%")
            return

        # Present result
        self.gesture_label.config(text=f"Detected: {best_label}\nConfidence: {best_conf:.2%}")
        messagebox.showinfo("Gesture Recognized", f"Gesture: {best_label}\nConfidence: {best_conf:.2%}")
    
    def save_drawing(self, from_key=False, filename=None):
        """Save current drawing as image. If called from key ('s'), save to saved_drawings auto.
        If filename provided, use it (used by dialog).
        """
        if len(self.drawing_points) == 0:
            messagebox.showwarning("No Drawing", "Nothing to save!")
            return

        # If not provided filename and not from_key, ask the user
        if filename is None and not from_key:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )

        # Auto-save to saved_drawings when triggered via keyboard
        if filename is None and from_key:
            os.makedirs('saved_drawings', exist_ok=True)
            filename = os.path.join('saved_drawings', f'drawing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')

        if filename:
            # Create blank canvas (1280x720, same as camera capture)
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

            # Draw points (points are in camera coordinates)
            points = list(self.drawing_points)
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i + 1],
                         self.current_color, self.brush_size)

            cv2.imwrite(filename, canvas)

            # Show saved thumbnail in UI
            try:
                thumb = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                thumb = thumb.resize((220, 140), Image.Resampling.LANCZOS)
                thumb_tk = ImageTk.PhotoImage(thumb)
                self.saved_thumb_label.config(image=thumb_tk, text='')
                self.saved_thumb_label.image = thumb_tk
            except Exception:
                # ignore thumbnail errors
                pass

            messagebox.showinfo("Saved", f"Drawing saved to {filename}")

    def start_label_recording(self):
        """Prompt the user for a label and enter labeled-recording mode."""
        label = simpledialog.askstring("Record Labeled Gesture", "Enter gesture label:")
        if not label:
            return
        # normalize label
        label = label.strip()
        self.pending_record_label = label
        self.is_recording_label = True
        self.status_label.config(text=f"🎥 Camera: Active | Recording label: {label}", fg='#e67e22')
        messagebox.showinfo("Recording", f"Recording enabled for label: {label}\nDraw the gesture and press 'Save Labeled Sample' when done.")

    def save_labeled_sample(self):
        """Save the current drawing as a labeled sample into `gesture_dataset/`.

        The saved JSON follows the dataset entry style with normalized landmarks (x,y,z=0).
        """
        if not getattr(self, 'is_recording_label', False) or not getattr(self, 'pending_record_label', None):
            messagebox.showwarning("Not Recording", "No active recording label. Click 'Record Labeled Gesture' first.")
            return

        label = self.pending_record_label
        if len(self.drawing_points) == 0:
            messagebox.showwarning("No Drawing", "Please draw the gesture before saving the labeled sample.")
            return

        # Normalize points to 0..1 using canvas size (same as save_drawing uses 1280x720)
        width = 1280
        height = 720
        pts = list(self.drawing_points)
        landmarks = []
        for (x, y) in pts:
            xn = float(x) / float(width)
            yn = float(y) / float(height)
            landmarks.extend([xn, yn, 0.0])

        index_tip = landmarks[-3:] if len(landmarks) >= 3 else [0.5, 0.5, 0.0]

        payload = {
            'session_id': f"user_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'total_samples': 1,
            'gesture_types': [label],
            'data': [
                {
                    'gesture_type': label,
                    'landmarks': landmarks,
                    'index_tip': index_tip,
                    'timestamp': datetime.now().isoformat()
                }
            ]
        }

        os.makedirs('gesture_dataset', exist_ok=True)
        fname = os.path.join('gesture_dataset', f"user_sample_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(fname, 'w') as f:
                json.dump(payload, f, indent=2)
            messagebox.showinfo('Saved', f'Labeled sample saved to {fname}')
            # Reset recording state
            self.is_recording_label = False
            self.pending_record_label = None
            self.status_label.config(text="🎥 Camera: Active | Mode: Idle", fg='#2ecc71')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to save labeled sample: {e}')

    def retrain_model(self):
        """Start retraining in a background thread by running train_focus_gestures.py."""
        script_path = os.path.join(os.getcwd(), 'train_focus_gestures.py')
        if not os.path.exists(script_path):
            messagebox.showerror('Script missing', 'train_focus_gestures.py not found in project root.')
            return

        if messagebox.askyesno('Retrain model', 'This will retrain the focused model (may take several minutes). Continue?'):
            # disable UI briefly
            self.status_label.config(text='🔁 Retraining model (running)...', fg='#f1c40f')
            thread = threading.Thread(target=self._run_retrain, args=(script_path,), daemon=True)
            thread.start()

    def _run_retrain(self, script_path):
        """Run the training script and reload the focused model when done."""
        try:
            # Run with same Python executable, capture output to a log and update progress
            log_path = os.path.join('models', 'retrain_log.txt')
            start_time = time.time()
            expected_duration = 300.0  # seconds heuristic for percentage estimation

            with open(log_path, 'w', encoding='utf-8') as logf:
                proc = subprocess.Popen([sys.executable, script_path], stdout=logf, stderr=logf)

                # update loop: estimate progress based on elapsed time (heuristic)
                progress = 0

                def ui_update(pct, text=None):
                    def _upd():
                        try:
                            self.retrain_progress['value'] = pct
                            self.retrain_progress_label.config(text=(text or f'Retraining... {pct}%'))
                        except Exception:
                            pass
                    self.root.after(0, _upd)

                # kick off with small visible progress
                ui_update(2, 'Retraining... 2%')

                while proc.poll() is None:
                    elapsed = time.time() - start_time
                    # map elapsed to progress up to 95%
                    pct = int(min(95, 5 + (elapsed / max(1.0, expected_duration)) * 90))
                    # add a small random jitter to make it feel alive
                    pct = min(95, pct + random.randint(0, 3))
                    if pct != progress:
                        progress = pct
                        ui_update(progress)
                    time.sleep(1.0)

                # process finished
                returncode = proc.returncode

            def on_complete():
                # Attempt to load the newly trained model
                focused_path = os.path.join('models', 'focused_random_forest.pkl')
                if os.path.exists(focused_path) and returncode == 0:
                    try:
                        with open(focused_path, 'rb') as mf:
                            self.rf_recognizer = pickle.load(mf)
                        messagebox.showinfo('Retrain complete', f'Retraining finished and model updated. Log saved to {log_path}')
                        self.retrain_progress['value'] = 100
                        self.retrain_progress_label.config(text='Retrain complete (100%)')
                        self.status_label.config(text='🎥 Camera: Active | Mode: Idle', fg='#2ecc71')
                    except Exception as e:
                        messagebox.showwarning('Reload failed', f'Retraining finished but failed to load model: {e}\nSee log: {log_path}')
                        self.retrain_progress['value'] = 0
                        self.retrain_progress_label.config(text='Reload failed')
                        self.status_label.config(text='🎥 Camera: Active | Mode: Idle', fg='#2ecc71')
                else:
                    messagebox.showerror('Retrain failed', f'Retraining did not produce model or exited with errors. See log: {log_path}')
                    self.retrain_progress['value'] = 0
                    self.retrain_progress_label.config(text='Retrain failed')
                    self.status_label.config(text='🎥 Camera: Active | Mode: Idle', fg='#2ecc71')

            # Schedule UI update on main thread
            self.root.after(100, on_complete)
        except Exception as e:
            def on_err():
                messagebox.showerror('Retrain error', f'Error running retrain: {e}')
                self.status_label.config(text='🎥 Camera: Active | Mode: Idle', fg='#2ecc71')
            self.root.after(100, on_err)

    def _create_synthetic_templates(self):
        """Create simple synthetic gesture templates (resampled & normalized) to improve recognition.
        These are quick, deterministic templates for common shapes the app expects.
        """
        def resample_and_normalize(points, n=128):
            pts = np.array(points)
            old_idx = np.linspace(0, len(pts) - 1, len(pts))
            new_idx = np.linspace(0, len(pts) - 1, n)
            res = np.zeros((n, 2))
            for d in range(2):
                res[:, d] = np.interp(new_idx, old_idx, pts[:, d])
            centroid = np.mean(res, axis=0)
            res = res - centroid
            md = np.max(np.linalg.norm(res, axis=1))
            if md > 0:
                res = res / md
            return res

        # Circle
        t = np.linspace(0, 2 * np.pi, 64)
        circle = np.stack([np.cos(t), np.sin(t)], axis=1)
        circle = resample_and_normalize(circle)
        self.template_recognizer.add_template('circle', circle)
        self.dtw_recognizer.add_template('circle', circle)

        # Star (5-pointed)
        def star_points(R=1.0, r=0.4, k=5):
            pts = []
            for i in range(2 * k):
                ang = i * np.pi / k
                rad = R if i % 2 == 0 else r
                pts.append([rad * np.cos(ang), rad * np.sin(ang)])
            return np.array(pts)

        star = resample_and_normalize(star_points())
        self.template_recognizer.add_template('star', star)
        self.dtw_recognizer.add_template('star', star)

        # Triangle
        tri = np.array([[0, -1], [0.95, 0.5], [-0.95, 0.5]])
        tri = resample_and_normalize(np.vstack([tri, tri[0]]))
        self.template_recognizer.add_template('triangle', tri)
        self.dtw_recognizer.add_template('triangle', tri)

        # Rectangle
        rect = np.array([[-1, -0.6], [1, -0.6], [1, 0.6], [-1, 0.6], [-1, -0.6]])
        rect = resample_and_normalize(rect)
        self.template_recognizer.add_template('rectangle', rect)
        self.dtw_recognizer.add_template('rectangle', rect)

        # Line horizontal and vertical
        line_h = np.array([[-1, 0], [1, 0]])
        line_h = resample_and_normalize(np.vstack([line_h, line_h[::-1]]))
        self.template_recognizer.add_template('line', line_h)
        self.dtw_recognizer.add_template('line', line_h)

        # Zigzag
        zz = np.array([[-1, 0.8], [-0.5, -0.8], [0, 0.8], [0.5, -0.8], [1, 0.8]])
        zz = resample_and_normalize(zz)
        self.template_recognizer.add_template('zigzag', zz)
        self.dtw_recognizer.add_template('zigzag', zz)

        # Wave (sine)
        tx = np.linspace(0, 2 * np.pi, 64)
        wave = np.stack([tx / (2 * np.pi) * 2 - 1, 0.5 * np.sin(2 * tx)], axis=1)
        wave = resample_and_normalize(wave)
        self.template_recognizer.add_template('wave', wave)
        self.dtw_recognizer.add_template('wave', wave)

        # Spiral
        th = np.linspace(0, 4 * np.pi, 200)
        r = np.linspace(0.1, 1.0, len(th))
        spiral = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
        spiral = resample_and_normalize(spiral)
        self.template_recognizer.add_template('spiral', spiral)
        self.dtw_recognizer.add_template('spiral', spiral)

    def _load_templates_from_dataset(self, dataset_dir='gesture_dataset'):
        """Load gesture templates from JSON files produced by DatasetCollector.
        Expects each data entry to have 'gesture_type' and 'landmarks' as a flat list of xyz values
        where landmarks can be reshaped to (-1, 3). We'll extract x,y trajectory and add templates.
        """
        files = glob.glob(os.path.join(dataset_dir, '*.json'))
        for f in files:
            try:
                with open(f, 'r') as fh:
                    j = json.load(fh)
            except Exception:
                continue

            for entry in j.get('data', []):
                gtype = entry.get('gesture_type') or entry.get('label') or 'unknown'
                lm = entry.get('landmarks')
                if not lm:
                    # try index_tip fallback
                    it = entry.get('index_tip')
                    if it and isinstance(it, (list, tuple)):
                        pts = np.array([it[:2]])
                    else:
                        continue
                else:
                    try:
                        arr = np.array(lm).reshape(-1, 3)
                        pts = arr[:, :2]
                    except Exception:
                        continue

                # if too short, skip
                if len(pts) < 4:
                    continue

                # resample/normalize same as templates
                n = 128
                old_idx = np.linspace(0, len(pts) - 1, len(pts))
                new_idx = np.linspace(0, len(pts) - 1, n)
                res = np.zeros((n, 2))
                for d in range(2):
                    res[:, d] = np.interp(new_idx, old_idx, pts[:, d])
                centroid = np.mean(res, axis=0)
                res = res - centroid
                md = np.max(np.linalg.norm(res, axis=1))
                if md > 0:
                    res = res / md

                # add to recognizers
                try:
                    if self.template_recognizer is not None:
                        self.template_recognizer.add_template(gtype, res)
                    if self.dtw_recognizer is not None:
                        self.dtw_recognizer.add_template(gtype, res)
                except Exception:
                    continue
    
    def export_data(self):
        """Export drawing data as JSON"""
        if len(self.drawing_points) == 0:
            messagebox.showwarning("No Data", "Nothing to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            data = {
                'timestamp': datetime.now().isoformat(),
                'points': list(self.drawing_points),
                'color': self.current_color,
                'brush_size': self.brush_size
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            messagebox.showinfo("Exported", f"Data exported to {filename}")
    
    def __del__(self):
        """Cleanup resources"""
        if self.cap is not None:
            self.cap.release()


# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawInAirUI(root)
    root.mainloop()