"""
Simple Draw in Air Integration (Standalone)
"""
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
from datetime import datetime

# Simple smoother class
class SimpleEMA:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.prev_x = None
        self.prev_y = None
    
    def filter(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return x, y
        smoothed_x = self.alpha * x + (1 - self.alpha) * self.prev_x
        smoothed_y = self.alpha * y + (1 - self.alpha) * self.prev_y
        self.prev_x, self.prev_y = smoothed_x, smoothed_y
        return smoothed_x, smoothed_y

class DrawInAirSystem:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize smoother
        self.smoother = SimpleEMA()
        
        # Drawing state
        self.drawing_mode = False
        self.drawing_points = deque(maxlen=1000)
        
        # Settings
        self.color = (0, 255, 0)  # Green
        self.brush_size = 3
        
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("="*60)
        print("Draw in Air - Simple Version")
        print("="*60)
        print("Controls:")
        print("  SPACE: Toggle drawing mode")
        print("  C: Clear canvas")
        print("  S: Save drawing")
        print("  Q: Quit")
        print("="*60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Create canvas
            canvas = np.zeros_like(frame)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Hand detection
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get index finger tip
                    index_tip = hand_landmarks.landmark[8]
                    x, y = int(index_tip.x * w), int(index_tip.y * h)
                    
                    # Smooth
                    x, y = self.smoother.filter(x, y)
                    x, y = int(x), int(y)
                    
                    # Draw tracking circle
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                    
                    # Drawing
                    if self.drawing_mode:
                        self.drawing_points.append((x, y))
                        
                        if len(self.drawing_points) > 1:
                            points = list(self.drawing_points)
                            for i in range(len(points) - 1):
                                cv2.line(canvas, points[i], points[i+1],
                                        self.color, self.brush_size)
            
            # Overlay canvas
            mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            frame = cv2.add(frame, canvas)
            
            # Add UI
            mode_text = "DRAWING" if self.drawing_mode else "TRACKING"
            mode_color = (0, 255, 0) if self.drawing_mode else (100, 100, 100)
            cv2.putText(frame, mode_text, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, mode_color, 3)
            
            # Instructions
            cv2.putText(frame, "SPACE: Draw | C: Clear | S: Save | Q: Quit",
                       (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.imshow('Draw in Air', frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                self.drawing_mode = not self.drawing_mode
                if not self.drawing_mode:
                    self.drawing_points.clear()
                print(f"Drawing: {'ON' if self.drawing_mode else 'OFF'}")
            
            elif key == ord('c'):
                self.drawing_points.clear()
                print("Canvas cleared")
            
            elif key == ord('s'):
                os.makedirs('saved_drawings', exist_ok=True)
                filename = f"saved_drawings/drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                canvas_save = np.zeros((720, 1280, 3), dtype=np.uint8)
                points = list(self.drawing_points)
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(canvas_save, points[i], points[i+1],
                                self.color, self.brush_size)
                cv2.imwrite(filename, canvas_save)
                print(f"Saved: {filename}")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = DrawInAirSystem()
    system.run()