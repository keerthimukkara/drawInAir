import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class DrawInAir:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Canvas and drawing settings
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0
        
        # Color palette
        self.colors = [
            (255, 0, 0),      # Blue
            (0, 255, 0),      # Green
            (0, 0, 255),      # Red
            (0, 255, 255),    # Yellow
            (255, 0, 255),    # Magenta
            (255, 255, 0),    # Cyan
            (255, 255, 255),  # White
            (0, 0, 0)         # Eraser (Black)
        ]
        self.current_color = self.colors[2]  # Start with red
        self.brush_size = 5
        
        # Smoothing buffer for coordinates
        self.point_buffer = deque(maxlen=5)
        
        # UI elements
        self.color_palette_height = 80
        self.button_width = 80
        
        # Drawing state
        self.drawing_mode = False
        self.eraser_mode = False
        
    def create_color_palette(self, frame):
        """Create color selection palette at top of frame"""
        height, width = frame.shape[:2]
        
        # Draw background for palette
        cv2.rectangle(frame, (0, 0), (width, self.color_palette_height), 
                     (50, 50, 50), -1)
        
        # Draw color buttons
        for i, color in enumerate(self.colors[:-1]):  # Exclude eraser from palette
            x_start = i * self.button_width + 10
            x_end = x_start + 60
            y_start = 10
            y_end = 70
            
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), 
                         color, -1)
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), 
                         (255, 255, 255), 2)
            
            # Highlight current color
            if color == self.current_color:
                cv2.rectangle(frame, (x_start-3, y_start-3), (x_end+3, y_end+3), 
                             (0, 255, 0), 3)
        
        # Draw eraser button
        eraser_x = len(self.colors[:-1]) * self.button_width + 10
        cv2.rectangle(frame, (eraser_x, 10), (eraser_x + 60, 70), 
                     (200, 200, 200), -1)
        cv2.putText(frame, "Erase", (eraser_x + 5, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        if self.eraser_mode:
            cv2.rectangle(frame, (eraser_x-3, 7), (eraser_x + 63, 73), 
                         (0, 255, 0), 3)
        
        # Draw clear button
        clear_x = eraser_x + 80
        cv2.rectangle(frame, (clear_x, 10), (clear_x + 60, 70), 
                     (100, 100, 100), -1)
        cv2.putText(frame, "Clear", (clear_x + 5, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def count_fingers(self, landmarks, hand_label):
        """Count number of fingers raised"""
        fingers = []
        
        # Thumb
        if hand_label == "Right":
            if landmarks[4][0] < landmarks[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if landmarks[4][0] > landmarks[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # Other fingers
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip][1] < landmarks[pip][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def smooth_coordinates(self, x, y):
        """Apply smoothing to reduce jitter"""
        self.point_buffer.append((x, y))
        
        if len(self.point_buffer) < 2:
            return x, y
        
        # Calculate moving average
        avg_x = int(np.mean([p[0] for p in self.point_buffer]))
        avg_y = int(np.mean([p[1] for p in self.point_buffer]))
        
        return avg_x, avg_y
    
    def detect_gesture(self, fingers):
        """Detect drawing or selection gestures"""
        # Index finger only = drawing mode
        if fingers == [0, 1, 0, 0, 0]:
            return "draw"
        # Index and middle finger = selection mode
        elif fingers == [0, 1, 1, 0, 0]:
            return "select"
        # Fist = erase mode
        elif fingers == [0, 0, 0, 0, 0]:
            return "erase"
        else:
            return "none"
    
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create canvas
        ret, frame = cap.read()
        if ret:
            self.canvas = np.zeros_like(frame)
        
        print("Draw in Air - Controls:")
        print("- Index finger only: Draw")
        print("- Index + Middle finger: Select color")
        print("- Fist: Erase mode")
        print("- 'c': Clear canvas")
        print("- 's': Save drawing")
        print("- 'q': Quit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Create UI overlay
            frame = self.create_color_palette(frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                       results.multi_handedness):
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append((int(lm.x * width), int(lm.y * height)))
                    
                    # Get hand label
                    hand_label = handedness.classification[0].label
                    
                    # Count fingers
                    fingers = self.count_fingers(landmarks, hand_label)
                    
                    # Detect gesture
                    gesture = self.detect_gesture(fingers)
                    
                    # Get index fingertip position (landmark 8)
                    x, y = landmarks[8]
                    
                    # Apply smoothing
                    x, y = self.smooth_coordinates(x, y)
                    
                    # Display gesture
                    cv2.putText(frame, f"Gesture: {gesture}", (10, height - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if gesture == "draw":
                        # Draw on canvas
                        if self.prev_x == 0 and self.prev_y == 0:
                            self.prev_x, self.prev_y = x, y
                        
                        if y > self.color_palette_height:  # Don't draw on palette
                            color = (0, 0, 0) if self.eraser_mode else self.current_color
                            thickness = self.brush_size * 3 if self.eraser_mode else self.brush_size
                            
                            cv2.line(self.canvas, (self.prev_x, self.prev_y), 
                                   (x, y), color, thickness)
                            cv2.line(frame, (self.prev_x, self.prev_y), 
                                   (x, y), color, thickness)
                        
                        self.prev_x, self.prev_y = x, y
                        
                    elif gesture == "select":
                        # Color selection mode
                        if y < self.color_palette_height:
                            # Check which color button is selected
                            for i, color in enumerate(self.colors[:-1]):
                                x_start = i * self.button_width + 10
                                x_end = x_start + 60
                                if x_start < x < x_end:
                                    self.current_color = color
                                    self.eraser_mode = False
                            
                            # Check eraser button
                            eraser_x = len(self.colors[:-1]) * self.button_width + 10
                            if eraser_x < x < eraser_x + 60:
                                self.eraser_mode = True
                            
                            # Check clear button
                            clear_x = eraser_x + 80
                            if clear_x < x < clear_x + 60:
                                self.canvas = np.zeros_like(frame)
                        
                        self.prev_x, self.prev_y = 0, 0
                        
                    else:
                        self.prev_x, self.prev_y = 0, 0
            else:
                self.prev_x, self.prev_y = 0, 0
            
            # Merge canvas with frame
            canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
            
            combined = cv2.add(frame_bg, canvas_fg)
            
            # Display instructions
            cv2.putText(combined, "Press 's' to save, 'c' to clear, 'q' to quit", 
                       (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            cv2.imshow("Draw in Air", combined)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros_like(frame)
            elif key == ord('s'):
                cv2.imwrite('drawing.png', self.canvas)
                print("Drawing saved as 'drawing.png'")
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(20, self.brush_size + 1)
                print(f"Brush size: {self.brush_size}")
            elif key == ord('-') or key == ord('_'):
                self.brush_size = max(1, self.brush_size - 1)
                print(f"Brush size: {self.brush_size}")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DrawInAir()
    app.run()