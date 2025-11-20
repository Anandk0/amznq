from flask import Flask, render_template_string, Response, request, jsonify
import cv2
import requests
import numpy as np
import socket
import threading
import time
import mediapipe as mp

app = Flask(__name__)

# Configuration
ESP32_STREAM_URL = "http://10.30.152.68/stream"
ESP8266_IP = "10.30.152.186"
ESP8266_PORT = 8888

# Global state
current_mode = "manual"  # manual, auto
camera_source = "esp32"  # esp32, local
frame_lock = threading.Lock()
current_frame = None
running = True

# UDP socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# MediaPipe for human detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0,
                    min_detection_confidence=0.45, min_tracking_confidence=0.4)

class CameraManager:
    def __init__(self):
        self.esp32_active = False
        self.local_active = False
        self.current_source = None
        
    def start_esp32_stream(self):
        if self.local_active:
            self.stop_local_camera()
        
        self.esp32_active = True
        self.current_source = "esp32"
        thread = threading.Thread(target=self._esp32_stream_worker, daemon=True)
        thread.start()
        
    def start_local_camera(self):
        if self.esp32_active:
            self.stop_esp32_stream()
            
        self.local_active = True
        self.current_source = "local"
        thread = threading.Thread(target=self._local_camera_worker, daemon=True)
        thread.start()
        
    def stop_esp32_stream(self):
        self.esp32_active = False
        
    def stop_local_camera(self):
        self.local_active = False
        
    def _esp32_stream_worker(self):
        global current_frame
        try:
            stream = requests.get(ESP32_STREAM_URL, stream=True, timeout=5)
            bytes_data = b''
            for chunk in stream.iter_content(chunk_size=2048):
                if not self.esp32_active or not running:
                    break
                bytes_data += chunk
                a = bytes_data.find(b'\\xff\\xd8')
                b = bytes_data.find(b'\\xff\\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with frame_lock:
                            current_frame = frame
        except Exception as e:
            print(f"ESP32 stream error: {e}")
            
    def _local_camera_worker(self):
        global current_frame
        cap = cv2.VideoCapture(0)  # Use default camera
        while self.local_active and running:
            ret, frame = cap.read()
            if ret:
                with frame_lock:
                    current_frame = frame
            time.sleep(0.033)  # ~30 FPS
        cap.release()

camera_manager = CameraManager()

def send_robot_command(command):
    """Send UDP command to ESP8266"""
    try:
        udp_socket.sendto(command.encode(), (ESP8266_IP, ESP8266_PORT))
        print(f"Sent: {command}")
        return True
    except Exception as e:
        print(f"UDP error: {e}")
        return False

def generate_frames():
    """Generate video frames for streaming"""
    global current_frame
    while running:
        with frame_lock:
            frame = current_frame.copy() if current_frame is not None else None
            
        if frame is None:
            time.sleep(0.1)
            continue
            
        # Process frame based on mode
        if current_mode == "auto":
            frame = process_human_detection(frame)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\\r\\n'
                   b'Content-Type: image/jpeg\\r\\n\\r\\n' + buffer.tobytes() + b'\\r\\n')
        time.sleep(0.033)

def process_human_detection(frame):
    """Process frame for human detection and tracking"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        # Draw pose landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Simple tracking logic
        landmarks = results.pose_landmarks.landmark
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        
        if nose.visibility > 0.5:
            h, w = frame.shape[:2]
            x = int(nose.x * w)
            
            # Draw center line and target
            cv2.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 2)
            cv2.circle(frame, (x, int(nose.y * h)), 10, (0, 0, 255), -1)
            
            # Simple tracking commands
            center_threshold = 50
            if x < w//2 - center_threshold:
                send_robot_command("LEFT")
            elif x > w//2 + center_threshold:
                send_robot_command("RIGHT")
            else:
                send_robot_command("FORWARD")
    
    return frame

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Unified Robot Controller</title>
    <style>
        body { font-family: Arial; text-align: center; background: #111; color: #fff; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        .video-container { margin: 20px 0; }
        .controls { margin: 20px 0; }
        button { padding: 10px 20px; margin: 5px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer; }
        .btn-primary { background: #007bff; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .btn-secondary { background: #6c757d; color: white; }
        .direction-pad { display: inline-block; margin: 20px; }
        .direction-pad button { width: 60px; height: 60px; font-size: 20px; }
        .status { margin: 10px 0; padding: 10px; background: #333; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Unified Robot Controller</h1>
        
        <div class="status">
            <p>Mode: <span id="mode">{{ mode }}</span> | Camera: <span id="camera">{{ camera_source }}</span></p>
        </div>
        
        <div class="video-container">
            <img src="/video_feed" style="max-width: 100%; border: 2px solid #333;">
        </div>
        
        <div class="controls">
            <h3>Camera Source</h3>
            <button class="btn-primary" onclick="setCameraSource('esp32')">ESP32 Camera</button>
            <button class="btn-primary" onclick="setCameraSource('local')">Local Camera</button>
        </div>
        
        <div class="controls">
            <h3>Control Mode</h3>
            <button class="btn-success" onclick="setMode('manual')">Manual Control</button>
            <button class="btn-danger" onclick="setMode('auto')">Auto Tracking</button>
        </div>
        
        <div class="controls" id="manual-controls" style="display: {{ 'block' if mode == 'manual' else 'none' }};">
            <h3>Manual Controls</h3>
            <div class="direction-pad">
                <div>
                    <button class="btn-secondary" onmousedown="sendCommand('FORWARD')" onmouseup="sendCommand('STOP')">↑</button>
                </div>
                <div>
                    <button class="btn-secondary" onmousedown="sendCommand('LEFT')" onmouseup="sendCommand('STOP')">←</button>
                    <button class="btn-danger" onclick="sendCommand('STOP')">STOP</button>
                    <button class="btn-secondary" onmousedown="sendCommand('RIGHT')" onmouseup="sendCommand('STOP')">→</button>
                </div>
                <div>
                    <button class="btn-secondary" onmousedown="sendCommand('BACKWARD')" onmouseup="sendCommand('STOP')">↓</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function setMode(mode) {
            fetch('/set_mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: mode})
            }).then(() => location.reload());
        }
        
        function setCameraSource(source) {
            fetch('/set_camera', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({source: source})
            }).then(() => location.reload());
        }
        
        function sendCommand(cmd) {
            fetch('/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: cmd})
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, mode=current_mode, camera_source=camera_source)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global current_mode
    data = request.json
    current_mode = data.get('mode', 'manual')
    if current_mode == 'manual':
        send_robot_command('STOP')
    return jsonify({'status': 'success', 'mode': current_mode})

@app.route('/set_camera', methods=['POST'])
def set_camera():
    global camera_source
    data = request.json
    new_source = data.get('source', 'esp32')
    
    if new_source == 'esp32':
        camera_manager.start_esp32_stream()
        camera_source = 'esp32'
    elif new_source == 'local':
        camera_manager.start_local_camera()
        camera_source = 'local'
        
    return jsonify({'status': 'success', 'camera_source': camera_source})

@app.route('/control', methods=['POST'])
def control():
    if current_mode != 'manual':
        return jsonify({'status': 'error', 'message': 'Not in manual mode'})
    
    data = request.json
    command = data.get('command', '').upper()
    
    if command in ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT', 'STOP']:
        success = send_robot_command(command)
        return jsonify({'status': 'success' if success else 'error'})
    
    return jsonify({'status': 'error', 'message': 'Invalid command'})

if __name__ == '__main__':
    # Start with ESP32 camera by default
    camera_manager.start_esp32_stream()
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        running = False
        camera_manager.stop_esp32_stream()
        camera_manager.stop_local_camera()