# main.py
from flask import Flask, render_template_string, Response
import threading
import time
import socket
import cv2
import numpy as np
import sys

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("ERROR: Install tensorflow")
        sys.exit(1)

# ===== CONFIG =====
ESP8266_IP = "10.30.152.186"  # robot UDP IP
ESP8266_PORT = 8888

FRAME_W = 320
FRAME_H = 240

# TFLite model
MODEL_PATH = "ei-model.tflite"
CONFIDENCE_THRESHOLD = 0.3

# Speed control variables
forward_speed = 150  # Default forward speed (0-255)
turn_speed = 80      # Default turn speed (0-255)

# Target tracking variables
DEBOUNCE_FRAMES = 5
SEARCH_FRAMES = 15
frames_without_detection = 0
last_known_x = None
target_locked = False

CMD_MIN_INTERVAL = 0.08
FRAME_SKIP = 2

# ===== GLOBALS =====
app = Flask(__name__)
frame_lock = threading.Lock()
output_frame = None
running = True

mode_lock = threading.Lock()
current_mode = "MANUAL"  # MANUAL or AUTO

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
last_send_time = 0.0
last_sent_cmd = None

# TFLite setup
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_height = input_shape[1]
input_width = input_shape[2]
input_channels = input_shape[3]
input_index = input_details[0]['index']
is_fomo = len(output_details) == 1 and len(output_details[0]['shape']) == 4

# ===== Pi Camera via rpicam-vid TCP stream =====
import subprocess
import os

# Start rpicam-vid in background
print("[CAMERA] Starting rpicam-vid...")
rpicam_process = subprocess.Popen([
    'rpicam-vid', '-t', '0', '--width', '640', '--height', '480',
    '--framerate', '20', '--inline', '--listen', '--nopreview', '-o', 'tcp://0.0.0.0:8080'
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

time.sleep(3)  # Wait for camera to start

# Connect to TCP stream
camera = cv2.VideoCapture('tcp://127.0.0.1:8080')
if camera.isOpened():
    print("[CAMERA] Connected to rpicam-vid stream")
else:
    print("[ERROR] Cannot connect to rpicam stream")

def send_burst(command, times, delay=0.05):
    for _ in range(times):
        sock.sendto(command.encode(), (ESP8266_IP, ESP8266_PORT))
        time.sleep(delay)
    sock.sendto("STOP".encode(), (ESP8266_IP, ESP8266_PORT))

def send_speed_command(direction):
    """Send direction command with current speed settings"""
    global forward_speed, turn_speed
    if direction == "FORWARD":
        cmd = f"FORWARD:{forward_speed}"
    elif direction in ["LEFT", "RIGHT"]:
        cmd = f"{direction}:{turn_speed}"
    else:
        cmd = direction
    sock.sendto(cmd.encode(), (ESP8266_IP, ESP8266_PORT))

# ===== UDP sending helpers (rate-limited, send-on-change) =====
def send_udp_once(cmd):
    global last_send_time, last_sent_cmd
    try:
        sock.sendto(cmd.encode(), (ESP8266_IP, ESP8266_PORT))
        last_send_time = time.time()
        last_sent_cmd = cmd
        print("[UDP] ->", cmd)
    except Exception as e:
        print("[UDP] send error:", e)

def send_udp_if_changed(cmd):
    now = time.time()
    global last_send_time, last_sent_cmd
    if cmd == last_sent_cmd and (now - last_send_time) < CMD_MIN_INTERVAL:
        return
    if (now - last_send_time) < CMD_MIN_INTERVAL and cmd != last_sent_cmd:
        # respect min interval
        return
    send_udp_once(cmd)

def tracking_loop():
    global output_frame, current_mode, frames_without_detection, last_known_x, target_locked
    frame_count = 0

    while running:
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        img = cv2.resize(frame, (FRAME_W, FRAME_H))
        H, W = img.shape[:2]

        with mode_lock:
            mode_now = current_mode

        if mode_now == "AUTO":
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                with frame_lock:
                    output_frame = img.copy()
                continue

            # TFLite inference
            img_resized = cv2.resize(img, (input_width, input_height), interpolation=cv2.INTER_NEAREST)
            if input_channels == 1:
                img_processed = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                img_processed = np.expand_dims(img_processed, axis=-1)
            else:
                img_processed = img_resized[:, :, ::-1]
            
            input_data = np.expand_dims(img_processed, axis=0)
            if input_details[0]['dtype'] == np.float32:
                input_data = np.float32(input_data) / 255.0
            elif input_details[0]['dtype'] == np.int8:
                input_data = (input_data.astype(np.int16) - 128).astype(np.int8)
            
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()

            detected = False
            center_x = 0

            if is_fomo:
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                if output_details[0]['dtype'] == np.int8:
                    output_data = (output_data.astype(np.float32) + 128) / 255.0
                if output_data.shape[2] > 1:
                    output_data = output_data[:, :, 1:]
                
                max_idx = np.argmax(output_data)
                max_y, max_x, max_c = np.unravel_index(max_idx, output_data.shape)
                max_val = output_data[max_y, max_x, max_c]

                if max_val > CONFIDENCE_THRESHOLD:
                    detected = True
                    grid_h, grid_w, _ = output_data.shape
                    center_x = int((max_x + 0.5) * (W / grid_w))
                    center_y = int((max_y + 0.5) * (H / grid_h))
                    
                    # Update tracking
                    last_known_x = center_x
                    frames_without_detection = 0
                    target_locked = True
                    
                    cv2.circle(img, (center_x, center_y), 12, (0, 255, 0), 2)
            else:
                boxes = interpreter.get_tensor(output_details[0]['index'])[0]
                scores = interpreter.get_tensor(output_details[2]['index'])[0]
                best_idx = np.argmax(scores)
                if scores[best_idx] > CONFIDENCE_THRESHOLD:
                    ymin, xmin, ymax, xmax = boxes[best_idx]
                    center_x = int((xmin + xmax) / 2 * W)
                    center_y = int((ymin + ymax) / 2 * H)
                    detected = True
                    
                    # Update tracking
                    last_known_x = center_x
                    frames_without_detection = 0
                    target_locked = True
                    
                    cv2.rectangle(img, (int(xmin*W), int(ymin*H)), (int(xmax*W), int(ymax*H)), (0,255,0), 2)

            # Target tracking logic
            if not detected:
                frames_without_detection += 1
            
            # Use last known position if recently lost
            tracking_x = center_x if detected else last_known_x
            
            # Control logic with tracking
            if detected or (target_locked and frames_without_detection < DEBOUNCE_FRAMES):
                z1 = W * 0.25; z2 = W * 0.40; z3 = W * 0.60; z4 = W * 0.75
                if tracking_x and tracking_x < z1:
                    send_speed_command("LEFT")
                    time.sleep(0.1)
                    send_udp_once("STOP")
                elif tracking_x and tracking_x < z2:
                    send_speed_command("LEFT")
                    time.sleep(0.05)
                    send_udp_once("STOP")
                elif tracking_x and tracking_x <= z3:
                    send_speed_command("FORWARD")
                elif tracking_x and tracking_x < z4:
                    send_speed_command("RIGHT")
                    time.sleep(0.05)
                    send_udp_once("STOP")
                else:
                    send_speed_command("RIGHT")
                    time.sleep(0.1)
                    send_udp_once("STOP")
            else:
                if frames_without_detection > SEARCH_FRAMES:
                    target_locked = False
                    last_known_x = None
                send_udp_if_changed("STOP")

        with frame_lock:
            output_frame = img.copy()
        time.sleep(0.005)

# ===== Flask endpoints and video generator =====
HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Robot Control</title>
  <style>
    body{background:#111;color:#eee;font-family:Arial;text-align:center}
    .video{border:4px solid #333;display:inline-block;margin:12px}
    .btn{padding:12px 18px;margin:6px;font-size:16px;border-radius:6px;border:none;cursor:pointer}
    .btn-mode{background:#1e90ff;color:#fff;width:80%}
    .btn-danger{background:#dc3545;color:#fff}
    .rc{display:flex;gap:10px;justify-content:center;align-items:center;flex-wrap:wrap}
    .dir{width:60px;height:60px;border-radius:6px;background:#333;color:#fff;font-size:20px}
    .speed-control{margin:20px auto;max-width:400px;padding:20px;background:#222;border-radius:10px}
    .slider-container{margin:15px 0;text-align:left}
    .slider{width:100%;height:25px;border-radius:5px;background:#444;outline:none;-webkit-appearance:none}
    .slider::-webkit-slider-thumb{appearance:none;width:25px;height:25px;border-radius:50%;background:#1e90ff;cursor:pointer}
    .slider::-moz-range-thumb{width:25px;height:25px;border-radius:50%;background:#1e90ff;cursor:pointer;border:none}
  </style>
</head>
<body>
  <h1>Human Follower Robot</h1>
  <div class="video">
    <img src="{{ url_for('video_feed') }}" width="400" />
  </div>
  <h3>Mode: <span id="mode">{{ mode }}</span></h3>
  <button class="btn btn-mode" onclick="setMode('AUTO')">ENABLE AUTO TRACKING</button>
  <button class="btn btn-mode btn-danger" onclick="setMode('MANUAL')">SWITCH TO MANUAL</button>

  <div class="speed-control">
    <h3>Speed Control</h3>
    <div class="slider-container">
      <label>Forward Speed: <span id="forward-value">{{ forward_speed }}</span></label>
      <input type="range" min="50" max="255" value="{{ forward_speed }}" class="slider" id="forward-speed" oninput="updateSpeed('forward', this.value)">
    </div>
    <div class="slider-container">
      <label>Turn Speed: <span id="turn-value">{{ turn_speed }}</span></label>
      <input type="range" min="30" max="150" value="{{ turn_speed }}" class="slider" id="turn-speed" oninput="updateSpeed('turn', this.value)">
    </div>
  </div>

  <div id="rc" style="display:{{ 'block' if mode=='MANUAL' else 'none' }};">
    <h3>Remote Control</h3>
    <div class="rc">
      <button class="dir" onmousedown="sendWithSpeed('FORWARD')" onmouseup="send('STOP')">▲</button>
      <button class="dir" onmousedown="sendWithSpeed('LEFT')" onmouseup="send('STOP')">◀</button>
      <button class="dir" onmousedown="send('STOP')" onmouseup="send('STOP')">■</button>
      <button class="dir" onmousedown="sendWithSpeed('RIGHT')" onmouseup="send('STOP')">▶</button>
      <button class="dir" onmousedown="send('BACKWARD')" onmouseup="send('STOP')">▼</button>
    </div>
    <br/>
    <button class="btn" onclick="testBackward()">TEST BACKWARD (0.5s) </button>
  </div>

<script>
let forwardSpeed = {{ forward_speed }};
let turnSpeed = {{ turn_speed }};

function setMode(m){
  fetch('/set_mode/' + m).then(()=> location.reload());
}
function send(cmd){
  fetch('/control/' + cmd);
}
function sendWithSpeed(direction){
  let speed = direction === 'FORWARD' ? forwardSpeed : turnSpeed;
  fetch('/control/' + direction + ':' + speed);
}
function updateSpeed(type, value){
  if(type === 'forward'){
    forwardSpeed = value;
    document.getElementById('forward-value').textContent = value;
    fetch('/set_speed/forward/' + value);
  } else {
    turnSpeed = value;
    document.getElementById('turn-value').textContent = value;
    fetch('/set_speed/turn/' + value);
  }
}
function testBackward(){
  send('BACKWARD');
  setTimeout(()=> send('STOP'), 500);
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    with mode_lock:
        mode = current_mode
    return render_template_string(HTML_PAGE, mode=mode, forward_speed=forward_speed, turn_speed=turn_speed)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<mode>')
def set_mode(mode):
    global current_mode
    with mode_lock:
        current_mode = mode
    # ensure robot safe state on mode switch
    send_udp_once("STOP")
    return "OK"

@app.route('/control/<cmd>')
def control(cmd):
    with mode_lock:
        if current_mode == "MANUAL":
            # direct immediate command (single send). The ESP implements safety timeout.
            send_udp_once(cmd.upper())
    return "OK"

@app.route('/set_speed/<speed_type>/<int:value>')
def set_speed(speed_type, value):
    global forward_speed, turn_speed
    if speed_type == 'forward':
        forward_speed = max(50, min(255, value))
    elif speed_type == 'turn':
        turn_speed = max(30, min(150, value))
    return "OK"

def generate():
    while running:
        with frame_lock:
            frame = None if output_frame is None else output_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.02)

# ===== app start =====
if __name__ == '__main__':
    t = threading.Thread(target=tracking_loop, daemon=True)
    t.start()
    # start Flask
    print("Starting Flask on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
