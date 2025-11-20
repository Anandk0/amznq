# main.py
from flask import Flask, render_template_string, Response
import threading
import time
import socket
import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2

# ===== CONFIG =====
ESP8266_IP = "10.30.152.186"  # robot UDP IP
ESP8266_PORT = 8888

FRAME_W = 400
FRAME_H = 300

# timings & tuning
CMD_MIN_INTERVAL = 0.12     # seconds
SCAN_WAIT = 1.0             # seconds: wait after a single-turn for fresh frames
TURN_STEP_MS = 350          # ESP single-turn duration (ms)
MAX_SCAN_STEPS = 12         # how many steps to try scanning 360
SMOOTH_WINDOW = 4

# visibility thresholds
SHOULDER_VIS = 0.4
KNEE_VIS = 0.3
ANKLE_VIS = 0.3
NOSE_VIS = 0.3

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

# mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0,
                    min_detection_confidence=0.45, min_tracking_confidence=0.4)

# ===== Pi Camera 2 setup =====
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
print("[PICAM2] Camera started")

# ===== helpers: pose -> legs/center detection =====
def is_legs_visible(lm):
    left_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
    v = 0
    if left_knee.visibility >= KNEE_VIS or right_knee.visibility >= KNEE_VIS:
        v += 1
    if left_ankle.visibility >= ANKLE_VIS or right_ankle.visibility >= ANKLE_VIS:
        v += 1
    return v >= 1

def compute_center_x_from_pose(lm, W):
    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = lm[mp_pose.PoseLandmark.NOSE]
    if ls.visibility >= SHOULDER_VIS and rs.visibility >= SHOULDER_VIS:
        return int(((ls.x + rs.x)/2.0) * W)
    if nose.visibility >= NOSE_VIS:
        return int(nose.x * W)
    return None

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

# ===== tracking state machine (single-turn strategy + scan) =====
def tracking_loop():
    global output_frame, current_mode
    smooth_buf = []
    scan_step_count = 0

    while running:
        frame = picam2.capture_array()
        if frame is None:
            time.sleep(0.05)
            continue

        img = cv2.resize(frame, (FRAME_W, FRAME_H))
        H, W = img.shape[:2]

        with mode_lock:
            mode_now = current_mode

        if mode_now == "AUTO":
            # pose detection
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            detected = False
            legs = False
            cx = None

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # leg detection: use knees/ankles visibility
                left_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE]
                right_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
                left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                legs = ((left_knee.visibility >= KNEE_VIS) or (right_knee.visibility >= KNEE_VIS) or
                        (left_ankle.visibility >= ANKLE_VIS) or (right_ankle.visibility >= ANKLE_VIS))

                cx = compute_center_x_from_pose(lm, W)
                if cx is not None:
                    smooth_buf.append(cx)
                    if len(smooth_buf) > SMOOTH_WINDOW:
                        smooth_buf.pop(0)
                    cx = int(sum(smooth_buf) / len(smooth_buf))
                    detected = True

            # ultrasonic control: enable only when legs visible
            if legs:
                send_udp_once("ULTRASONIC_ON")
            else:
                send_udp_once("ULTRASONIC_OFF")

            if detected and legs:
                # reset scanning
                scan_step_count = 0
                # zones
                z1 = W * 0.25; z2 = W * 0.40; z3 = W * 0.60; z4 = W * 0.75
                if cx < z1:
                    action = "TURN_RIGHT_ONCE"
                elif cx < z2:
                    action = "TURN_RIGHT_ONCE"
                elif cx < z3:
                    action = "FORWARD"
                elif cx < z4:
                    action = "TURN_LEFT_ONCE"
                else:
                    action = "TURN_LEFT_ONCE"

                if action == "FORWARD":
                    send_udp_if_changed("FORWARD")
                else:
                    # send single-turn and wait for SCAN_WAIT so ESP has time to turn & camera to update
                    send_udp_once(action)
                    t0 = time.time()
                    while time.time() - t0 < SCAN_WAIT:
                        time.sleep(0.05)
                    # after wait, loop will re-evaluate on fresh frames
            else:
                # SCAN mode: legs not visible -> perform single-turn scanning steps
                if scan_step_count < MAX_SCAN_STEPS:
                    send_udp_once("TURN_LEFT_ONCE")
                    scan_step_count += 1
                    t0 = time.time()
                    while time.time() - t0 < SCAN_WAIT:
                        time.sleep(0.05)
                    # next iteration will check again for detection
                else:
                    # completed scan -> STOP and pause
                    send_udp_if_changed("STOP")
                    scan_step_count = 0
                    time.sleep(0.5)

        # update output frame
        with frame_lock:
            output_frame = img.copy()

        time.sleep(0.005)  # cooperative

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

  <div id="rc" style="display:{{ 'block' if mode=='MANUAL' else 'none' }};">
    <h3>Remote Control</h3>
    <div class="rc">
      <button class="dir" onmousedown="send('FORWARD')" onmouseup="send('STOP')">▲</button>
      <button class="dir" onmousedown="send('LEFT')" onmouseup="send('STOP')">◀</button>
      <button class="dir" onmousedown="send('STOP')" onmouseup="send('STOP')">■</button>
      <button class="dir" onmousedown="send('RIGHT')" onmouseup="send('STOP')">▶</button>
      <button class="dir" onmousedown="send('BACKWARD')" onmouseup="send('STOP')">▼</button>
    </div>
    <br/>
    <button class="btn" onclick="testBackward()">TEST BACKWARD (0.5s) </button>
  </div>

<script>
function setMode(m){
  fetch('/set_mode/' + m).then(()=> location.reload());
}
function send(cmd){
  fetch('/control/' + cmd);
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
    return render_template_string(HTML_PAGE, mode=mode)

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
