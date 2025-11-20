from flask import Flask, render_template, request, jsonify, Response
import cv2
import requests
import numpy as np
import socket
import threading
import time
from threading import Thread

app = Flask(__name__)

# Configuration
ESP32_STREAM_URL = "http://10.30.152.68/stream"
ESP8266_IP = "10.30.152.186"
ESP8266_PORT = 8888

# Global state
current_mode = "manual"  # manual, human_follow, object_follow
robot_status = "stopped"
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

class VideoStream:
    def __init__(self):
        self.frame = None
        self.running = True
        self.thread = Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()
    
    def update(self):
        try:
            stream = requests.get(ESP32_STREAM_URL, stream=True, timeout=5)
            bytes_data = b''
            for chunk in stream.iter_content(chunk_size=2048):
                if not self.running:
                    break
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.frame = frame
        except Exception as e:
            print(f"Stream error: {e}")
    
    def get_frame(self):
        if self.frame is not None:
            ret, buffer = cv2.imencode('.jpg', self.frame)
            return buffer.tobytes()
        return None

video_stream = VideoStream()

def send_robot_command(command):
    """Send UDP command to ESP8266"""
    try:
        udp_socket.sendto(command.encode(), (ESP8266_IP, ESP8266_PORT))
        return True
    except Exception as e:
        print(f"UDP error: {e}")
        return False

def send_http_command(command):
    """Send HTTP command to ESP8266"""
    try:
        response = requests.get(f"http://{ESP8266_IP}/?State={command}", timeout=2)
        return response.status_code == 200
    except Exception as e:
        print(f"HTTP error: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = video_stream.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control():
    global current_mode, robot_status
    
    data = request.json
    command = data.get('command')
    
    if command == 'set_mode':
        current_mode = data.get('mode', 'manual')
        if current_mode != 'human_follow':
            send_robot_command('STOP')
        return jsonify({'status': 'success', 'mode': current_mode})
    
    elif command in ['forward', 'backward', 'left', 'right', 'stop']:
        if current_mode == 'manual':
            cmd_map = {
                'forward': 'F', 'backward': 'B', 
                'left': 'L', 'right': 'R', 'stop': 'S'
            }
            success = send_http_command(cmd_map[command])
            robot_status = command
            return jsonify({'status': 'success' if success else 'error'})
    
    return jsonify({'status': 'error', 'message': 'Invalid command'})

@app.route('/status')
def status():
    return jsonify({
        'mode': current_mode,
        'status': robot_status,
        'esp8266_ip': ESP8266_IP
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)