from flask import Flask, render_template, request, jsonify
import socket

app = Flask(__name__)

# Robot Configuration
ESP8266_IP = "10.30.152.186"
ESP8266_PORT = 8888

def send_command(command):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(command.encode(), (ESP8266_IP, ESP8266_PORT))
        sock.close()
        return True
    except:
        return False

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Robot Controller</title>
    <style>
        body { font-family: Arial; text-align: center; margin: 50px; }
        .controls { display: inline-block; margin: 20px; }
        button { width: 80px; height: 80px; font-size: 20px; margin: 5px; }
        .row { display: block; }
    </style>
</head>
<body>
    <h1>Robot Web Controller</h1>
    <div class="controls">
        <div class="row">
            <button onclick="sendCmd('FORWARD')">↑</button>
        </div>
        <div class="row">
            <button onclick="sendCmd('LEFT')">←</button>
            <button onclick="sendCmd('STOP')">STOP</button>
            <button onclick="sendCmd('RIGHT')">→</button>
        </div>
    </div>
    
    <script>
        function sendCmd(cmd) {
            fetch('/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: cmd})
            });
        }
    </script>
</body>
</html>
    '''

@app.route('/control', methods=['POST'])
def control():
    data = request.get_json()
    command = data.get('command')
    
    if send_command(command):
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)