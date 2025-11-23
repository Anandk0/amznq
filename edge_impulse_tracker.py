import cv2
import time
import socket
import numpy as np
import sys
import subprocess
import tempfile
import os

# ===== ROBUST IMPORT (PC vs PI) =====
try:
    import tflite_runtime.interpreter as tflite
    print("[INIT] Using tflite_runtime (Raspberry Pi mode)")
except ImportError:
    try:
        import tensorflow.lite as tflite
        print("[INIT] Using full tensorflow.lite (PC mode)")
    except ImportError:
        print("\nCRITICAL ERROR: Missing TensorFlow libraries.")
        print("Please run: pip install tensorflow")
        sys.exit(1)

# ===== CONFIGURATION =====
ESP8266_IP = "10.30.152.186"                      # CHECK IP
ESP8266_PORT = 8888

MODEL_PATH = "ei-model.tflite" 
CONFIDENCE_THRESHOLD = 0.6

# ===== SETUP UDP =====
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"Targeting Robot at {ESP8266_IP}:{ESP8266_PORT}")

# ===== SETUP TFLITE =====
print(f"[INIT] Loading {MODEL_PATH}...")
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_height = input_shape[1]
    input_width = input_shape[2]
    input_channels = input_shape[3]
    input_index = input_details[0]['index']
    
    # Determine Model Type
    is_fomo = False
    if len(output_details) == 1 and len(output_details[0]['shape']) == 4:
        is_fomo = True
        print("[INFO] FOMO Model Detected (Heatmap mode)")
    else:
        print("[INFO] Standard SSD Model Detected (Bounding Box mode)")

    print(f"Model expects input: {input_width}x{input_height} with {input_channels} channels")
    
except Exception as e:
    print(f"\nError loading model: {e}")
    input("Press Enter to exit...")
    sys.exit(1)

# ===== HELPER FUNCTIONS =====
def send_burst(command, times, delay=0.05):
    for _ in range(times):
        sock.sendto(command.encode(), (ESP8266_IP, ESP8266_PORT))
        time.sleep(delay)
    sock.sendto("STOP".encode(), (ESP8266_IP, ESP8266_PORT))

# ===== PI CAMERA SETUP =====
print("[INIT] Setting up Pi Camera with rpicam-still...")
tmp_img = "/tmp/capture.jpg"

# Test camera
try:
    subprocess.run(["rpicam-still", "-o", tmp_img, "--timeout", "1", "--width", "640", "--height", "480", "--nopreview"], 
                   check=True, capture_output=True)
    if os.path.exists(tmp_img):
        os.remove(tmp_img)
        print("[INFO] Pi Camera working with rpicam-still")
    else:
        raise Exception("No image captured")
except Exception as e:
    print(f"[ERROR] Camera test failed: {e}")
    sys.exit(1)

# ===== MAIN LOOP =====
print("Starting Edge Impulse Tracker...")

try:
    while True:
        # Capture frame from Pi camera using rpicam-still
        try:
            subprocess.run(["rpicam-still", "-o", tmp_img, "--timeout", "1", "--width", "640", "--height", "480", "--nopreview"], 
                          check=True, capture_output=True, timeout=3)
            image_bgr = cv2.imread(tmp_img)
            if image_bgr is None:
                continue
        except Exception:
            continue

        H_orig, W_orig = image_bgr.shape[:2]

        # 1. Preprocessing
        img_resized = cv2.resize(image_bgr, (input_width, input_height))
        
        if input_channels == 1:
            img_processed = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_processed = np.expand_dims(img_processed, axis=-1)
        else:
            img_processed = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        input_data = np.expand_dims(img_processed, axis=0)

        # Normalize based on model type
        input_type = input_details[0]['dtype']
        if input_type == np.float32:
            input_data = (np.float32(input_data) / 255.0)
        elif input_type == np.int8:
            input_data = (input_data.astype(np.int16) - 128).astype(np.int8)
        else:
            input_data = input_data.astype(input_type)

        # 2. Inference
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()

        detected = False
        center_x = 0
        max_score = 0

        # 3. Output Decoding
        if is_fomo:
            # === FOMO LOGIC (Optimized) ===
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            
            # Convert int8 to float
            if output_details[0]['dtype'] == np.int8:
                output_data = (output_data.astype(np.float32) + 128) / 255.0
            
            # Check dimensions
            num_classes = output_data.shape[2]
            
            if num_classes > 1:
                # !!! CRITICAL FIX: IGNORE CLASS 0 (BACKGROUND) !!!
                # Slice the array to only include classes 1, 2, etc.
                search_area = output_data[:, :, 1:]
            else:
                # If there's only 1 class, we have to use it
                search_area = output_data

            # Find max in the OBJECT classes only
            max_idx = np.argmax(search_area)
            max_y, max_x, max_c_rel = np.unravel_index(max_idx, search_area.shape)
            max_val = search_area[max_y, max_x, max_c_rel]

            if max_val > CONFIDENCE_THRESHOLD:
                detected = True
                # Scale Grid -> Pixel
                grid_h, grid_w, _ = output_data.shape
                center_x = int((max_x + 0.5) * (W_orig / grid_w))
                center_y = int((max_y + 0.5) * (H_orig / grid_h))
                
                # Correct class index (add 1 because we skipped background)
                real_class_id = max_c_rel + 1 if num_classes > 1 else max_c_rel
                
                # Draw
                cv2.circle(image_bgr, (center_x, center_y), 15, (0, 255, 0), 2)
                cv2.circle(image_bgr, (center_x, center_y), 3, (0, 255, 0), -1)
                label_text = f"Class {real_class_id}: {max_val:.2f}"
                cv2.putText(image_bgr, label_text, (center_x+10, center_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        else:
            # === SSD LOGIC (Legacy) ===
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            classes = interpreter.get_tensor(output_details[1]['index'])[0]
            scores = interpreter.get_tensor(output_details[2]['index'])[0]

            for i in range(len(scores)):
                if scores[i] > CONFIDENCE_THRESHOLD:
                    ymin, xmin, ymax, xmax = boxes[i]
                    left = int(xmin * W_orig)
                    right = int(xmax * W_orig)
                    top = int(ymin * H_orig)
                    bottom = int(ymax * H_orig)
                    
                    center_x = int((left + right) / 2)
                    center_y = int((top + bottom) / 2)
                    detected = True
                    
                    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
                    break # Track first

        # 4. Zone Logic
        ZONE_FAR_LEFT = W_orig * 0.25
        ZONE_SLIGHT_LEFT = W_orig * 0.40
        ZONE_SLIGHT_RIGHT = W_orig * 0.60
        ZONE_FAR_RIGHT = W_orig * 0.75
        
        status = "SEARCHING"

        if detected:
            if center_x < ZONE_FAR_LEFT:
                status = "HARD LEFT"
                send_burst("LEFT", 5)
            elif center_x < ZONE_SLIGHT_LEFT:
                status = "SLIGHT LEFT"
                send_burst("LEFT", 2)
            elif center_x >= ZONE_SLIGHT_LEFT and center_x <= ZONE_SLIGHT_RIGHT:
                status = "FORWARD"
                sock.sendto("FORWARD".encode(), (ESP8266_IP, ESP8266_PORT))
            elif center_x > ZONE_SLIGHT_RIGHT and center_x < ZONE_FAR_RIGHT:
                status = "SLIGHT RIGHT"
                send_burst("RIGHT", 2)
            elif center_x >= ZONE_FAR_RIGHT:
                status = "HARD RIGHT"
                send_burst("RIGHT", 5)
        else:
            sock.sendto("STOP".encode(), (ESP8266_IP, ESP8266_PORT))

        # Visuals
        cv2.line(image_bgr, (int(ZONE_FAR_LEFT), 0), (int(ZONE_FAR_LEFT), H_orig), (0,0,255), 1)
        cv2.line(image_bgr, (int(ZONE_SLIGHT_LEFT), 0), (int(ZONE_SLIGHT_LEFT), H_orig), (255,255,0), 1)
        cv2.line(image_bgr, (int(ZONE_SLIGHT_RIGHT), 0), (int(ZONE_SLIGHT_RIGHT), H_orig), (255,255,0), 1)
        cv2.line(image_bgr, (int(ZONE_FAR_RIGHT), 0), (int(ZONE_FAR_RIGHT), H_orig), (0,0,255), 1)
        
        cv2.putText(image_bgr, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('EI Custom Tracker', image_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    if os.path.exists(tmp_img):
        os.remove(tmp_img)
    sock.sendto("STOP".encode(), (ESP8266_IP, ESP8266_PORT))
    sock.close()
    cv2.destroyAllWindows()
