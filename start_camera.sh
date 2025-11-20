#!/bin/bash
# Start rpicam MJPEG stream on port 8080
rpicam-vid -t 0 --width 640 --height 480 --framerate 30 --inline --listen -o tcp://0.0.0.0:8080
