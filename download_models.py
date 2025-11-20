#!/usr/bin/env python3
"""Download MobileNet-SSD model files"""

import urllib.request
import os

MODEL_URLS = {
    "MobileNetSSD_deploy.prototxt": "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel": "https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.caffemodel"
}

def download_file(url, filename):
    """Download file with progress"""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"[OK] Downloaded {filename}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")
        return False

if __name__ == "__main__":
    print("Downloading MobileNet-SSD model files...")
    
    for filename, url in MODEL_URLS.items():
        if os.path.exists(filename):
            print(f"[OK] {filename} already exists")
            continue
        
        if filename.endswith('.caffemodel'):
            print(f"Note: {filename} is large (~23MB) and may take time to download")
        
        download_file(url, filename)
    
    print("\nModel download complete!")