import os
import mss
import cv2
import numpy as np
import time
import glob
from ultralytics import YOLO
from openpyxl import Workbook

# Define paths
home_dir = os.path.expanduser("~")
save_path = os.path.join(home_dir, "yolo_detection")
screenshots_path = os.path.join(save_path, "screenshots")
detect_path = os.path.join(save_path, "runs", "detect")
os.makedirs(screenshots_path, exist_ok=True)
os.makedirs(detect_path, exist_ok=True)

# Define pattern classes
classes = ['Head and shoulders bottom', 'Head and shoulders top', 'M_Head', 'StockLine', 'Triangle', 'W_Bottom']

# Load YOLOv8 model
model = YOLO('best.pt')

# Define screen capture region
monitor = {"top": 0, "left": 683, "width": 683, "height": 768}

# Create Excel file
excel_file = os.path.join(save_path, "classification_results.xlsx")
wb = Workbook()
ws = wb.active
ws.append(["Timestamp", "Predicted Image Path", "Label"])

# Initialize video writer
video_path = os.path.join(save_path, "annotated_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 0.5
video_writer = None

# Start capturing
with mss.mss() as sct:
    monitor = sct.monitors[2]  # Toàn bộ màn hình
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < 60:
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        image_name = f"predicted_images_{timestamp}_{frame_count}.png"
        image_path = os.path.join(screenshots_path, image_name)
        cv2.imwrite(image_path, img)

        results = model(image_path, save=True)
        predict_path = results[0].save_dir if results else None
        annotated_images = sorted(glob.glob(os.path.join(predict_path, "*.jpg")), key=os.path.getmtime,
                                  reverse=True) if predict_path else []
        final_image_path = annotated_images[0] if annotated_images else image_path

        predicted_label = classes[int(results[0].boxes.cls.tolist()[0])] if results and results[
            0].boxes else "No pattern detected"
        ws.append([timestamp, final_image_path, predicted_label])
        wb.save(excel_file)

        frame_count += 1
        time.sleep(5)

print(f"Results saved to {excel_file}")