from ultralytics import YOLO

model = YOLO("models/ball_detector_model.pt")

results = model.track("input_videos/video_1.mp4", save=True)
print(results)
print("----------")
for box in results[0].boxes:
    print(box)