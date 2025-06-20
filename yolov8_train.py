from ultralytics import YOLO

# โหลด YOLOv8 รุ่นเบา (เปลี่ยนได้เป็น yolov8s.pt, yolov8m.pt)
model = YOLO('yolov8n.pt')

# ฝึกโมเดล
model.train(
    data='datasets/oyster_mushroom/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='yolov8-oyster'
)
