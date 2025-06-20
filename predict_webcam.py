from ultralytics import YOLO

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = YOLO("runs/detect/yolov8-oyster/weights/best.pt")

# เปิดกล้องและเริ่มพยากรณ์แบบ real-time
model.predict(
    source=0,        # กล้อง webcam
    show=True,       # แสดงภาพพร้อมกรอบวัตถุ
    conf=0.4,        # ความมั่นใจขั้นต่ำ
    save=False       # ไม่เซฟภาพผลลัพธ์
)
