from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    # 训练
    model.train(
        data="Bdataset/data.yaml",
        epochs=200,
        imgsz=640,
        batch=16,
        device=0,
        project="run/yolov8_realtime",
        name="exp",
        exist_ok=True,
        lr0=0.001,
        lrf=0.01
    )

if __name__ == "__main__":
    main()

