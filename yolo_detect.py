import cv2
from ultralytics import YOLOWorld
#加载模型
model = YOLOWorld('yolov8s-worldv2.pt')
#设置你想识别的物体
custom_classes = ["person","hand","headphones", "cell phone", "bottle", "keyboard", "mouse", "glasses", "book"]
model.set_classes(custom_classes)
#启动摄像头
cap = cv2.VideoCapture(0)
print(" YOLO 实时检测启动成功！按 'q' 退出。")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    # 推理识别
    results = model.predict(frame, conf=0.3, stream=True)
    #在画面上绘制结果
    for r in results:
        annotated_frame = r.plot()
    #显示画面
    cv2.imshow("YOLO Real-time Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()