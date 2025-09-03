from re import match
import cv2
import numpy as np
import torch
import pygetwindow as gw
import mss
import keyboard
import time
from PIL import Image, ImageDraw, ImageFont
import threading
from ultralytics import YOLO

class WindowCapture:
    def __init__(self, window_title):
        """初始化窗口捕获"""
        self.window_title = window_title
        self.window = None
        self.update_window_reference()
        self.sct = mss.mss()   # 放在实例里，但仅在单线程里使用

    def update_window_reference(self):
        """更新窗口引用"""
        try:
            windows = gw.getWindowsWithTitle(self.window_title)
            if windows:
                self.window = windows[0]
                print(f"找到窗口: {self.window.title}")
            else:
                print(f"未找到标题为'{self.window_title}'的窗口")
                self.window = None
        except Exception as e:
            print(f"获取窗口时出错: {e}")
            self.window = None

    def get_screenshot(self):
        """获取窗口截图（线程安全版本）"""
        if self.window is None:
            self.update_window_reference()
            if self.window is None:
                return None

        if self.window.isMinimized:
            return None

        # 获取窗口边界
        left, top, right, bottom = (
            self.window.left,
            self.window.top,
            self.window.right,
            self.window.bottom,
        )
        width = right - left
        height = bottom - top

        monitor = {"top": top, "left": left, "width": width, "height": height}

        try:
            # 每次截图都新建一个 mss 实例，避免多线程冲突
            with mss.mss() as sct:
                sct_img = sct.grab(monitor)
                img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                return img
        except Exception as e:
            print(f"截图失败: {e}")
            return None
        
import torch
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.5):
        # 加载模型
        self.model = YOLO(model_path, verbose=False)   # 自动识别YOLOv5/YOLOv8/YOLOv11
        self.conf_threshold = conf_threshold

        # 融合 Conv+BN，加速推理
        self.model.fuse()

        # 自动选择设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device

        print(f"[INFO] YOLO 模型已加载到 {self.device}，阈值={self.conf_threshold}")

    def detect(self, image):
        # 预测
        results = self.model.predict(image, conf=self.conf_threshold, device=self.device, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    'bbox': box.xyxy[0].int().tolist(),        # [x1, y1, x2, y2]
                    'confidence': float(box.conf[0]),          # 置信度
                    'class': int(box.cls[0]),                  # 类别ID
                    'class_name': self.model.names[int(box.cls[0])]  # 类别名
                })
        return detections


class Visualizer:
    def __init__(self):
        """初始化可视化工具"""
        self.font = ImageFont.load_default()
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128), 
            (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
    
    def draw_detections(self, image, detections):
        """在图像上绘制检测结果"""
        if image is None:
            return None
            
        # 转换为PIL图像以便绘制文本
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            color = self.colors[detection['class'] % len(self.colors)]
            draw.rectangle(bbox, outline=color, width=2)

            # 新写法: 用 textbbox 获取文字大小
            label = f"{class_name} {confidence:.2f}"
            label_bbox = draw.textbbox((0, 0), label, font=self.font)
            label_width = label_bbox[2] - label_bbox[0]
            label_height = label_bbox[3] - label_bbox[1]

            draw.rectangle(
                [bbox[0], bbox[1], bbox[0] + label_width + 4, bbox[1] + label_height + 4],
                fill=color
            )
            draw.text((bbox[0] + 2, bbox[1] + 2), label, fill=(255, 255, 255), font=self.font)
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

class box:
    index = -1
    def __init__(self, style, name, late, destion):
        self.id = name
        self.style = style
        self.late = late
        self.destion = destion
    def getindixb_byid(self,id,list):
        for i in range(len(list)):
            if list[i].id == id:
                self.index = i
                return i
        return -1


class RealTimeWindowDetection:
    def __init__(self, window_title, model_path):
        """初始化实时窗口检测系统"""
        self.window_capture = WindowCapture(window_title)
        self.detector = YOLODetector(model_path)
        self.visualizer = Visualizer()
        self.is_running = False
        self.fps = 0
        self.detection_fps = 0

    # holdbox_id = ["hold"]
    # hitbox_id = ['Music note', 'che', 'cheboss', 'gu', 'gui', 'jian', 'jiqi', 'jiqiqiu', 'jiqireng', 'kuai', 'mao', 'qiangboss', 'qiu', 'shuangya', 'tang', 'tapboss', 'xin', 'yuan', 'yun']
    # dis_hitbox_id = ['ci']

    default_late = 0
    default_destion = 500
    box_count = 21
    lead_range = 100
    upanddown_range = 600
    boxs =[box("hold",0,default_late,default_destion),
           box("Music note",1,default_late,default_destion),
           box("che",1,default_late,default_destion),
           box("cheboss",1,default_late,default_destion),
           box("gu",1,default_late,default_destion),
           box("gui",1,default_late,default_destion),
           box("jian",1,default_late,default_destion),
           box("jiqi",1,default_late,default_destion),
           box("jiqiqiu",1,default_late,default_destion),
           box("jiqireng",1,default_late,default_destion),
           box("kuai",1,default_late,default_destion),
           box("mao",1,default_late,default_destion),
           box("qiangboss",1,default_late,default_destion),
           box("qiu",1,default_late,default_destion),
           box("shuangya",3,default_late,default_destion),
           box("tang",1,default_late,default_destion),
           box("tapboss",1,default_late,default_destion),
           box("xin",1,default_late,default_destion),
           box("yuan",1,default_late,default_destion),
           box("yun",1,default_late,default_destion),
           box("ci",2,default_late,default_destion)
           ]

    box_in = [False] * box_count
    box_lasttime_in = [False] * box_count

    #判断与模拟按键
    def simulate_key_press(self,detections):
        # 这里可以添加代码来模拟按键
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            type = -1
            index = -1
            # 根据class_name查找对应的box
            for box in self.boxs:
                if box.style == class_name:
                    type = box.id
                    index = box.index
                    center = [0, 0]
                    center[0] = (bbox[0] + bbox[2]) / 2
                    center[1] = (bbox[1] + bbox[3]) / 2
                    if type != -1:
                        #print(f"检测到目标: {class_name} \t (置信度: {confidence:.2f}) \t 中心点: {center}")
                        if abs(center[0] - box.destion) <= self.lead_range:
                            print(f"目标 {class_name} 进入判定范围")
                            print(f"目标: {class_name} \t (置信度: {confidence:.2f}) \t 中心点: {center}")
                            if (self.box_lasttime_in[index] == False):
                                print(f"目标 {class_name} 第一次进入判定范围")
                            self.box_in[index] = True
                            if type == 0:
                                if center[1] > self.upanddown_range:
                                    self.late_presscode_add("j", box.late, True)
                                else:
                                    self.late_presscode_add("f", box.late, True)
                            if type == 1:
                                if (self.box_lasttime_in[index] == False):
                                    if center[1] > self.upanddown_range:
                                        self.late_presscode_add("j", box.late, True)
                                    else:
                                        self.late_presscode_add("f", box.late, True)
                            if type == 2:
                                if (self.box_lasttime_in[index] == False):
                                    if center[1] <= self.upanddown_range:
                                        self.late_presscode_add("j", box.late, True)
                                    else:
                                        self.late_presscode_add("f", box.late, True)
                            if type == 3:
                                self.late_presscode_add("j", box.late, True)
                                self.late_presscode_add("f", box.late, True)
                            self.box_lasttime_in[index] = True
                        else:
                            self.box_lasttime_in[index] = self.box_in[index]
                            self.box_in[index] = False
                            if self.box_lasttime_in[index]:
                                if type != 2:
                                    if center[1] > self.upanddown_range:
                                        self.late_presscode_add("j", box.late, False)
                                    else:
                                        self.late_presscode_add("f", box.late, False)
                                else:
                                    if center[1] <= self.upanddown_range:
                                        self.late_presscode_add("j", box.late, False)
                                    else:
                                        self.late_presscode_add("f", box.late, False)
                    break

    late_presscode_list = []

    def late_presscode_add(self, key_code, late, press_down=True):
        self.late_presscode_list.append((key_code, late, press_down))
    def refresh_late_presscode(self,loss_time):
        if self.late_presscode_list.__len__() > 0:
            i=0
            while i < self.late_presscode_list.__len__():
                key_code, late, press_down = self.late_presscode_list[i]
                late -= loss_time
                if late < 0:
                    if press_down:
                        keyboard.press(key_code)
                    else:
                        keyboard.release(key_code)
                    self.late_presscode_list.pop(i)
                else:
                    self.late_presscode_list[i] = (key_code, late, press_down)
                    i += 1

    def start(self):
        """开始实时检测"""
        self.is_running = True
        detection_thread = threading.Thread(target=self._detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        
        # 主线程用于显示
        self._display_loop()
    
    def stop(self):
        """停止检测"""
        self.is_running = False
    
    def _detection_loop(self):
        """检测循环"""
        last_time = time.time()
        frame_count = 0
        
        while self.is_running:
            start_time = time.time()
            
            # 获取截图
            screenshot = self.window_capture.get_screenshot()
            if screenshot is None:
                time.sleep(0.1)
                continue

            # #黑白化(依旧三通道）
            # screenshot = cv2.cvtColor(screenshot_raw, cv2.COLOR_BGR2GRAY)
            # screenshot = cv2.cvtColor(screenshot, cv2.COLOR_GRAY2BGR)

            # 执行检测
            detections = self.detector.detect(screenshot)
            
            # 计算FPS
            frame_count += 1
            if time.time() - last_time >= 1.0:
                self.detection_fps = frame_count
                frame_count = 0
                last_time = time.time()
            
            # 存储结果用于显示
            self.last_screenshot = screenshot
            self.last_detections = detections
            
            # 控制检测频率
            processing_time = time.time() - start_time
            if processing_time < 0.01:  # 约30FPS
                time.sleep(0.01 - processing_time)
    
    def _display_loop(self):
        """显示循环"""
        last_time = time.time()
        frame_count = 0
        
        cv2.namedWindow("Real-time Window Detection", cv2.WINDOW_NORMAL)
        
        while self.is_running:
            # 检查是否有可用的检测结果
            if hasattr(self, 'last_screenshot') and hasattr(self, 'last_detections'):
                # 可视化检测结果
                visualized_image = self.visualizer.draw_detections(
                    self.last_screenshot, self.last_detections
                )
                
                # 显示FPS信息
                fps_text = f"Detection FPS: {self.detection_fps} | Display FPS: {self.fps}"
                cv2.putText(visualized_image, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示图像
                cv2.imshow("Real-time Window Detection", visualized_image)
                
                #判定与模拟按键
                self.simulate_key_press(self.last_detections)

                # 计算显示FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    self.fps = frame_count
                    frame_count = 0
                    last_time = current_time
            if self.fps != 0:
                self.refresh_late_presscode(1000/self.fps)
            # 检查退出条件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 使用示例
    window_title = "MuseDash"  # 替换为你的窗口标题
    model_path = "model\\best.pt"       # 替换为你的模型路径

    detector = RealTimeWindowDetection(window_title, model_path)
    
    try:
        detector.start()
    except KeyboardInterrupt:
        detector.stop()
        print("程序已停止")