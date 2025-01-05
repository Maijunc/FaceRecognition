import sys
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox, QInputDialog
import cv2
import numpy as np
from ultralytics import YOLO
from keras_facenet import FaceNet
import sqlite3
import json
import os
import time


# 加载配置文件
def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

config = load_config()

# 获取当前脚本的目录
base_dir = os.path.dirname(os.path.realpath(__file__))

# 使用配置中的相对路径
model_path = os.path.join(base_dir, config["model_path"])
db_path = os.path.join(base_dir, config["database_path"])

# 加载 YOLO 模型
model = YOLO(model_path)

# 初始化 FaceNet 模型
facenet_model = FaceNet()

def detect_faces(frame):
    """检测人脸并返回人脸框坐标"""
    start_time = time.time()
    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()  # 转为 NumPy 格式
    detection_time = time.time() - start_time
    print(f"Detection time: {detection_time:.4f}s")
    return detections  # 返回所有检测到的框


def extract_features(face_image):
    """
    提取单张人脸的特征向量
    :param face_image: 裁剪后的人脸图像（RGB 格式）
    :return: 特征向量（512 维）
    """
    start_time = time.time()
    embeddings = facenet_model.embeddings([face_image])
    feature_extraction_time = time.time() - start_time
    print(f"Feature extraction time: {feature_extraction_time:.4f}s")
    return embeddings[0]


def initialize_database(db_path=db_path):
    """初始化数据库，创建表格"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def store_face(name, embedding, db_path=db_path):
    """存储人脸特征到数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO faces (name, embedding) VALUES (?, ?)',
                   (name, embedding.tobytes()))
    conn.commit()
    conn.close()


def process_and_store_face(frame, name, db_path=db_path):
    """
    检测人脸 -> 提取特征 -> 存储到数据库
    :param frame: 输入图像（BGR 格式）
    :param name: 人脸的名字
    :param db_path: 数据库路径
    """
    detections = detect_faces(frame)
    for x1, y1, x2, y2 in detections:
        # 裁剪人脸图像并转换为 RGB 格式
        face = frame[int(y1):int(y2), int(x1):int(x2)]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # 提取特征
        embedding = extract_features(face_rgb)

        # 存储到数据库
        store_face(name, embedding, db_path=db_path)
        print(f"人脸 {name} 已存储到数据库")

def load_faces(db_path=db_path):
    """加载数据库中的所有人脸特征"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT name, embedding FROM faces')
    data = cursor.fetchall()
    conn.close()

    # 转换特征向量为 NumPy 格式
    faces = []
    for name, embedding in data:
        faces.append((name, np.frombuffer(embedding, dtype=np.float32)))
    return faces


def cosine_similarity(vec1, vec2):
    """
    计算两个特征向量之间的余弦相似度
    :param vec1: 特征向量1
    :param vec2: 特征向量2
    :return: 余弦相似度（范围 -1 到 1）
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def match_face(new_embedding, db_path=db_path, threshold=0.6):
    """
    匹配数据库中的人脸
    :param new_embedding: 新检测到的人脸特征向量
    :param db_path: 数据库路径
    :param threshold: 匹配的相似度阈值
    :return: 匹配到的名字和相似度（如果未匹配，返回 None）
    """
    start_time = time.time()
    faces = load_faces(db_path)
    best_match = None
    highest_similarity = -1

    for name, embedding in faces:
        similarity = cosine_similarity(new_embedding, embedding)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = name

    matching_time = time.time() - start_time
    print(f"Matching time: {matching_time:.4f}s")
    if highest_similarity >= threshold:
        return best_match, highest_similarity
    else:
        return None, None


class FaceRecognitionThread(QThread):
    update_frame_signal = pyqtSignal(np.ndarray)  # 发出图像更新信号
    update_label_signal = pyqtSignal(list)  # 发出多个标签更新信号，返回格式为 [(name, similarity, (x1, y1, x2, y2)), ...]

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.running = True  # 初始化 running 属性，用来控制线程是否运行

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 获取当前时间
            start_time = time.time()

            # 进行图像预处理
            frame = self.preprocess_image(frame)

            labels = []  # 存储多个标签的列表
            # 检测人脸
            detections = detect_faces(frame)
            for x1, y1, x2, y2 in detections:
                # 裁剪人脸并转换为 RGB
                face = frame[int(y1):int(y2), int(x1):int(x2)]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                # 提取特征并匹配
                embedding = extract_features(face_rgb)
                name, similarity = match_face(embedding)

                # 绘制结果
                label = f"{name} ({similarity:.2f})" if name else "Unknown"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 保存人脸标签信息
                labels.append((name, similarity, (x1, y1, x2, y2)))

            # 发出图像更新信号
            self.update_frame_signal.emit(frame)

            # 发出标签更新信号
            self.update_label_signal.emit(labels)

            # 总时间
            total_time = time.time() - start_time

            print(f"Total time: {total_time:.4f}s")

    def stop(self):
        """停止线程并释放摄像头"""
        self.running = False  # 设置 running 为 False，停止线程
        self.cap.release()

    def preprocess_image(self, frame):
        """应用图像预处理步骤"""
        # 1. 直方图均衡化
        # 对每个通道进行均衡化
        for i in range(3):  # 针对 B, G, R 通道
            channel = frame[:, :, i]
            frame[:, :, i] = cv2.equalizeHist(channel)

        # 2. 高斯滤波
        # 在彩色图像上应用高斯滤波
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # 3. 拉普拉斯锐化
        # 分别对每个通道进行拉普拉斯锐化
        for i in range(3):
            channel = frame[:, :, i]
            laplacian = cv2.Laplacian(channel, cv2.CV_64F)
            laplacian = cv2.convertScaleAbs(laplacian)  # 处理负值
            frame[:, :, i] = cv2.add(frame[:, :, i], laplacian)

        return frame




class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('实时人脸识别')
        self.setGeometry(100, 100, 800, 600)

        # 创建界面布局
        self.layout = QVBoxLayout()

        # 摄像头画面显示区域
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)  # 设置固定的宽度和高度
        self.layout.addWidget(self.image_label)

        # 识别结果显示区域
        self.face_name_label = QLabel("等待识别...", self)
        self.layout.addWidget(self.face_name_label)

        # 新增拍照结果显示区域
        self.photo_label = QLabel("拍照结果将在此显示", self)
        self.photo_label.setFixedSize(200, 200)  # 设置固定的宽度和高度
        self.layout.addWidget(self.photo_label)

        # 按钮区域
        self.button_layout = QHBoxLayout()
        self.start_button = QPushButton("开始识别", self)
        self.start_button.clicked.connect(self.start_detection)
        self.register_button = QPushButton("注册", self)
        self.register_button.clicked.connect(self.register)
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.register_button)

        self.layout.addLayout(self.button_layout)

        self.setLayout(self.layout)

        # 后台线程进行识别
        self.face_recognition_thread = FaceRecognitionThread()
        self.face_recognition_thread.update_frame_signal.connect(self.update_frame)
        self.face_recognition_thread.update_label_signal.connect(self.update_label)

        initialize_database()

        # 追加一个标志用于判断拍照是否完成
        self.is_registering = False

    def start_detection(self):
        """启动人脸识别后台线程"""
        self.face_recognition_thread.start()

    def update_frame(self, frame):
        """更新显示的摄像头图像"""
        # 获取当前 widget 的大小
        widget_width = self.image_label.width()
        widget_height = self.image_label.height()

        # 调整图像大小为与 widget 大小一致
        frame_resized = cv2.resize(frame, (widget_width, widget_height))

        # 转换为 RGB 格式
        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # 将 NumPy 数组转换为 QPixmap
        h, w, ch = frame_resized.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_resized.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def update_label(self, labels):
        """更新显示的识别结果标签"""
        label_text = "识别结果:\n"
        for name, similarity, bbox in labels:
            # 如果 similarity 为 None，设置为 0.0
            similarity = similarity if similarity is not None else 0.0
            label_text += f"{name or 'Unknown'} ({similarity:.2f}) at {bbox}\n"

        self.face_name_label.setText(label_text)

    def register(self):
        """注册新用户并拍照"""
        # 获取用户名
        name, ok = QInputDialog.getText(self, "输入用户名", "请输入用户名：")
        if ok and name:
            # 暂时停止人脸识别线程
            self.face_recognition_thread.stop()

            # 等待用户点击拍照按钮
            self.face_recognition_thread.running = True  # 重新启动线程以进行拍照
            self.face_recognition_thread.cap.release()  # 释放摄像头
            self.face_recognition_thread.cap = cv2.VideoCapture(0)  # 重新打开摄像头

            # 设置标志位，表示正在进行注册
            self.is_registering = True

            # 等待用户拍照
            self.face_recognition_thread.start()
            self.face_recognition_thread.update_frame_signal.connect(lambda frame: self.take_picture(frame, name))  # 传递 name

    def take_picture(self, frame, name):
        """拍照并显示检测到的人脸"""
        if self.is_registering:  # 只有在注册时才执行拍照
            # 检测人脸
            detections = detect_faces(frame)
            if detections.size > 0:  # 检查 detections 是否为空
                for x1, y1, x2, y2 in detections:
                    # 裁剪人脸图像
                    face = frame[int(y1):int(y2), int(x1):int(x2)]
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                    # 提取特征
                    embedding = extract_features(face_rgb)

                    # 存储人脸
                    store_face(name, embedding)

                    # 显示拍到的人脸
                    face_resized = cv2.resize(face_rgb, (200, 200))  # 将人脸图像缩放
                    face_img = QImage(face_resized.data, face_resized.shape[1], face_resized.shape[0], face_resized.strides[0], QImage.Format.Format_RGB888)
                    self.photo_label.setPixmap(QPixmap.fromImage(face_img))  # 更新显示为人脸图像

                    # 设置标志位为 False，表示拍照完成
                    self.is_registering = False

                    # 拍照后重新启动人脸识别线程
                    self.face_recognition_thread.start()

                    break  # 一次只拍一张人脸，若检测到多张则只保存一张

    def closeEvent(self, event):
        """关闭时停止线程"""
        self.face_recognition_thread.stop()
        self.face_recognition_thread.wait()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec())
