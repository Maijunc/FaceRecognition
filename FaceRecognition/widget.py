# This Python file uses the following encoding: utf-8
import sys

from PySide6.QtWidgets import QApplication, QWidget

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_Widget

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from ultralytics import YOLO
from keras_facenet import FaceNet
import sqlite3

# 假设你已经有以下函数：
# detect_faces(), extract_features(), match_face()


# 加载 YOLO 模型
model = YOLO('/Users/maitianjun/PycharmProjects/DIPCode/Yolo/src/model/yolov8s-face-lindevs.pt')  # 请确保使用的是人脸检测模型权重

# 初始化 FaceNet 模型
facenet_model = FaceNet()

def detect_faces(frame):
    """检测人脸并返回人脸框坐标"""
    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()  # 转为 NumPy 格式
    return detections  # 返回所有检测到的框

def extract_features(face_image):
    """
    提取单张人脸的特征向量
    :param face_image: 裁剪后的人脸图像（RGB 格式）
    :return: 特征向量（512 维）
    """
    embeddings = facenet_model.embeddings([face_image])
    return embeddings[0]

def initialize_database(db_path='/Users/maitianjun/PycharmProjects/DIPCode/Yolo/faces.db'):
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

def store_face(name, embedding, db_path='/Users/maitianjun/PycharmProjects/DIPCode/Yolo/faces.db'):
    """存储人脸特征到数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO faces (name, embedding) VALUES (?, ?)',
                   (name, embedding.tobytes()))
    conn.commit()
    conn.close()


def process_and_store_face(frame, name, db_path='/Users/maitianjun/PycharmProjects/DIPCode/Yolo/faces.db'):
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

def capture_faces():
    initialize_database()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 显示实时画面
        cv2.imshow("Capture Faces", frame)

        # 按下 's' 键保存人脸
        if cv2.waitKey(10) & 0xFF == ord('s'):
            name = input("请输入人名：")
            process_and_store_face(frame, name)

        # 按下 'q' 键退出
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def load_faces(db_path='/Users/maitianjun/PycharmProjects/DIPCode/Yolo//Users/maitianjun/PycharmProjects/DIPCode/Yolo/faces.db'):
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

def match_face(new_embedding, db_path='/Users/maitianjun/PycharmProjects/DIPCode/Yolo/faces.db', threshold=0.6):
    """
    匹配数据库中的人脸
    :param new_embedding: 新检测到的人脸特征向量
    :param db_path: 数据库路径
    :param threshold: 匹配的相似度阈值
    :return: 匹配到的名字和相似度（如果未匹配，返回 None）
    """
    faces = load_faces(db_path)
    best_match = None
    highest_similarity = -1

    for name, embedding in faces:
        similarity = cosine_similarity(new_embedding, embedding)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = name

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

    def stop(self):
        """停止线程并释放摄像头"""
        self.running = False  # 设置 running 为 False，停止线程
        self.cap.release()


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

        # 按钮区域
        self.button_layout = QHBoxLayout()
        self.start_button = QPushButton("开始识别", self)
        self.start_button.clicked.connect(self.start_detection)
        self.button_layout.addWidget(self.start_button)

        self.layout.addLayout(self.button_layout)

        self.setLayout(self.layout)

        # 后台线程进行识别
        self.face_recognition_thread = FaceRecognitionThread()
        self.face_recognition_thread.update_frame_signal.connect(self.update_frame)
        self.face_recognition_thread.update_label_signal.connect(self.update_label)

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

        height, width, _ = frame_resized.shape
        bytes_per_line = 3 * width

        # 创建 QImage
        q_image = QImage(frame_resized.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # 更新 QLabel 显示
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def update_label(self, labels):
        """更新显示的识别结果标签"""
        label_text = ""
        for name, similarity, bbox in labels:
            # 确保 similarity 不是 None，避免格式化错误
            similarity = similarity if similarity is not None else 0.0
            label_text += f"{name or 'Unknown'} ({similarity:.2f}) at {bbox}\n"


        self.face_name_label.setText(label_text)

    def closeEvent(self, event):
        """关闭应用时停止线程并释放摄像头"""
        self.face_recognition_thread.stop()
        self.face_recognition_thread.wait()
        event.accept()

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

                # 按钮区域
                self.button_layout = QHBoxLayout()
                self.start_button = QPushButton("开始识别", self)
                self.start_button.clicked.connect(self.start_detection)
                self.button_layout.addWidget(self.start_button)

                self.layout.addLayout(self.button_layout)

                self.setLayout(self.layout)

                # 后台线程进行识别
                self.face_recognition_thread = FaceRecognitionThread()
                self.face_recognition_thread.update_frame_signal.connect(self.update_frame)
                self.face_recognition_thread.update_label_signal.connect(self.update_label)

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

                height, width, _ = frame_resized.shape
                bytes_per_line = 3 * width

                # 创建 QImage
                q_image = QImage(frame_resized.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

                # 更新 QLabel 显示
                self.image_label.setPixmap(QPixmap.fromImage(q_image))

            def update_label(self, labels):
                """更新显示的识别结果标签"""
                label_text = "识别结果:\n"
                for name, similarity, bbox in labels:
                    label_text += f"{name or 'Unknown'} ({similarity:.2f}) at {bbox}\n"

                self.face_name_label.setText(label_text)

            def closeEvent(self, event):
                """关闭应用时停止线程并释放摄像头"""
                self.face_recognition_thread.stop()
                self.face_recognition_thread.wait()
                event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec())
