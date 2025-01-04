from ultralytics import YOLO
from keras_facenet import FaceNet
import cv2
import sqlite3
import numpy as np

# 加载 YOLO 模型
model = YOLO('/Users/maitianjun/PycharmProjects/DIPCode/Yolo/src/model/yolov8x-face-lindevs.pt')  # 请确保使用的是人脸检测模型权重

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

def initialize_database(db_path='faces.db'):
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

def store_face(name, embedding, db_path='faces.db'):
    """存储人脸特征到数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO faces (name, embedding) VALUES (?, ?)',
                   (name, embedding.tobytes()))
    conn.commit()
    conn.close()


def process_and_store_face(frame, name, db_path='faces.db'):
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


def load_faces(db_path='faces.db'):
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

def match_face(new_embedding, db_path='faces.db', threshold=0.6):
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

def recognize_faces():
    """
    实时人脸识别
    """
    initialize_database()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        # 显示视频流
        cv2.imshow("Real-Time Face Recognition", frame)

        # 按下 'q' 键退出
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_faces()


