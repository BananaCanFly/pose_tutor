# realtime_extractor.py - 实时摄像头版本
import cv2
import mediapipe as mp
import json
import numpy as np
from pathlib import Path

from mediapipe.python.solutions import pose as mp_pose
def overlay_rgba(background, overlay, x, y):
    """
    将 RGBA overlay 按 alpha 通道叠加到 BGR background
    """
    h, w = overlay.shape[:2]

    bg_roi = background[y:y+h, x:x+w]

    alpha = overlay[:, :, 3] / 255.0
    alpha = alpha[..., None]

    bg_roi[:] = (
        alpha * overlay[:, :, :3] +
        (1 - alpha) * bg_roi
    ).astype(np.uint8)
class PoseExtractor:
    def __init__(self):
        """初始化MediaPipe姿势检测"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,            # 摄像头必须使用 video 模式
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose_connections = mp_pose.POSE_CONNECTIONS  # 官方关键点连接

    def extract_from_frame(self, frame):
        """从摄像头单帧提取关键点"""
        if frame is None:
            return None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 检测姿势
        results = self.pose.process(rgb)

        # 整理关键点数据
        keypoints = []
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints.append({
                "id": i,
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
                "visibility": float(landmark.visibility)
            })

        return keypoints

    def draw_skeleton(self, image, keypoints):
        """在图片上绘制骨架"""
        if image is None or keypoints is None:
            return image

        h, w = image.shape[:2]

        # 把关键点转换成字典，方便按 id 查找
        kp_dict = {kp['id']: kp for kp in keypoints}

        # 绘制骨架连线
        for start_id, end_id in self.pose_connections:
            if start_id in kp_dict and end_id in kp_dict:
                p1 = kp_dict[start_id]
                p2 = kp_dict[end_id]

                # 只绘制可见的关节点
                if p1['visibility'] > 0.5 and p2['visibility'] > 0.5:
                    # 颜色区分左右
                    if start_id in [11, 13, 15, 23, 25, 27]:  # 左侧关节
                        color = (0, 255, 0)  # 绿色
                    elif start_id in [12, 14, 16, 24, 26, 28]:  # 右侧关节
                        color = (255, 0, 0)  # 蓝色
                    else:
                        color = (0, 255, 255)  # 其他（躯干、头部）黄色

                    start_point = (int(p1['x'] * w), int(p1['y'] * h))
                    end_point = (int(p2['x'] * w), int(p2['y'] * h))
                    cv2.line(image, start_point, end_point, color, 2)

        # 绘制关节点
        for kp in keypoints:
            if kp['visibility'] > 0.5:
                x, y = int(kp['x'] * w), int(kp['y'] * h)
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # 红色点

        return image

    def draw_skeleton_mini(self, image, keypoints, mini_w, mini_h, margin=15, position='left_bottom'):
        """
        在小窗口上绘制骨架并叠加到原图
        - image: 原图 (BGR)
        - keypoints: 关键点列表
        - mini_w/mini_h: 小窗口尺寸
        - margin: 距离边缘的间距
        - position: 'left_bottom', 'right_bottom' 等
        """
        if image is None or keypoints is None:
            return image

        h, w = image.shape[:2]

        # 1️⃣ 创建透明小画布
        mini_overlay = np.zeros((mini_h, mini_w, 4), dtype=np.uint8)

        # 2️⃣ 计算关键点范围
        # 计算关键点范围
        xs = [kp["x"] for kp in keypoints]
        ys = [kp["y"] for kp in keypoints]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        keypoint_w = max_x - min_x
        keypoint_h = max_y - min_y

        # 统一缩放系数，保证比例不变
        scale_x = mini_w / (keypoint_w + 1e-6)
        scale_y = mini_h / (keypoint_h + 1e-6)
        scale = min(scale_x, scale_y) * 0.9  # 留边距

        offset_x = (mini_w - (max_x - min_x) * scale) / 2
        offset_y = (mini_h - (max_y - min_y) * scale) / 2

        # 3️⃣ 绘制骨架连线
        kp_dict = {kp['id']: kp for kp in keypoints}
        for start_id, end_id in self.pose_connections:
            if start_id in kp_dict and end_id in kp_dict:
                p1, p2 = kp_dict[start_id], kp_dict[end_id]
                if p1['visibility'] > 0.5 and p2['visibility'] > 0.5:
                    start_point = (int((p1['x'] - min_x) * scale + offset_x),
                                   int((p1['y'] - min_y) * scale + offset_y))
                    end_point = (int((p2['x'] - min_x) * scale + offset_x),
                                 int((p2['y'] - min_y) * scale + offset_y))

                    # 颜色区分左右
                    if start_id in [11, 13, 15, 23, 25, 27]:
                        color = (0, 255, 0, 255)  # 左侧关节
                    elif start_id in [12, 14, 16, 24, 26, 28]:
                        color = (255, 0, 0, 255)  # 右侧关节
                    else:
                        color = (0, 255, 255, 255)  # 躯干/头部

                    cv2.line(mini_overlay, start_point, end_point, color, 2)

        # 4️⃣ 绘制关节点
        for kp in keypoints:
            if kp['visibility'] > 0.5:
                x = int((kp['x'] - min_x) * scale + offset_x)
                y = int((kp['y'] - min_y) * scale + offset_y)
                cv2.circle(mini_overlay, (x, y), 1, (0, 0, 255, 255), -1)

        # 5️⃣ 计算贴到原图的坐标
        if position == 'left_bottom':
            x1 = margin
            y1 = h - mini_h - margin
        elif position == 'right_bottom':
            x1 = w - mini_w - margin
            y1 = h - mini_h - margin
        else:
            x1 = margin
            y1 = margin

        # 6️⃣ 叠加到原图
        overlay_rgba(image, mini_overlay, x1, y1)

        return image

    def run_realtime_camera(self):
        """实时摄像头模式"""
        print("摄像头启动中... 按 Q 退出")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("无法打开摄像头")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = self.extract_from_frame(frame)
            frame_drawn = self.draw_skeleton(frame.copy(), keypoints)

            cv2.imshow("Pose Tutor - Camera Mode", frame_drawn)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    extractor = PoseExtractor()
    extractor.run_realtime_camera()


if __name__ == "__main__":
    main()
