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
        """在图片上绘制完整骨架（包含面部、手部细节、脚踝）"""
        try:
            if image is None: return None
            height, width = image.shape[:2]

            # === 1. 定义连线关系 (全细节) ===
            connections = [
                # 躯干
                (11, 12), (11, 23), (12, 24), (23, 24),
                # 手臂
                (11, 13), (13, 15), (12, 14), (14, 16),
                # 腿部
                (23, 25), (25, 27), (24, 26), (26, 28),
                # --- 新增细节 ---
                # 面部
                (0, 1), (1, 2), (2, 3), (3, 7),  # 左眼区
                (0, 4), (4, 5), (5, 6), (6, 8),  # 右眼区
                (9, 10),  # 嘴巴
                # 手部 (手腕到指尖)
                (15, 17), (15, 19), (15, 21), (17, 19),  # 左手掌
                (16, 18), (16, 20), (16, 22), (18, 20),  # 右手掌
                # 脚部 (脚踝到脚跟、脚尖)
                (27, 29), (27, 31), (29, 31),  # 左脚
                (28, 30), (28, 32), (30, 32)  # 右脚
            ]

            # === 2. 绘制连线 ===
            for start, end in connections:
                if start < len(keypoints) and end < len(keypoints):
                    kp1 = keypoints[start]
                    kp2 = keypoints[end]

                    # 只要可见度 > 0.5 就画出来，不做严格过滤
                    if kp1['visibility'] > 0.5 and kp2['visibility'] > 0.5:
                        p1 = (int(kp1['x'] * width), int(kp1['y'] * height))
                        p2 = (int(kp2['x'] * width), int(kp2['y'] * height))

                        # 根据身体部位使用不同颜色
                        color = (0, 255, 0)  # 默认绿色
                        if start <= 10:
                            color = (255, 200, 0)  # 面部青色
                        elif start >= 25:
                            color = (0, 165, 255)  # 腿部橙色
                        elif start >= 15 and start <= 22:
                            color = (255, 0, 255)  # 手部紫色

                        cv2.line(image, p1, p2, color, 2)

            # === 3. 绘制关键点 ===
            for i, kp in enumerate(keypoints):
                if kp['visibility'] > 0.5:
                    x = int(kp['x'] * width)
                    y = int(kp['y'] * height)

                    # 关键点颜色
                    if i <= 10:
                        c = (255, 200, 0)  # 面部
                    elif i >= 15 and i <= 22:
                        c = (255, 0, 255)  # 手部
                    elif i >= 27:
                        c = (0, 165, 255)  # 脚部
                    else:
                        c = (0, 0, 255)  # 躯干红色

                    cv2.circle(image, (x, y), 4, c, -1)

            # return image

        except Exception as e:
            print(f"❌ 绘制骨架时出错: {e}")
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

        # 1️⃣ 创建透明小画布 (RGBA)
        mini_overlay = np.zeros((mini_h, mini_w, 4), dtype=np.uint8)

        # 2️⃣ 计算关键点范围
        xs = [kp["x"] for kp in keypoints]
        ys = [kp["y"] for kp in keypoints]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        keypoint_w = max_x - min_x
        keypoint_h = max_y - min_y

        # 统一缩放系数，保持比例
        scale_x = mini_w / (keypoint_w + 1e-6)
        scale_y = mini_h / (keypoint_h + 1e-6)
        scale = min(scale_x, scale_y) * 0.9  # 留边距

        offset_x = (mini_w - (max_x - min_x) * scale) / 2
        offset_y = (mini_h - (max_y - min_y) * scale) / 2

        # === 3. 绘制骨架连线 ===
        connections = [
            # 躯干
            (11, 12), (11, 23), (12, 24), (23, 24),
            # 手臂
            (11, 13), (13, 15), (12, 14), (14, 16),
            # 腿部
            (23, 25), (25, 27), (24, 26), (26, 28),
            # --- 新增细节 ---
            # 面部
            (0, 1), (1, 2), (2, 3), (3, 7),  # 左眼区
            (0, 4), (4, 5), (5, 6), (6, 8),  # 右眼区
            (9, 10),  # 嘴巴
            # 手部 (手腕到指尖)
            (15, 17), (15, 19), (15, 21), (17, 19),  # 左手掌
            (16, 18), (16, 20), (16, 22), (18, 20),  # 右手掌
            # 脚部 (脚踝到脚跟、脚尖)
            (27, 29), (27, 31), (29, 31),  # 左脚
            (28, 30), (28, 32), (30, 32)  # 右脚
        ]

        # 绘制连线
        for start, end in connections:
            if start < len(keypoints) and end < len(keypoints):
                kp1 = keypoints[start]
                kp2 = keypoints[end]
                if kp1['visibility'] > 0.5 and kp2['visibility'] > 0.5:
                    p1 = (int((kp1['x'] - min_x) * scale + offset_x),
                          int((kp1['y'] - min_y) * scale + offset_y))
                    p2 = (int((kp2['x'] - min_x) * scale + offset_x),
                          int((kp2['y'] - min_y) * scale + offset_y))

                    # 根据身体部位使用相同颜色
                    color = (0, 255, 0,255)  # 默认绿色
                    if start <= 10:
                        color = (255, 200, 0,255)  # 面部青色
                    elif start >= 25:
                        color = (0, 165, 255,255)  # 腿部橙色
                    elif 15 <= start <= 22:
                        color = (255, 0, 255, 255)  # 手部紫色

                    cv2.line(mini_overlay, p1, p2, color, 1)

        # 绘制关键点
        for i, kp in enumerate(keypoints):
            if kp['visibility'] > 0.5:
                x = int((kp['x'] - min_x) * scale + offset_x)
                y = int((kp['y'] - min_y) * scale + offset_y)

                if i <= 10:
                    c = (255, 200, 0,255)  # 面部
                elif 15 <= i <= 22:
                    c = (255, 0, 255,255)  # 手部
                elif i >= 27:
                    c = (0, 165, 255,255)  # 脚部
                else:
                    c = (0, 0, 255,255)  # 躯干

                cv2.circle(mini_overlay, (x, y), 2, c, -1)


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
