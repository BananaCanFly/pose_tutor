# pose_extractor.py
import cv2
import mediapipe as mp
import json
import numpy as np
from pathlib import Path
import os


class PoseExtractor:
    def __init__(self):
        """初始化MediaPipe姿势检测"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            model_complexity=1  # 保持默认的中等模型
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_from_image(self, image_path):
        """从单张图片提取骨架关键点"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ 无法读取图片: {image_path}")
                return None

            print(f"📷 正在处理图片: {image_path}")

            # 转换颜色空间
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 检测姿势
            results = self.pose.process(image_rgb)

            if not results.pose_landmarks:
                print(f"⚠️ 未检测到人体姿势: {image_path}")
                return None

            print(f"✅ 检测到姿势，共 {len(results.pose_landmarks.landmark)} 个关键点")

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

        except Exception as e:
            print(f"❌ 处理图片时出错 {image_path}: {e}")
            return None

    def save_keypoints(self, keypoints, output_path):
        """保存关键点到JSON文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(keypoints, f, indent=2, ensure_ascii=False)
            print(f"✅ 已保存关键点: {output_path}")
            return True
        except Exception as e:
            print(f"❌ 保存关键点时出错: {e}")
            return False

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

            return image

        except Exception as e:
            print(f"❌ 绘制骨架时出错: {e}")
            return image

    def create_preview_image(self, image_path, keypoints, output_path):
        """创建带骨架标注的预览图"""
        try:
            image = cv2.imread(image_path)
            if image is None: return

            image_with_skeleton = image.copy()
            image_with_skeleton = self.draw_skeleton(image_with_skeleton, keypoints)

            if image_with_skeleton is not None:
                img_name = Path(image_path).stem
                preview_path = output_path / f"{img_name}_preview.jpg"
                cv2.imwrite(str(preview_path), image_with_skeleton)
                print(f"✅ 已保存预览图: {preview_path}")

        except Exception as e:
            print(f"❌ 创建预览图时出错: {e}")

    def process_folder(self, input_folder="standard_poses_raw", output_folder="standard_poses"):
        """处理整个文件夹"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)

        if not input_path.exists():
            print(f"❌ 输入文件夹不存在: {input_path}")
            input_path.mkdir(exist_ok=True)
            return 0

        output_path.mkdir(exist_ok=True)

        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(input_path.glob(f"*{ext}")))
            image_files.extend(list(input_path.glob(f"*{ext.upper()}")))

        if len(image_files) == 0:
            print("⚠️ 没有找到图片文件")
            return 0

        processed_count = 0
        for img_file in image_files:
            print(f"\n🔄 正在处理: {img_file.name}")
            keypoints = self.extract_from_image(str(img_file))

            if keypoints:
                json_name = img_file.stem + '.json'
                json_path = output_path / json_name
                if self.save_keypoints(keypoints, json_path):
                    self.create_preview_image(str(img_file), keypoints, output_path)
                    processed_count += 1
            else:
                print(f"❌ 无法提取 {img_file.name} 的骨架")

        return processed_count


def main():
    print("=" * 60)
    print("🤖 AI姿势教练 - 全身骨架提取工具")
    print("=" * 60)

    extractor = PoseExtractor()
    processed = extractor.process_folder()

    if processed > 0:
        print(f"\n✅ 成功处理 {processed} 张图片")
    else:
        print("\n❌ 没有处理图片")


if __name__ == "__main__":
    main()