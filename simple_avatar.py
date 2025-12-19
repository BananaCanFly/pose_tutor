# simple_avatar.py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

#
# class SimpleAvatar:
#     def __init__(self, width=400, height=600):
#         self.width = width
#         self.height = height
#
#     def create_pose_image(self, keypoints, title="姿势示意图"):
#         """创建姿势示意图"""
#         if not keypoints or len(keypoints) == 0:
#             # 创建空白图像
#             img = Image.new('RGB', (self.width, self.height), color='white')
#             draw = ImageDraw.Draw(img)
#             draw.text((50, 50), "无姿势数据", fill='red')
#             return img
#
#         fig, ax = plt.subplots(figsize=(6, 8))
#
#         # 设置背景
#         ax.set_facecolor('#f8f9fa')
#         fig.patch.set_facecolor('#ffffff')
#
#         # 绘制骨架
#         self._draw_skeleton(ax, keypoints)
#
#         # 绘制关节点
#         self._draw_joints(ax, keypoints)
#
#         # 添加标题
#         ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
#
#         # 设置坐标轴
#         ax.set_xlim(0, 1)
#         ax.set_ylim(1, 0)  # 反转Y轴，使顶部为0
#         ax.set_aspect('equal')
#
#         # 隐藏坐标轴
#         ax.axis('off')
#
#         # 调整布局
#         plt.tight_layout()
#
#         # 将图像保存到内存
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
#         plt.close(fig)
#
#         buf.seek(0)
#         img = Image.open(buf)
#         return img
#
#     def _draw_skeleton(self, ax, keypoints):
#         """绘制骨架连线"""
#         # 骨架连接关系（MediaPipe 33个关键点）
#         connections = [
#             (11, 13), (13, 15),  # 左臂
#             (12, 14), (14, 16),  # 右臂
#             (11, 12),  # 肩膀
#             (11, 23), (12, 24),  # 躯干
#             (23, 25), (25, 27),  # 左腿
#             (24, 26), (26, 28)  # 右腿
#         ]
#
#         xs = [kp['x'] for kp in keypoints]
#         ys = [kp['y'] for kp in keypoints]
#
#         for start, end in connections:
#             if start < len(xs) and end < len(xs):
#                 # 绘制线条
#                 ax.plot([xs[start], xs[end]],
#                         [ys[start], ys[end]],
#                         color='#3498db',  # 蓝色
#                         linewidth=3,
#                         alpha=0.8,
#                         solid_capstyle='round')
#
#     def _draw_joints(self, ax, keypoints):
#         """绘制关节点"""
#         # 重要关节及其颜色
#         important_joints = {
#             0: ("头部", "#e74c3c"),  # 红色
#             11: ("左肩", "#2ecc71"),  # 绿色
#             12: ("右肩", "#2ecc71"),
#             13: ("左肘", "#f39c12"),  # 橙色
#             14: ("右肘", "#f39c12"),
#             15: ("左腕", "#3498db"),  # 蓝色
#             16: ("右腕", "#3498db"),
#             23: ("左臀", "#9b59b6"),  # 紫色
#             24: ("右臀", "#9b59b6")
#         }
#
#         xs = [kp['x'] for kp in keypoints]
#         ys = [kp['y'] for kp in keypoints]
#
#         for joint_id, (label, color) in important_joints.items():
#             if joint_id < len(xs):
#                 # 绘制圆点
#                 ax.scatter(xs[joint_id], ys[joint_id],
#                            s=100,  # 点的大小
#                            color=color,
#                            edgecolors='white',
#                            linewidth=2,
#                            zorder=5)  # 确保点在线条上面
#
#                 # 添加标签
#                 ax.text(xs[joint_id] + 0.02, ys[joint_id] - 0.02,
#                         label,
#                         fontsize=9,
#                         color=color,
#                         fontweight='bold',
#                         bbox=dict(boxstyle="round,pad=0.2",
#                                   facecolor="white",
#                                   alpha=0.7,
#                                   edgecolor=color))
#
#     def create_comparison_image(self, user_keypoints, std_keypoints, user_label="你的姿势", std_label="标准姿势"):
#         """创建对比图"""
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
#
#         # 绘制用户姿势
#         self._draw_skeleton(ax1, user_keypoints)
#         self._draw_joints(ax1, user_keypoints)
#         ax1.set_title(f"👤 {user_label}", fontsize=14, fontweight='bold')
#         ax1.set_xlim(0, 1)
#         ax1.set_ylim(1, 0)
#         ax1.set_aspect('equal')
#         ax1.axis('off')
#
#         # 绘制标准姿势
#         self._draw_skeleton(ax2, std_keypoints)
#         self._draw_joints(ax2, std_keypoints)
#         ax2.set_title(f"🎯 {std_label}", fontsize=14, fontweight='bold')
#         ax2.set_xlim(0, 1)
#         ax2.set_ylim(1, 0)
#         ax2.set_aspect('equal')
#         ax2.axis('off')
#
#         # 设置整体背景
#         fig.patch.set_facecolor('#ffffff')
#
#         # 调整布局
#         plt.tight_layout()
#
#         # 保存到内存
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
#         plt.close(fig)
#
#         buf.seek(0)
#         img = Image.open(buf)
#         return img

import matplotlib.pyplot as plt
from mediapipe.python.solutions import pose as mp_pose
from PIL import Image
import io

class SimpleAvatar:
    def __init__(self):
        self.pose_connections = mp_pose.POSE_CONNECTIONS  # 官方完整连接

    def _draw_skeleton(self, ax, keypoints):
        """绘制完整骨架连线"""
        # 转换为 dict，方便索引
        kp_dict = {kp["id"]: kp for kp in keypoints}

        for start_id, end_id in self.pose_connections:
            if start_id in kp_dict and end_id in kp_dict:
                p1 = kp_dict[start_id]
                p2 = kp_dict[end_id]
                # 可见性阈值
                if p1["visibility"] > 0.5 and p2["visibility"] > 0.5:
                    ax.plot([p1["x"], p2["x"]],
                            [p1["y"], p2["y"]],
                            color='green', linewidth=2)

    def _draw_joints(self, ax, keypoints):
        """绘制关节点"""
        for kp in keypoints:
            if kp["visibility"] > 0.5:
                ax.scatter(kp["x"], kp["y"], color='red', s=20)

    def create_comparison_image(self, user_keypoints, std_keypoints,
                                user_label="你的姿势", std_label="标准姿势"):
        """创建对比图（官方骨架）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

        # 用户姿势
        self._draw_skeleton(ax1, user_keypoints)
        self._draw_joints(ax1, user_keypoints)
        ax1.set_title(f"👤 {user_label}", fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(1, 0)
        ax1.set_aspect('equal')
        ax1.axis('off')

        # 标准姿势
        self._draw_skeleton(ax2, std_keypoints)
        self._draw_joints(ax2, std_keypoints)
        ax2.set_title(f"🎯 {std_label}", fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(1, 0)
        ax2.set_aspect('equal')
        ax2.axis('off')

        fig.patch.set_facecolor('#ffffff')
        plt.tight_layout()

        # 保存到内存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        return img
