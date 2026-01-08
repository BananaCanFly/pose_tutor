# app.py - AI姿势教练主应用
import streamlit as st
import cv2
import tempfile
import numpy as np
import os
from pathlib import Path

import torch
from PIL import Image
import json
import matplotlib
matplotlib.use("Agg")


# 导入我们的模块
try:
    # from pose_extractor import PoseExtractor
    from realtime_extractor import PoseExtractor
    from pose_analyzer import PoseAnalyzer
    from simple_avatar import SimpleAvatar
    from CompositionAnalyzer import analyze_crop_and_zoom
except ImportError as e:
    st.error(f"导入模块出错: {e}")
    st.stop()

SCORE_DIFF_THRESHOLD = 5
FRAME_COUNT_EVERY_PROCESS = 5

# 页面配置
st.set_page_config(
    page_title="拍照提示",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .suggestion-card {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .part-analysis-card {
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 4px solid;
        transition: transform 0.2s;
    }
    .joint-difference-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    .success-badge {
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .warning-badge {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .error-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
</style>
""", unsafe_allow_html=True)



class PoseCoachApp:
    def __init__(self):
        """初始化应用"""
        self.extractor = PoseExtractor()
        self.analyzer = PoseAnalyzer()
        self.avatar = SimpleAvatar()

        # 创建输出目录
        Path("outputs").mkdir(exist_ok=True)
        Path("user_photos").mkdir(exist_ok=True)

    def process_realtime_frame(
            self,
            frame,
            selected_pose=None,
            prev_analysis=None,
            score_threshold=SCORE_DIFF_THRESHOLD
    ):
        """
        实时处理单帧：
        - 原图：用户骨架
        - 左下角：目标姿势骨架（始终显示）
        """

        # ======================
        # 1️⃣ 提取用户关键点
        # ======================
        keypoints = self.extractor.extract_from_frame(frame)
        # print(keypoints)
        if keypoints is None:
            # 在关键点为None时使用一个默认的空关键点数据
            keypoints = {"keypoints": []}
            st.warning("无法提取到有效的关键点数据，正在使用默认的空数据进行处理。")

        # ======================
        # 2️⃣ 姿势分析
        # ======================
        if selected_pose and selected_pose != "自动识别":
            analysis = self.analyzer.compare_poses(keypoints, selected_pose)
        else:
            analysis = self.analyzer.compare_poses(keypoints)

        if "error" in analysis:
            return {
                "success": False,
                "frame": frame,
                "analysis": analysis,
                "user_keypoints": keypoints,
                "std_keypoints": prev_analysis.get("std_keypoints") if prev_analysis else None
            }

        # ======================
        # 3️⃣ 稳定性判断
        # ======================
        stable_update = True
        if prev_analysis and prev_analysis.get("analysis"):
            score_diff = abs(
                analysis.get("score", 0)
                - prev_analysis["analysis"].get("score", 0)
            )
            if score_diff < score_threshold:
                stable_update = False

        # ======================
        # 4️⃣ 决定使用哪一套模板关键点
        # ======================
        if stable_update or not prev_analysis:
            std_pose_name = analysis["standard_pose"]
            std_keypoints = self.analyzer.standard_poses[std_pose_name]["keypoints"]
        else:
            std_keypoints = prev_analysis.get("std_keypoints")

        # # ======================
        # # 5️⃣ 画用户骨架（每帧）
        # # ======================
        # output_frame = self.extractor.draw_skeleton(
        #     frame.copy(),
        #     keypoints
        # )
        output_frame = frame

        # ======================
        # 6️⃣ 画模板骨架（每帧）
        # ======================
        if std_keypoints is not None:
            h, w = output_frame.shape[:2]
            mini_w, mini_h = int(w * 0.28), int(h * 0.38)

            output_frame = self.extractor.draw_skeleton_mini(
                output_frame,
                std_keypoints,
                mini_w,
                mini_h,
                margin=15,
                position="left_bottom"
            )

        # ======================
        # 7️⃣ 叠加评分与目标名称
        # ======================
        score = analysis.get("score", 0)
        score_color = (0, 255, 0) if score >= 70 else (0, 0, 255)

        # 绘制进度条
        bar_width = 300  # 进度条的宽度
        bar_height = 25  # 进度条的高度
        progress = int((score / 100) * bar_width)  # 映射分数到进度条宽度

        # 绘制背景矩形（灰色）
        cv2.rectangle(output_frame, (20, 20), (20 + bar_width, 20 + bar_height), (200, 200, 200), -1)

        # 绘制前景矩形（进度条）
        cv2.rectangle(output_frame, (20, 20), (20 + progress, 20 + bar_height), score_color, -1)

        # ======================
        # 8️⃣ 返回完整状态
        # ======================
        return {
            "success": True,
            "frame": output_frame,
            "analysis": analysis,
            "stable_update": stable_update,
            "user_keypoints": keypoints,
            "std_keypoints": std_keypoints  # ⭐ 关键：缓存模板关键点
        }



def display_camera_suggestions(suggestions):
    st.markdown("### 💡 相机操纵建议")

    for suggestion in suggestions:
        # 判断是否需要修改并设置颜色
        if suggestion["need_modify"]:
            color = "#FF4B4B"  # 红色警告
            icon = "⚠️"
            bg_color = "#FFEAEA"
            status = "待修改"
        else:
            color = "#4CAF50"  # 绿色确认
            icon = "✅"
            bg_color = "#E8F8F0"
            status = "已修改"

        # 使用 HTML 卡片样式展示
        st.markdown(f"""
        <div style="
            border: 1px solid {color};
            border-radius: 10px;
            padding: 15px;
            margin: 5px 0;
            background-color: {bg_color};
        ">
            <span style="font-size:18px; font-weight:bold; color:{color}">{icon} {suggestion['text']}</span>
        </div>
        """, unsafe_allow_html=True)
        # """<p style="margin:5px 0; color:#333; font-size:16px">状态: {status}</p>"""


def display_suggestions_ui(total_suggestions, current_suggestions):
    # st.markdown("### 💡 姿势建议总览")

    st.markdown("### 💡 姿势建议")
    # 获取实时建议的ID集合
    current_ids = {s['id'] for s in current_suggestions}

    # 遍历总建议
    for s in total_suggestions:
        if s['id'] in current_ids:
            # 未实现建议 → 红色警示
            color = "#FF4B4B"
            icon = "⚠️"
            bg_color = "#FFEAEA"
        else:
            # 已实现建议 → 绿色 ✅
            color = "#4CAF50"
            icon = "✅"
            bg_color = "#E8F8F0"

        # 使用 HTML 卡片样式展示
        st.markdown(f"""
        <div style="
            border: 1px solid {color};
            border-radius: 10px;
            padding: 15px;
            margin: 5px 0;
            background-color: {bg_color};
        ">
            <span style="font-size:18px; font-weight:bold; color:{color}">{icon} {s['id']}</span>
            <p style="margin:5px 0; color:#333; font-size:16px">{s['text']}</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    if "realtime_running" not in st.session_state:
        st.session_state.realtime_running = False

    # 应用标题
    st.markdown('<h1 class="main-header">📸 拍照提示助手</h1>', unsafe_allow_html=True)
    # st.markdown("拍摄你的姿势，与标准姿势对比，获取专业的姿势指导建议！")

    # 初始化应用
    try:
        app = PoseCoachApp()
    except Exception as e:
        st.error(f"初始化应用出错: {e}")
        st.info("请确保已运行: python pose_extractor.py")
        return

    # 加载 YOLOv5 模型（使用预训练权重）
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # 主界面
    col1, col2, col3 = st.columns([3, 4, 3])
    with col2:
        # 检查是否有标准姿势
        available_poses = list(app.analyzer.standard_poses.keys())
        # 选择标准姿势
        if available_poses:
            st.subheader("选择目标姿势")

            pose_options = ["自动识别"] + available_poses
            selected_pose = st.selectbox(
                "选择要对比的标准姿势",
                pose_options,
                format_func=lambda x: f"✨ {x}" if x != "自动识别" else "🤖 自动识别"
            )

            # 显示标准姿势预览
            if selected_pose != "自动识别":
                st.subheader("标准姿势预览")

                # 如果有预览图就显示
                preview_path = Path("standard_poses") / f"{selected_pose}_preview.jpg"
                if preview_path.exists():
                    st.image(str(preview_path), use_column_width=True, caption=f"标准姿势: {selected_pose}")

        suggestion_box = st.empty()
    with col1:

        # st.warning("正在使用摄像头，按下按钮开始实时分析")

        # 按钮在循环外，只出现一次
        start_btn = st.button("开始实时姿势分析", type="primary", key="start_realtime_btn")
        stop_btn = st.button("停止实时分析", key="stop_realtime_btn")  # 注意：按钮不在循环里
        # ctrlc_btn = st.button("📋 复制建议", key="ctrlc_suggestion_btn")
        # ctrls_btn = st.button("📸 保存对比图", key="ctrls_contract_pic_btn")
        # 在循环外初始化标志
        if 'buttons_created' not in st.session_state:
            st.session_state.buttons_created = False

        video_box = st.empty()

        if start_btn:
            # cap = cv2.VideoCapture(0)
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                st.error("无法打开摄像头")
            else:
                # 设置摄像头分辨率
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)  # 设置宽度为1280
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)  # 设置高度为720

                frame_count = 0
                prev_analysis = None  # 保存上一次分析结果
                global_total_suggestions = None
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 停止条件在此检测（读取按钮状态，而不重新创建按钮）
                    if stop_btn:
                        break

                    # 1️⃣ 姿势分析每 FRAME_COUNT_EVERY_PROCESS 帧执行一次
                    if frame_count % FRAME_COUNT_EVERY_PROCESS == 0:
                        # 实时处理
                        result = app.process_realtime_frame(
                            frame,
                            selected_pose if 'selected_pose' in locals() else None,
                            prev_analysis=prev_analysis
                        )
                        if result.get("success"):
                            prev_analysis = result
                            stable_update = result["stable_update"]
                            overlay_frame = result["frame"]
                        else:
                            overlay_frame = frame.copy()
                    else:

                        # 非分析帧使用上一帧的关键点绘制骨架
                        # if prev_analysis:
                        #     overlay_frame = app.extractor.draw_skeleton(frame.copy(),
                        #                                                 prev_analysis["user_keypoints"])
                        # else:
                        #     overlay_frame = frame.copy()
                        if prev_analysis and "user_keypoints" in prev_analysis and prev_analysis["user_keypoints"]:
                            overlay_frame = frame.copy()

                            # 2️⃣ 画模板骨架（关键！）
                            if prev_analysis and prev_analysis.get("std_keypoints") is not None:
                                h, w = overlay_frame.shape[:2]
                                mini_w, mini_h = int(w * 0.28), int(h * 0.38)
                                overlay_frame = app.extractor.draw_skeleton_mini(
                                    overlay_frame,
                                    prev_analysis["std_keypoints"],
                                    mini_w,
                                    mini_h,
                                    margin=15,
                                    position="left_bottom"
                                )

                            # 3️⃣ 画分数和目标姿势
                            if prev_analysis and prev_analysis.get("analysis"):
                                analysis = prev_analysis["analysis"]
                                score = analysis.get("score", 0)
                                score_color = (0, 255, 0) if score >= 70 else (0, 0, 255)
                                # ---- 绘制进度条 ----
                                bar_width = 300  # 进度条的宽度
                                bar_height = 25  # 进度条的高度
                                progress = int((score / 100) * bar_width)  # 映射分数到进度条宽度

                                # 绘制背景矩形（灰色）
                                cv2.rectangle(overlay_frame, (20, 20), (20 + bar_width, 20 + bar_height),
                                              (200, 200, 200),
                                              -1)

                                # 绘制前景矩形（进度条）
                                cv2.rectangle(overlay_frame, (20, 20), (20 + progress, 20 + bar_height), score_color,
                                              -1)

                        else:
                            overlay_frame = frame.copy()

                        # ===== 每帧都显示评分、目标姿势和建议 =====
                        if prev_analysis and prev_analysis.get("analysis"):

                            # 网页建议显示
                            posture_suggestions = analysis.get("suggestions", [])

                            # 获取裁剪和缩放建议
                            camera_suggestion = analyze_crop_and_zoom(frame, PoseExtractor().extract_from_frame(frame), yolo_model)

                            suggestions = []

                            # print(camera_suggestion)
                            # 裁剪建议永远放第一个
                            # if crop_suggestion:
                            #     print(crop_suggestion)
                                # suggestions.append(crop_suggestion)

                            # 再加入你已有的姿态建议
                            suggestions.extend(posture_suggestions)

                            suggestions = suggestions[:3]

                            with suggestion_box.container():
                                display_camera_suggestions(camera_suggestion)
                                if not suggestions:
                                    st.success("姿势良好，请继续保持")
                                else:

                                    if global_total_suggestions == None:
                                        global_total_suggestions = suggestions
                                    # with col2:

                                    display_suggestions_ui(global_total_suggestions, suggestions)

                    # ===== 显示摄像头画面 =====
                    video_box.image(overlay_frame, channels="BGR")

                    frame_count += 1

                cap.release()
                st.stop()

    with col3:
        st.subheader("📚 标准姿势库")

        # if not available_poses:
        #     st.info("👈 请先运行骨架提取工具")
        #     st.code("python pose_extractor.py")
        # else:
        #     st.success(f"✅ 已加载 {len(available_poses)} 个标准姿势")

        # 显示所有标准姿势
        for pose_name in available_poses:
            with st.expander(f"姿势: {pose_name}"):

                preview_path = Path("standard_poses") / f"{pose_name}_preview.jpg"
                json_path = Path("standard_poses") / f"{pose_name}.json"

                tab1, tab2 = st.tabs(["预览图", "关键点数据"])

                with tab1:
                    if preview_path.exists():
                        st.image(str(preview_path), use_container_width=True, caption=f"姿势: {pose_name}")
                    else:
                        st.info("没有找到预览图")

                with tab2:
                    if json_path.exists():
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            st.json(data)
                        except Exception as e:
                            st.error(f"读取 JSON 失败: {e}")
                    else:
                        st.info("没有找到 JSON 文件")


if __name__ == "__main__":
    main()