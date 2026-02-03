# app.py - AI姿势教练主应用
import queue

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
import threading
from vosk import Model, KaldiRecognizer
import sounddevice as sd

# 导入我们的模块
try:
    # from pose_extractor import PoseExtractor
    from realtime_extractor import PoseExtractor
    from pose_analyzer import PoseAnalyzer
    from simple_avatar import SimpleAvatar
    from CompositionAnalyzer import analyze_crop_and_zoom, compute_bbox
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

def draw_soft_bbox(img, bbox,
                   color=(0, 180, 255),
                   alpha=0.25,
                   radius=18,
                   thickness=4):
    overlay = img.copy()

    x1, y1, x2, y2 = map(int, list(bbox))

    # 四条边（不封死，留呼吸感）
    cv2.line(overlay, (x1+radius, y1), (x2-radius, y1), color, thickness)
    cv2.line(overlay, (x1+radius, y2), (x2-radius, y2), color, thickness)
    cv2.line(overlay, (x1, y1+radius), (x1, y2-radius), color, thickness)
    cv2.line(overlay, (x2, y1+radius), (x2, y2-radius), color, thickness)

    # 四个角
    cv2.ellipse(overlay, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)

    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

def draw_center_indicator(img, center,
                          size=16,
                          color=(255, 200, 0)):
    cx, cy = [int(i) for i in center]

    # 外圈
    cv2.circle(img, (cx, cy), size, color, 2)
    # 中心点
    cv2.circle(img, (cx, cy), 3, color, -1)

    # 十字
    cv2.line(img, (cx-size, cy), (cx+size, cy), color, 1)
    cv2.line(img, (cx, cy-size), (cx, cy+size), color, 1)

def draw_direction_arrow(img, from_pt, to_pt,
                         color=(0, 180, 255)):
    cv2.arrowedLine(
        img,
        tuple(map(int, from_pt)),
        tuple(map(int, to_pt)),
        color,
        2,
        tipLength=0.25
    )

def draw_score_bar(img, score,
                   pos=(20, 20),
                   size=(300, 26)):
    x, y = pos
    w, h = size

    bg = img.copy()
    fg = img.copy()

    cv2.rectangle(bg, (x+2, y+2), (x+w+2, y+h+2), (0,0,0), -1)
    cv2.rectangle(bg, (x, y), (x+w, y+h), (220,220,220), -1)

    fill_w = int(w * score / 100)
    color = (0,180,0) if score >= 70 else (0,0,180)
    cv2.rectangle(fg, (x, y), (x+fill_w, y+h), color, -1)

    img[:] = cv2.addWeighted(bg, 0.6, img, 0.4, 0)
    img[:] = cv2.addWeighted(fg, 0.9, img, 0.1, 0)

    cv2.putText(img, f"{int(score)}",
                (x+w+10, y+h-6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (60,60,60), 2)


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
            bbox,
            selected_pose=None,
            prev_analysis=None,
            score_threshold=SCORE_DIFF_THRESHOLD,
    ):
        """
        实时处理单帧：
        - 原图：用户骨架
        - 左下角：目标姿势骨架（始终显示）
        """

        height, width = frame.shape[:2]
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
        # 构图可视化（美观版）
        # ======================

        output_frame = draw_soft_bbox(
            output_frame,
            bbox["bbox"]
        )

        draw_center_indicator(
            output_frame,
            bbox["center"],
        )

        cx, cy = bbox["center"]
        ox, oy = width/2, height/2
        # 如果需要引导移动
        if abs(cx - ox) > 10 or abs(cy - oy) > 10:
            draw_direction_arrow(
                output_frame,
                (width/2, height/2),
                (cx, cy)
            )

        # ======================
        # 7️⃣ 叠加评分与目标名称
        # ======================
        score = analysis.get("score", 0)
        # score_color = (0, 255, 0) if score >= 70 else (0, 0, 255)
        draw_score_bar(output_frame, score)

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
current_suggestions = []

def get_main_voice_suggestion():
    print(current_suggestions)
    if not current_suggestions:
        return "你的姿势整体很好，可以保持"

    # 只播报第一条（最重要）
    return current_suggestions[0]["text"][1:]

def display_camera_suggestions(suggestions):
    need_modify = [s for s in suggestions if s["need_modify"]]
    modified = [s for s in suggestions if not s["need_modify"]]

    main_camera_suggestion = need_modify[0] if need_modify else None
    st.markdown("### 💡 相机操纵建议")

    if main_camera_suggestion:
        s = main_camera_suggestion
        st.markdown(f"""
        <div style="
            border: 2px solid #FF4B4B;
            border-radius: 12px;
            padding: 18px;
            background-color: #FFEAEA;
        ">
            <div style="font-size:20px; font-weight:600; color:#FF4B4B">
                ⚠️ 当前需要调整
            </div>
            <p style="margin-top:8px; font-size:17px; color:#333">
                {s['text']}
            </p>
            <p style="font-size:14px; color:#888">
                请优先完成此项，其余操纵建议已隐藏
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ 当前相机角度已符合所有建议")
    with st.expander("📋 查看详情", expanded=False):

        # 其余未修改的相机建议
        if len(need_modify) > 1:
            st.markdown("#### ⚠️ 其他待修改的相机建议")
            for s in need_modify[1:]:
                st.markdown(f"""
                <div style="
                    border: 1px solid #FFB3B3;
                    border-radius: 8px;
                    padding: 12px;
                    margin: 6px 0;
                    background-color: #FFF3F3;
                ">
                    <b style="color:#FF4B4B">⚠️ 待修改</b>
                    <p style="margin:4px 0">{s['text']}</p>
                </div>
                """, unsafe_allow_html=True)

        # 已修改的相机建议
        if modified:
            st.markdown("#### ✅ 已修改的相机建议")
            for s in modified:
                st.markdown(f"""
                <div style="
                    border: 1px solid #4CAF50;
                    border-radius: 8px;
                    padding: 10px;
                    margin: 6px 0;
                    background-color: #E8F8F0;
                ">
                    <b style="color:#4CAF50">✅ 已修改</b>
                    <p style="margin:4px 0">{s['text']}</p>
                </div>
                """, unsafe_allow_html=True)

    # st.markdown("### 💡 相机操纵建议")
    #
    # for suggestion in suggestions:
    #     # 判断是否需要修改并设置颜色
    #     if suggestion["need_modify"]:
    #         color = "#FF4B4B"  # 红色警告
    #         icon = "⚠️"
    #         bg_color = "#FFEAEA"
    #         status = "待修改"
    #     else:
    #         color = "#4CAF50"  # 绿色确认
    #         icon = "✅"
    #         bg_color = "#E8F8F0"
    #         status = "已修改"
    #
    #     # 使用 HTML 卡片样式展示
    #     st.markdown(f"""
    #     <div style="
    #         border: 1px solid {color};
    #         border-radius: 10px;
    #         padding: 15px;
    #         margin: 5px 0;
    #         background-color: {bg_color};
    #     ">
    #         <span style="font-size:18px; font-weight:bold; color:{color}">{icon} {suggestion['text']}</span>
    #     </div>
    #     """, unsafe_allow_html=True)
        # """<p style="margin:5px 0; color:#333; font-size:16px">状态: {status}</p>"""


def display_suggestions_ui(total_suggestions, current_suggestions):
    current_ids = {s['id'] for s in current_suggestions}

    unfixed = [s for s in total_suggestions if s['id'] in current_ids]
    fixed = [s for s in total_suggestions if s['id'] not in current_ids]
    main_suggestion = unfixed[0] if unfixed else None

    st.markdown("### 💡 姿势建议")

    if main_suggestion:
        s = main_suggestion
        st.markdown(f"""
        <div style="
            border: 2px solid #FF4B4B;
            border-radius: 12px;
            padding: 18px;
            background-color: #FFEAEA;
        ">
            <div style="font-size:20px; font-weight:600; color:#FF4B4B">
                ⚠️ {s['id']}
            </div>
            <p style="margin-top:8px; font-size:17px; color:#333">
                {s['text']}
            </p>
            <p style="font-size:14px; color:#888">
                请优先完成此项，其余建议可在下方查看
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ 当前姿势已满足所有建议")

    with st.expander("📋 查看详情", expanded=False):

        # 其余未改正建议
        if len(unfixed) > 1:
            st.markdown("#### ⚠️ 其他未改正建议")
            for s in unfixed[1:]:
                st.markdown(f"""
                <div style="
                    border: 1px solid #FFB3B3;
                    border-radius: 8px;
                    padding: 12px;
                    margin: 6px 0;
                    background-color: #FFF3F3;
                ">
                    <b style="color:#FF4B4B">⚠️ {s['id']}</b>
                    <p style="margin:4px 0">{s['text']}</p>
                </div>
                """, unsafe_allow_html=True)

        # 已改正建议
        if fixed:
            st.markdown("#### ✅ 已改正建议")
            for s in fixed:
                st.markdown(f"""
                <div style="
                    border: 1px solid #4CAF50;
                    border-radius: 8px;
                    padding: 10px;
                    margin: 6px 0;
                    background-color: #E8F8F0;
                ">
                    <b style="color:#4CAF50">✅ {s['id']}</b>
                    <p style="margin:4px 0">{s['text']}</p>
                </div>
                """, unsafe_allow_html=True)

    # st.markdown("### 💡 姿势建议")
    #
    #
    # # 获取实时建议的ID集合
    # current_ids = {s['id'] for s in current_suggestions}
    #
    # # 遍历总建议
    # for s in total_suggestions:
    #     if s['id'] in current_ids:
    #         # 未实现建议 → 红色警示
    #         color = "#FF4B4B"
    #         icon = "⚠️"
    #         bg_color = "#FFEAEA"
    #     else:
    #         # 已实现建议 → 绿色 ✅
    #         color = "#4CAF50"
    #         icon = "✅"
    #         bg_color = "#E8F8F0"
    #
    #     # 使用 HTML 卡片样式展示
    #     st.markdown(f"""
    #     <div style="
    #         border: 1px solid {color};
    #         border-radius: 10px;
    #         padding: 15px;
    #         margin: 5px 0;
    #         background-color: {bg_color};
    #     ">
    #         <span style="font-size:18px; font-weight:bold; color:{color}">{icon} {s['id']}</span>
    #         <p style="margin:5px 0; color:#333; font-size:16px">{s['text']}</p>
    #     </div>
    #     """, unsafe_allow_html=True)


def main():
    from VoiceAssistent import VoiceAssistant

    voice = VoiceAssistant(
        model_path="models/vosk-model-small-cn-0.22",
        get_suggestion_func=get_main_voice_suggestion,
        cooldown=6.0,  # 防止太吵
    )

    voice.start()

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
    yolo_model.classes = [0]
    yolo_model.conf = 0.4
    yolo_model.iou = 0.5
    yolo_model.max_det = 1
    # print(next(yolo_model.parameters()).device)

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

        camera_suggestion = []

        if start_btn:
            # cap = cv2.VideoCapture(0)
            cap = cv2.VideoCapture(3)
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

                    height, width = frame.shape[:2]
                    bbox_result = compute_bbox(frame, PoseExtractor().extract_from_frame(frame), yolo_model)

                    cached_bbox = {
                        "center": bbox_result["target_center"],
                        "bbox": bbox_result["bbox"],
                        "scale": bbox_result["scale"],
                    }

                    # print(cached_bbox)

                    # 1️⃣ 姿势分析每 FRAME_COUNT_EVERY_PROCESS 帧执行一次
                    if frame_count % FRAME_COUNT_EVERY_PROCESS == 0:
                        camera_suggestion = analyze_crop_and_zoom(frame, PoseExtractor().extract_from_frame(frame),
                                                                  yolo_model)
                        # 实时处理
                        result = app.process_realtime_frame(
                            frame,
                            cached_bbox,
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
                                # score_color = (0, 255, 0) if score >= 70 else (0, 0, 255)

                                overlay_frame = draw_soft_bbox(
                                    overlay_frame,
                                    cached_bbox["bbox"]
                                )

                                draw_center_indicator(
                                    overlay_frame,
                                    cached_bbox["center"],
                                )

                                cx, cy = cached_bbox["center"]
                                ox, oy = width/2, height/2
                                # 如果需要引导移动
                                if abs(cx - ox) > 10 or abs(cy - oy) > 10:
                                    draw_direction_arrow(
                                        overlay_frame,
                                        (ox, oy),
                                        (cx, cy)
                                    )

                                # ---- 绘制进度条 ----

                                draw_score_bar(overlay_frame, score)

                        else:
                            overlay_frame = frame.copy()

                        # ===== 每帧都显示评分、目标姿势和建议 =====
                        if prev_analysis and prev_analysis.get("analysis"):

                            # 网页建议显示
                            posture_suggestions = analysis.get("suggestions", [])

                            # 获取裁剪和缩放建议

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
                                st.markdown(f"""
                                    <div style="
                                        border: 1px solid #4DA3FF;
                                        border-radius: 10px;
                                        padding: 15px;
                                        margin: 5px 0;
                                        background-color: #EAF4FF;
                                    ">
                                        <span style="font-size:18px; font-weight:bold; color:#4DA3FF">
                                            建议缩放：{cached_bbox['scale']}倍
                                        </span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                display_camera_suggestions(camera_suggestion)
                                if not suggestions:
                                    st.success("姿势良好，请继续保持")
                                else:

                                    if global_total_suggestions == None:
                                        global_total_suggestions = suggestions
                                    # with col2:

                                    display_suggestions_ui(global_total_suggestions, suggestions)

                                    current_suggestions.clear()
                                    current_suggestions.extend(suggestions)  # ✅ 修改原列表内容
                                    # print(current_suggestions)

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