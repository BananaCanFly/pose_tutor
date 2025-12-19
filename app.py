# app.py - AI姿势教练主应用
import streamlit as st
import cv2
import tempfile
import numpy as np
import os
from pathlib import Path
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
except ImportError as e:
    st.error(f"导入模块出错: {e}")
    st.stop()

SCORE_DIFF_THRESHOLD = 5
FRAME_COUNT_EVERY_PROCESS = 5

# 页面配置
st.set_page_config(
    page_title="拍照姿势提示",
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
        if keypoints is None:
            return {
                "success": False,
                "frame": frame,
                "analysis": None,
                "user_keypoints": None,
                "std_keypoints": prev_analysis.get("std_keypoints") if prev_analysis else None
            }

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

        # ======================
        # 5️⃣ 画用户骨架（每帧）
        # ======================
        output_frame = self.extractor.draw_skeleton(
            frame.copy(),
            keypoints
        )

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

        cv2.putText(
            output_frame,
            f"Score: {score}/100",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            score_color,
            2
        )

        cv2.putText(
            output_frame,
            f"Target Pose: {analysis['standard_pose']}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

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


def display_part_analysis(analysis_data):
    """显示身体部位详细分析"""
    if not analysis_data:
        return

    # 中文部位名称映射
    part_names = {
        "face": "面部",
        "shoulders": "肩膀",
        "elbows": "手肘",
        "wrists": "手腕",
        "hands": "手部",
        "hips": "髋部",
        "knees": "膝盖",
        "ankles": "脚踝",
        "feet": "脚部",
        "other": "其他"
    }

    st.subheader("🔍 身体部位分析")

    # 使用列布局
    cols = st.columns(3)
    col_idx = 0

    for part, data in analysis_data.items():
        if part in ["total_points", "avg_distance", "max_distance"]:
            continue

        chinese_name = part_names.get(part, part)
        accuracy = data.get("accuracy_rate", 0)
        avg_distance = data.get("avg_distance", 0)

        with cols[col_idx]:
            # 根据准确率显示不同颜色的指标
            if accuracy >= 90:
                color = "#4CAF50"
                emoji = "✅"
                badge_class = "success-badge"
            elif accuracy >= 70:
                color = "#FF9800"
                emoji = "⚠️"
                badge_class = "warning-badge"
            else:
                color = "#F44336"
                emoji = "❌"
                badge_class = "error-badge"

            st.markdown(f"""
            <div class="part-analysis-card" style="border-left-color: {color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <strong>{emoji} {chinese_name}</strong>
                    <span class="joint-difference-badge {badge_class}">{accuracy:.1f}%</span>
                </div>
                <div style="font-size: 0.9rem; color: #666;">
                    • 平均偏差: {avg_distance:.3f}<br>
                    • 检测点: {data.get('total_points', 0)}个<br>
                    • 需调整: {data.get('points_need_adjustment', 0)}个
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_idx = (col_idx + 1) % 3


def display_joint_differences(differences):
    """显示关节差异详情"""
    if not differences:
        return

    st.write("**关键关节差异分析**:")

    # 重要关节的优先级
    important_joints = ["shoulders", "hips", "elbows", "knees"]
    secondary_joints = ["wrists", "ankles"]

    # 按优先级分组显示
    for joint_group in [important_joints, secondary_joints]:
        displayed = False

        for joint_name, diff in differences.items():
            part = diff.get("part", "")

            if part in joint_group and diff.get("needs_adjustment", False):
                if not displayed:
                    displayed = True

                distance = diff.get("distance", 0)
                diff_x = diff.get("diff_x", 0)
                diff_y = diff.get("diff_y", 0)

                # 判断偏移方向
                direction = ""
                if abs(diff_x) > abs(diff_y) * 1.5:
                    direction = "偏右" if diff_x > 0 else "偏左"
                elif abs(diff_y) > abs(diff_x) * 1.5:
                    direction = "偏高" if diff_y > 0 else "偏低"
                else:
                    # 对角方向
                    if diff_x > 0 and diff_y > 0:
                        direction = "偏右上"
                    elif diff_x < 0 and diff_y > 0:
                        direction = "偏左上"
                    elif diff_x > 0 and diff_y < 0:
                        direction = "偏右下"
                    else:
                        direction = "偏左下"

                # 获取中文名称
                part_names = {
                    "shoulders": "肩膀",
                    "hips": "髋部",
                    "elbows": "手肘",
                    "knees": "膝盖",
                    "wrists": "手腕",
                    "ankles": "脚踝",
                    "face": "面部",
                    "hands": "手部",
                    "feet": "脚部"
                }

                chinese_part = part_names.get(part, part)

                # 根据距离大小显示不同颜色的标签
                if distance > 0.15:
                    badge_color = "#F44336"
                elif distance > 0.1:
                    badge_color = "#FF9800"
                else:
                    badge_color = "#4CAF50"

                st.markdown(f"""
                <div style="
                    background: {badge_color}10;
                    padding: 0.6rem;
                    border-radius: 6px;
                    margin: 0.3rem 0;
                    border-left: 3px solid {badge_color};
                ">
                    <strong>🔸 {chinese_part}</strong> • 偏移{direction}<br>
                    <span style="font-size: 0.85rem; color: #666;">
                        距离差异: {distance:.3f}
                    </span>
                </div>
                """, unsafe_allow_html=True)



def main():
    if "realtime_running" not in st.session_state:
        st.session_state.realtime_running = False

    # 应用标题
    st.markdown('<h1 class="main-header">📸 拍照姿势提示助手</h1>', unsafe_allow_html=True)
    st.markdown("拍摄你的姿势，与标准姿势对比，获取专业的姿势指导建议！")

    # 初始化应用
    try:
        app = PoseCoachApp()
    except Exception as e:
        st.error(f"初始化应用出错: {e}")
        st.info("请确保已运行: python pose_extractor.py")
        return

    # 侧边栏
    with st.sidebar:
        st.header("🎯 设置")

        # 检查是否有标准姿势
        if not app.analyzer.standard_poses:
            st.warning("⚠️ 还没有标准姿势数据")
            st.info("请先运行: python pose_extractor.py")
            available_poses = []
        else:
            available_poses = list(app.analyzer.standard_poses.keys())
            st.success(f"✅ 已加载 {len(available_poses)} 个标准姿势")

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

        st.markdown("---")
        st.subheader("📖 使用指南")
        st.info("""
        1. 📸 打开摄像头拍摄全身照片
        2. 🎯 选择要对比的标准姿势（或自动识别）
        3. 🔍 点击"分析我的姿势"按钮
        4. 📊 查看分析结果和改进建议
        """)

        st.markdown("---")
        st.subheader("📝 拍照建议")
        st.write("""
        ✅ 好的照片应该:
        - 光线充足，清晰可见
        - 全身入镜，姿势完整
        - 正面或侧面站立

        ❌ 避免:
        - 太暗或模糊
        - 只拍到部分身体
        - 遮挡身体的衣服
        """)

    # 主界面
    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("实时摄像头分析")
        run_realtime = st.checkbox("开启实时摄像头模式", value=False)

        if run_realtime:
            st.warning("正在使用摄像头，按下按钮开始实时分析")

            # 按钮在循环外，只出现一次
            start_btn = st.button("开始实时姿势分析", type="primary", key="start_realtime_btn")
            stop_btn = st.button("停止实时分析", key="stop_realtime_btn")  # 注意：按钮不在循环里
            # ctrlc_btn = st.button("📋 复制建议", key="ctrlc_suggestion_btn")
            # ctrls_btn = st.button("📸 保存对比图", key="ctrls_contract_pic_btn")
            # 在循环外初始化标志
            if 'buttons_created' not in st.session_state:
                st.session_state.buttons_created = False

            video_box = st.empty()
            suggestion_box = st.empty()

            if start_btn:
                # cap = cv2.VideoCapture(0)
                cap = cv2.VideoCapture(1)
                if not cap.isOpened():
                    st.error("无法打开摄像头")
                else:
                    frame_count = 0
                    prev_analysis = None  # 保存上一次分析结果
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

                                # 1️⃣ 画用户骨架
                                if prev_analysis and prev_analysis.get("user_keypoints") is not None:
                                    overlay_frame = app.extractor.draw_skeleton(
                                        overlay_frame,
                                        prev_analysis["user_keypoints"]
                                    )

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

                                    cv2.putText(overlay_frame, f"Score: {score}/100", (20, 40),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, score_color, 2)
                                    cv2.putText(overlay_frame, f"Target Pose: {analysis.get('standard_pose', '--')}",
                                                (20, 80),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                            else:
                                overlay_frame = frame.copy()

                            # ===== 每帧都显示评分、目标姿势和建议 =====
                            if prev_analysis and prev_analysis.get("analysis"):
                                analysis = prev_analysis["analysis"]
                                score = analysis.get("score", 0)
                                score_color = (0, 255, 0) if score >= 70 else (0, 0, 255)

                                cv2.putText(overlay_frame, f"Score: {score}/100", (20, 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, score_color, 2)
                                cv2.putText(overlay_frame, f"Target Pose: {analysis.get('standard_pose', '--')}",
                                            (20, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                                # 网页建议显示
                                suggestions = analysis.get("suggestions", [])
                                with suggestion_box.container():
                                    st.markdown("### 💡 实时姿势建议")
                                    if not suggestions:
                                        st.success("姿势良好，请继续保持")
                                    else:
                                        for i, s in enumerate(suggestions[:3], 1):
                                            st.warning(f"建议 {i}：{s}")

                                    # 详细分析
                                    with st.expander("📊 查看详细分析", expanded=False):
                                        st.write(f"**匹配的标准姿势**: {analysis['standard_pose']}")
                                        st.write(f"**是否合格**: {'✅ 是' if analysis['is_good'] else '❌ 否'}")

                                        # 显示身体部位分析
                                        if "detailed_analysis" in analysis:
                                            display_part_analysis(analysis["detailed_analysis"])

                                        # 显示关节差异
                                        if "differences" in analysis and analysis["differences"]:
                                            display_joint_differences(analysis["differences"])

                                        # 显示关键点统计
                                        if "user_keypoints" in result:
                                            st.write(f"**检测到关键点**: {len(result['user_keypoints'])}个")

                                    # 导出结果选项
                                    with st.expander("💾 导出分析结果", expanded=False):
                                        col_exp1, col_exp2 = st.columns(2)
                                        # 在循环内
                                        if not st.session_state.buttons_created:
                                            ctrlc_btn = st.button("📋 复制建议", key="ctrlc_suggestion_btn")
                                            ctrls_btn = st.button("📸 保存对比图", key="ctrls_contract_pic_btn")
                                            st.session_state.buttons_created = True

                                        with col_exp1:
                                            if ctrlc_btn:
                                                suggestions_text = "\n".join(
                                                    [f"{i + 1}. {s}" for i, s in
                                                     enumerate(analysis.get("suggestions", []))])
                                                st.code(suggestions_text)
                                        with col_exp2:
                                            if ctrls_btn:
                                                # 保存对比图
                                                import datetime
                                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                                save_path = f"outputs/comparison_{timestamp}.jpg"
                                                cv2.imwrite(save_path,
                                                            cv2.cvtColor(result["comparison_image"], cv2.COLOR_RGB2BGR))
                                                st.success(f"已保存到: {save_path}")

                        # ===== 显示摄像头画面 =====
                        video_box.image(overlay_frame, channels="BGR", use_container_width=True)

                        frame_count += 1

                    cap.release()
                st.stop()

    with col2:
        st.subheader("📚 标准姿势库")

        if not available_poses:
            st.info("👈 请先运行骨架提取工具")
            st.code("python pose_extractor.py")
        else:
            st.success(f"✅ 已加载 {len(available_poses)} 个标准姿势")

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