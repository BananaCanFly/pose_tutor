# crop_advisor.py
import cv2
import numpy as np

import mediapipe as mp
import torch

# MediaPipe Pose 中的关键关节（避免裁切）
CRITICAL_JOINT_IDS = {
    13: "左手肘",
    14: "右手肘",
    15: "左手腕",
    16: "右手腕",
    25: "左膝盖",
    26: "右膝盖",
    27: "左脚踝",
    28: "右脚踝",
}
mp_pose = mp.solutions.pose

def mask_hip_below(frame, keypoints):
    """
    将图像中髋部以下的部分设置为黑色，髋部以上保留。

    参数:
    - frame: 输入图像（BGR格式）
    - keypoints: 包含人体关键点的字典，假设有 'hip_left' 和 'hip_right'

    返回:
    - 修改后的图像
    """

    # 获取图像的高和宽
    height, width = frame.shape[:2]

    mp_pose = mp.solutions.pose
    # 假设我们有左右髋部的 y 坐标

    hip_left = keypoints[mp_pose.PoseLandmark.LEFT_HIP]
    hip_right = keypoints[mp_pose.PoseLandmark.RIGHT_HIP]

    if hip_left is None or hip_right is None:
        print("髋部关键点缺失!")
        return frame

    # 获取髋部的平均 y 坐标
    hip_y = (hip_left['y'] + hip_right['y']) / 2 * height  # 将 [0, 1] 范围的 y 坐标转为实际像素

    # 将髋部以下的区域设置为黑色
    frame[int(hip_y):, :] = 0  # 将 y 坐标下方的所有像素置为黑色（BGR中为0）

    return frame

def get_box(frame, results):
    """
    计算人物的最高点（边界框的顶部中心点）
    """

    # 获取检测结果（boxes, labels, scores）
    boxes = results.xywh[0][:, :-2]  # 获取所有的边界框
    scores = results.xywh[0][:, -2]  # 得到置信度
    labels = results.xywh[0][:, -1]  # 类别名称

    best_box = None
    best_score = -1.0

    PERSON_CLASS_ID = 0
    for box, score, label in zip(boxes, scores, labels):
        if int(label) != PERSON_CLASS_ID:
            continue

        if score > best_score:
            best_score = score
            best_box = box

    return best_box

def get_highest_point(frame, results):
    """
    计算人物的最高点（边界框的顶部中心点）
    """

    # 获取检测结果（boxes, labels, scores）
    boxes = results.xywh[0][:, :-2]  # 获取所有的边界框
    scores = results.xywh[0][:, -2]  # 得到置信度
    labels = results.xywh[0][:, -1]  # 类别名称

    # 提取每个边界框的上下左右坐标
    highest_point = 1
    for box, score, label in zip(boxes, scores, labels):
        # print(box, score, label)
        if label != 0 or score < 0.5:  # 如果置信度低于0.5，忽略
            continue

        # 获取边界框的坐标 (x_center, y_center, width, height)
        x_center, y_center, w, h = box

        # 计算边界框的顶部中心点
        highest_point = (y_center - h / 2)/frame.shape[0]  # 顶部的y坐标
        # top_x = x_center  # 顶部的x坐标与中心相同

        # 记录最高点的位置
        # if highest_point is None or top_y < highest_point[1]:
            # highest_point = (top_x, top_y)

    return highest_point


def get_edge_point(frame, results):
    """
    计算人物的最高点（边界框的顶部中心点）
    """

    # 获取检测结果（boxes, labels, scores）
    boxes = results.xywh[0][:, :-2]  # 获取所有的边界框
    scores = results.xywh[0][:, -2]  # 得到置信度
    labels = results.xywh[0][:, -1]  # 类别名称

    # 提取每个边界框的上下左右坐标
    left_point = 0
    right_point = 1
    for box, score, label in zip(boxes, scores, labels):
        # print(box, score, label)
        if label != 0 or score < 0.5:  # 如果置信度低于0.5，忽略
            continue

        # 获取边界框的坐标 (x_center, y_center, width, height)
        x_center, y_center, w, h = box

        # 计算边界框的顶部中心点
        left_point = (x_center - w / 2)/frame.shape[1]  # 顶部的y坐标
        right_point = (x_center + w / 2)/frame.shape[1]
        # top_x = x_center  # 顶部的x坐标与中心相同

        # 记录最高点的位置
        # if highest_point is None or top_y < highest_point[1]:
            # highest_point = (top_x, top_y)

    return [left_point, right_point]


def estimate_knee_height(landmarks, visibility_threshold=0.5):
    """
    根据肩部和髋部的关键点推测膝盖高度
    :param landmarks: Mediapipe Pose 模型返回的关键点列表
    :param visibility_threshold: 关键点可见性阈值，默认为 0.5
    :return: 推测的膝盖高度（垂直坐标比例）
    """

    # 提取肩部和髋部关键点
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    # print("左脚踝，右脚踝，左膝盖，右膝盖",landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]['y'],landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]['y'],
    #       landmarks[mp_pose.PoseLandmark.LEFT_KNEE]['y'], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]['y'])

    # print(left_shoulder, right_shoulder, left_hip, right_hip)

    # 可视性判断
    if left_shoulder["visibility"] > visibility_threshold and right_shoulder["visibility"] > visibility_threshold and \
            left_hip["visibility"] > visibility_threshold and right_hip["visibility"] > visibility_threshold:

        # 计算肩部和髋部之间的垂直距离（Y坐标的平均值）
        shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
        hip_y = (left_hip["y"] + right_hip["y"]) / 2

        # 膝盖高度假设在髋部到肩部之间的70%位置
        knee_y = hip_y + abs(shoulder_y - hip_y) * 0.7  # 膝盖位置是髋部到肩部的70%高度

        return knee_y

    # 如果关键点不可见，返回 None
    return None


# def analyze_crop_and_zoom(frame, keypoints, yolo_box):
#     """
#     分析拍照建议，包括头部留白、膝盖脚踝裁剪、胳膊显示、人物居中等
#     参数:
#     - frame: 当前帧图像（OpenCV格式）
#     - keypoints: 人物的关键点列表，包含头部、肩膀、肘部、膝盖、脚踝等部位的坐标
#
#     返回:
#     - dict: 包含裁剪和缩放建议的信息
#     """
#
#     # print(keypoints)
#     suggestions = []
#     height, width = frame.shape[:2]
#
#     # 获取关键点
#     # head_y = keypoints[0]['y']  # 头部位置
#     # shoulder_left_y = keypoints[11]['y']  # 左肩
#     # shoulder_right_y = keypoints[12]['y']  # 右肩
#     knee_left_y = keypoints[25]['y']  # 左膝
#     knee_right_y = keypoints[26]['y']  # 右膝
#     ankle_left_y = keypoints[27]['y']  # 左脚踝
#     ankle_right_y = keypoints[28]['y']  # 右脚踝
#     # elbow_left_x = keypoints[13]['x']  # 左肘
#     # elbow_right_x = keypoints[14]['x']  # 右肘
#     # wrist_left_y = keypoints[15]['y']  # 左手腕
#     # wrist_right_y = keypoints[16]['y']  # 右手腕
#
#     # 1. 先降分辨率（非常关键）
#
#     frame = cv2.resize(frame, (320, 320))
#     edge_frame = mask_hip_below(frame, keypoints)
#
#     elbow_left_x, foot_y, elbow_right_x, head_y = yolo_box
#     # head_y = get_highest_point(edge_frame, results)
#     # elbow_left_x, elbow_right_x = get_edge_point(frame, results)
#
#     # 计算头部上方的留白（理想高度为头部的20%-30%）
#     # print("头顶高度:", get_highest_point(edge_frame, results))
#     head_height = abs(keypoints[0]['y'] - head_y)  # 头部高度
#     head_margin = head_height * 0.4  # 留白高度（头部高度的20%）
#
#     # print(head_height, head_margin)
#     # 判断头部是否靠近画面顶部
#     if head_y < head_margin:
#         suggestions.append(
#             {"id": "留白", "text": "⬆ 请向上移动一点（头顶空间不足）", "need_modify": True}
#         )
#     # else:
#     #     suggestions.append(
#     #         {"id": "留白", "text": "✅ 头顶留白足够", "need_modify": False}
#     #     )
#
#
#     # if knee_y > 0.95:
#     if 1 > knee_left_y > 0.95 or 1 > knee_right_y > 0.95:
#         suggestions.append(
#             {"id": "关节", "text": "⬆ 请向上移动一点（膝盖部分被裁剪）", "need_modify": True}
#         )
#     elif 1.02 > ankle_left_y > 0.95 or 1.02 > ankle_right_y > 0.95:
#         suggestions.append(
#             {"id": "关节", "text": "⬇ 请向下移动一点（脚踝部分被裁剪）", "need_modify": True}
#         )
#     # else:
#     #     suggestions.append(
#     #         {"id": "关节", "text": "✅ 关节完整显示", "need_modify": False}
#     #     )
#
#     # 判断胳膊是否完全可见
#     # if 0.02<elbow_left_x<0.98 and 0.02<elbow_right_x<0.98:
#     #     suggestions.append(
#     #         {"id": "胳膊", "text": "胳膊已完整露出，无需调整", "need_modify": False}
#     #     )
#     # else:
#     #     suggestions.append(
#     #         {"id": "胳膊", "text": "建议调整，胳膊部分不可见，可能需要缩放或调整角度", "need_modify": True}
#     #     )
#     if elbow_left_x<0.02 and elbow_right_x>0.98:
#         suggestions.append(
#             {"id": "胳膊", "text": "⬆/⬇ 缩放画面（两侧胳膊均部分不可见）", "need_modify": True}
#         )
#     elif elbow_left_x<0.02:
#         suggestions.append(
#             {"id": "胳膊", "text": "⬅ 请左移一点（左侧胳膊不可见）", "need_modify": True}
#         )
#     elif elbow_right_x>0.98:
#         suggestions.append(
#             {"id": "胳膊", "text": "➡ 请右移一点（右侧胳膊不可见）", "need_modify": True}
#         )
#     # else:
#     #     suggestions.append(
#     #         {"id": "胳膊", "text": "✅ 胳膊完整显示", "need_modify": False}
#     #     )
#
#     # 判断人物是否居中
#     center_x = width // 2
#     head_center_x = (keypoints[0]['x'] + keypoints[1]['x'] + keypoints[2]['x']) / 3
#     shoulder_center_x = (keypoints[1]['x'] + keypoints[2]['x']) / 2
#     person_center_x = (head_center_x + shoulder_center_x) / 2
#     # print(head_center_x, shoulder_center_x, person_center_x, center_x)
#     if abs(person_center_x - 0.5) > 0.1:
#         if person_center_x < 0.5:
#             suggestions.append(
#                 {"id": "中心", "text": "➡ 请右移一点（人物偏左）", "need_modify": True}
#             )
#         else:
#             suggestions.append(
#                 {"id": "中心", "text": "⬅ 请左移一点（人物偏右）", "need_modify": True}
#             )
#     # else:
#     #     suggestions.append(
#     #         {"id": "中心", "text": "✅ 人物居中良好", "need_modify": False}
#     #     )
#
#     # # 判断是否需要缩放（通过肩膀宽度来判断）
#     # shoulder_width = abs(keypoints[1]['x'] - keypoints[2]['x'])  # 计算肩膀宽度
#     # zoom_suggestion = ""
#     # if shoulder_width < width * 0.2:
#     #     zoom_suggestion = "建议放大，人物显得太小"
#     # elif shoulder_width > width * 0.6:
#     #     zoom_suggestion = "建议缩小，人物占据空间过大"
#     return suggestions

def analyze_crop_and_zoom(frame, keypoints, yolo_box):
    """
    智能拍照指导逻辑：检测构图禁忌并给出调整建议
    """
    suggestions = []
    h_img, w_img = frame.shape[:2]

    # 1. 安全解包 YOLO BBox (像素坐标)
    if yolo_box is not None and len(yolo_box) == 4:
        # yolo_box 格式: [x1, y1, x2, y2]
        y_x1, y_y1, y_x2, y_y2 = yolo_box
        # 归一化 YOLO 坐标，方便与 0.0-1.0 比较
        ny1, ny2 = y_y1 / h_img, y_y2 / h_img
        nx1, nx2 = y_x1 / w_img, y_x2 / w_img
    else:
        ny1 = ny2 = nx1 = nx2 = None

    # 2. 辅助函数：安全获取关键点
    def get_kp(idx):
        if idx < len(keypoints):
            return keypoints[idx]['x'], keypoints[idx]['y']
        return None, None

    # 获取核心点
    nose_x, nose_y = get_kp(0)
    lk_x, lk_y = get_kp(25)  # 左膝
    rk_x, rk_y = get_kp(26)  # 右膝
    la_x, la_y = get_kp(27)  # 左踝
    ra_x, ra_y = get_kp(28)  # 右踝

    # --- 策略 A: 头部留白分析 ---
    # 使用 YOLO 的边界框顶部作为“头顶”参考，MediaPipe 鼻子作为参考点
    if ny1 is not None and nose_y is not None:
        head_height_norm = abs(nose_y - ny1)
        # 理想留白：头顶上方应留出约 0.5 到 1.0 个头部高度的空间
        if ny1 < 0.05:
            suggestions.append({"id": "留白", "text": "⬆ 请向上移动镜头（头顶快出界了）", "need_modify": True})
        elif ny1 > 0.3:
            suggestions.append({"id": "留白", "text": "⬇ 请向下移动镜头（头顶留白过多）", "need_modify": True})

    # --- 策略 B: 关节裁剪分析 (构图大忌) ---
    # 摄影原则：不要在关节处裁剪。如果膝盖或脚踝在边缘 5% 范围内，视为裁剪。
    if lk_y is not None:
        max_knee_y = max(lk_y, rk_y)
        max_ankle_y = max(la_y, ra_y)

        if 0.92 < max_knee_y < 0.99:
            suggestions.append({"id": "关节", "text": "⬆ 请稍向上移（不要从膝盖处截断）", "need_modify": True})
        elif 0.92 < max_ankle_y < 0.99:
            suggestions.append({"id": "关节", "text": "⬇ 请稍向下移（不要从脚踝处截断）", "need_modify": True})

    # --- 策略 C: 胳膊与横向空间 ---
    if nx1 is not None:
        # 检测左右出界
        left_out = nx1 < 0.02
        right_out = nx2 > 0.98

        if left_out and right_out:
            suggestions.append({"id": "胳膊", "text": "🔍 请远离一点（身体两侧显示不全）", "need_modify": True})
        elif left_out:
            suggestions.append({"id": "胳膊", "text": "⬅ 请向左移动（左臂出界）", "need_modify": True})
        elif right_out:
            suggestions.append({"id": "胳膊", "text": "➡ 请向右移动（右臂出界）", "need_modify": True})

    # --- 策略 D: 黄金分割与居中 ---
    if nose_x is not None:
        # 计算躯干中心（以鼻子和双肩中心为准）
        person_x_center = nose_x
        offset = person_x_center - 0.5  # 偏离中心的距离

        if abs(offset) > 0.15:
            direction = "⬅ 左" if offset > 0 else "➡ 右"
            suggestions.append({"id": "中心", "text": f"{direction} 移动一点（人物不在中心）", "need_modify": True})

    # --- 策略 E: 姿势评分逻辑补偿 ---
    # 如果没有任何修改建议，添加一个正面反馈
    if not suggestions:
        suggestions.append({"id": "状态", "text": "✅ 构图完美，请保持", "need_modify": False})

    return suggestions


def choose_scale(scale,
                 scale_candidates=(0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0),
                 threshold=0.95):
    """
    从候选缩放比例中选择合适的值。

    逻辑：
    1. 找到不大于 scale 的最大候选值
    2. 如果找不到就取最小候选值
    3. 如果最终结果 >= threshold 临近 1，则直接返回 1
    """
    # 取小于等于 scale 的候选值
    valid_scales = [s for s in scale_candidates if s <= scale]

    # 默认选取
    selected = max(valid_scales) if valid_scales else min(scale_candidates)

    # 如果接近 1，则直接返回 1
    if selected >= threshold:
        return 1.0

    return selected


def compute_bbox_by_mode(base_frame, keypoints, yolo_box, mode="全身像", target_aspect_ratio=None):
    h_img, w_img = base_frame.shape[:2]
    if target_aspect_ratio is None:
        target_aspect_ratio = w_img / h_img

    # --- 1. 获取关键点像素坐标 ---
    mp_coords = []
    if keypoints:
        is_dict = isinstance(keypoints[0], dict)
        for kp in keypoints:
            kx = kp['x'] if is_dict else kp.x
            ky = kp['y'] if is_dict else kp.y
            mp_coords.append((kx * w_img, ky * h_img))

    # --- 2. 构图中心与覆盖范围配置 ---
    # 格式: (中心参考点索引, 覆盖范围参考的关键点数量, 垂直缩放因子)
    # 垂直缩放因子决定了以中心点发散出去的视野大小
    comp_config = {
        "面部特写": {"anchor_idx": [0], "kp_count": 11, "v_scale": 1.5},  # 以鼻子为中心
        "半身像": {"anchor_idx": [11, 12], "kp_count": 25, "v_scale": 1.2},  # 肩部中点(锁骨)
        "全身像": {"anchor_idx": [23, 24], "kp_count": 33, "v_scale": 1.1}  # 胯部中点
    }

    cfg = comp_config.get(mode, comp_config["全身像"])

    # --- 3. 计算中心锚点 (Target Center) ---
    if mp_coords:
        anchors = [mp_coords[i] for i in cfg["anchor_idx"] if i < len(mp_coords)]
        center_x = sum(a[0] for a in anchors) / len(anchors)
        center_y = sum(a[1] for a in anchors) / len(anchors)
    else:
        # 如果没 MP 数据，退而求其次用 YOLO 中心
        if yolo_box:
            center_x = (yolo_box[0] + yolo_box[2]) / 2
            center_y = (yolo_box[1] + yolo_box[3]) / 2
        else:
            return {"mode": mode, "bbox": [0, 0, 1, 1], "scale": 1.0}  # 兜底

    # --- 4. 计算覆盖范围 (BBox Size) ---
    # 获取对应模式的关键点集，计算一个基础跨度
    selected_kp = mp_coords[:cfg["kp_count"]] if mp_coords else []
    if selected_kp:
        xs, ys = zip(*selected_kp)
        raw_w = (max(xs) - min(xs)) * 1.5  # 适当增加宽度留白
        raw_h = (max(ys) - min(ys)) * cfg["v_scale"]
    else:
        # 只有 YOLO 框时的处理
        raw_w = (yolo_box[2] - yolo_box[0]) if yolo_box else w_img
        raw_h = (yolo_box[3] - yolo_box[1]) if yolo_box else h_img

    # --- 5. 按照目标比例锁定最终长宽 ---
    # 确保框不会比目标比例窄
    if raw_w / raw_h < target_aspect_ratio:
        final_h = raw_h
        final_w = final_h * target_aspect_ratio
    else:
        final_w = raw_w
        final_h = final_w / target_aspect_ratio

    # --- 6. 生成最终边界并防止越界 ---
    f_x1 = max(0, center_x - final_w / 2)
    f_y1 = max(0, center_y - final_h / 2)
    f_x2 = min(w_img, f_x1 + final_w)
    f_y2 = min(h_img, f_y1 + final_h)

    # 重新修正因越界导致的位移
    final_w = f_x2 - f_x1
    final_h = f_y2 - f_y1

    return {
        "mode": mode,
        "target_center": (round(center_x / w_img, 4), round(center_y / h_img, 4)),
        "bbox": [round(f_x1 / w_img, 4), round(f_y1 / h_img, 4), round(f_x2 / w_img, 4), round(f_y2 / h_img, 4)],
        "scale": round(h_img / final_h, 1) if final_h > 0 else 1.0
    }



# def compute_bbox_by_mode(base_frame, keypoints, yolo_box, mode="全身像", target_aspect_ratio=None):
#     """
#     智能构图计算：结合 YOLO 稳定性与 MediaPipe 精确度
#     mode: "面部特写", "半身像", "全身像"
#     """
#     # print(f"[构图分析] 当前模式: {mode}")
#     h_img, w_img = base_frame.shape[:2]
#
#     if target_aspect_ratio is None:
#         target_aspect_ratio = w_img / h_img
#
#     # --- 1. 预处理：安全获取关键点像素坐标 ---
#     mp_coords = []
#     if keypoints:
#         try:
#             # 自动识别是字典 kp['x'] 还是对象 kp.x
#             is_dict = isinstance(keypoints[0], dict)
#             for kp in keypoints:
#                 kx = kp['x'] if is_dict else kp.x
#                 ky = kp['y'] if is_dict else kp.y
#                 mp_coords.append((kx * w_img, ky * h_img))
#         except Exception as e:
#             print(f"MediaPipe 数据解析异常: {e}")
#
#     # --- 2. 模式参数配置 (留白比例) ---
#     # 定义：(上留白, 下留白, 左右留白)
#     config = {
#         "面部特写": (0.5, 0.3, 0.4, 11),  # 取前11点
#         "半身像": (0.2, 0.15, 0.2, 25),  # 取前25点
#         "全身像": (0.1, 0.05, 0.1, 33)  # 全取
#     }
#     pad_top, pad_bottom, pad_x, kp_count = config.get(mode, config["全身像"])
#
#     # --- 3. 确定原始边界 (MP 与 YOLO 融合) ---
#     selected_mp = mp_coords[:kp_count] if mp_coords else []
#
#     # 初始化边界为 None
#     mp_x1 = mp_y1 = mp_x2 = mp_y2 = None
#     if selected_mp:
#         xs, ys = zip(*selected_mp)
#         mp_x1, mp_y1, mp_x2, mp_y2 = min(xs), min(ys), max(xs), max(ys)
#
#     # 安全处理 YOLO 框 (防止 TypeError: cannot unpack non-iterable NoneType object)
#     y_x1 = y_y1 = y_x2 = y_y2 = None
#     if yolo_box is not None and len(yolo_box) == 4:
#         y_x1, y_y1, y_x2, y_y2 = yolo_box
#
#     # 逻辑融合：
#     # 如果是面部特写，完全信任 MediaPipe；否则取 MP 和 YOLO 的并集增强稳定性
#     if mode == "面部特写" or y_x1 is None:
#         f_x1, f_y1, f_x2, f_y2 = mp_x1, mp_y1, mp_x2, mp_y2
#     elif mp_x1 is None:
#         f_x1, f_y1, f_x2, f_y2 = y_x1, y_y1, y_x2, y_y2
#     else:
#         # 融合：取两者并集
#         f_x1 = min(mp_x1, y_x1)
#         f_y1 = min(mp_y1, y_y1)
#         f_x2 = max(mp_x2, y_x2)
#         f_y2 = max(mp_y2, y_y2)
#
#     # 兜底：如果所有算法都没抓到人，返回全图
#     if f_x1 is None:
#         return {
#             "mode": mode,
#             "target_center": (0.5, 0.5),
#             "bbox": [0.0, 0.0, 1.0, 1.0],
#             "scale": 1.0
#         }
#
#     # --- 4. 智能留白与纵横比修正 ---
#     box_w, box_h = f_x2 - f_x1, f_y2 - f_y1
#
#     # 应用初始裁剪（带留白）
#     cx1 = max(0, f_x1 - box_w * pad_x)
#     cx2 = min(w_img, f_x2 + box_w * pad_x)
#     cy1 = max(0, f_y1 - box_h * pad_top)
#     cy2 = min(h_img, f_y2 + box_h * pad_bottom)
#
#     # 纵横比锁定逻辑
#     curr_w, curr_h = cx2 - cx1, cy2 - cy1
#     curr_ratio = curr_w / curr_h
#
#     if curr_ratio < target_aspect_ratio:
#         # 太瘦了，补宽度
#         needed_w = curr_h * target_aspect_ratio
#         diff = needed_w - curr_w
#         cx1 -= diff / 2
#         cx2 += diff / 2
#     else:
#         # 太胖了，补高度
#         needed_h = curr_w / target_aspect_ratio
#         diff = needed_h - curr_h
#         cy1 -= diff / 2
#         cy2 += diff / 2
#
#     # 最终像素坐标
#     final_x1, final_y1 = max(0, cx1), max(0, cy1)
#     final_x2, final_y2 = min(w_img, cx2), min(h_img, cy2)
#
#     # --- 6. 构造响应结构 ---
#     final_w = final_x2 - final_x1
#     final_h = final_y2 - final_y1
#
#     # 计算相对于原图的缩放倍率
#     scale = round(h_img / final_h, 1) if final_h > 0 else 1.0
#
#     return {
#         "mode": mode,
#         "target_center": (round((final_x1 + final_w / 2) / w_img, 4), round((final_y1 + final_h / 2) / h_img, 4)),
#         "bbox": [
#             round(final_x1 / w_img, 4),
#             round(final_y1 / h_img, 4),
#             round(final_x2 / w_img, 4),
#             round(final_y2 / h_img, 4)
#         ],
#         "scale": scale,
#     }



# def compute_bbox(base_frame, keypoints, model):
#     """
#     根据现有 analyze_crop_and_zoom 规则
#     返回：新的目标中心点 (cx, cy)，normalized 坐标
#     """
#     frame = base_frame.copy()
#
#     height, width = frame.shape[:2]
#
#     # ========= 原始人物中心 =========
#     head_center_x = (keypoints[mp_pose.PoseLandmark.LEFT_EYE]['x'] + keypoints[mp_pose.PoseLandmark.RIGHT_EYE]['x']) / 2
#     shoulder_center_x = (keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER]['x'] + keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER]['x']) / 2
#
#     person_center_x = (head_center_x + shoulder_center_x) / 2
#
#     person_center_y = (
#         float(keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER]['y']) +
#         float(keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER]['y'])
#     ) / 2
#
#     # 初始化偏移
#     dx = 0.0
#
#     # ========= 1️⃣ 头部留白 =========
#     edge_frame = mask_hip_below(frame, keypoints)
#     results = model(cv2.cvtColor(edge_frame, cv2.COLOR_BGR2RGB))
#     head_y = get_highest_point(edge_frame, results)
#     elbow_left_x, elbow_right_x = get_edge_point(frame, results)
#
#     head_height = abs(keypoints[0]['y'] - head_y)  # 头部高度
#     head_margin = head_height * 0.4
#
#     highest = head_y - head_margin
#
#     # ========= 2️⃣ 膝盖 / 脚踝 =========
#     knee_left_y = keypoints[25]['y']
#     knee_right_y = keypoints[26]['y']
#     # ankle_left_y = keypoints[27]['y']
#     # ankle_right_y = keypoints[28]['y']
#     ankle_y = (keypoints[mp_pose.PoseLandmark.LEFT_ANKLE]['y'] + keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE]['y']) / 2
#     hip_y = (keypoints[mp_pose.PoseLandmark.LEFT_HIP]['y'] + keypoints[mp_pose.PoseLandmark.RIGHT_HIP]['y']) / 2
#
#
#     lowest = 1
#     if 1 > knee_left_y > 0.95 or 1 > knee_right_y > 0.95:
#         # 膝盖被裁 → 人整体上移
#         lowest = (ankle_y + hip_y) / 2
#         # dy -= 0.06
#     elif 1.02 > ankle_y > 0.95:
#         # 脚踝被裁 → 人整体下移
#         # dy += 0.06
#         lowest = ankle_y + 0.1
#     # ========= 3️⃣ 胳膊左右裁切 =========
#     elbow_left_x, elbow_right_x = get_edge_point(frame, results)
#
#     if elbow_left_x < 0.02 and elbow_right_x > 0.98:
#         pass  # 这是缩放问题，不动中心
#     elif elbow_left_x < 0.02:
#         dx += 0.05
#     elif elbow_right_x > 0.98:
#         dx -= 0.05
#
#     # ========= 4️⃣ 人物居中 =========
#     if abs(person_center_x - 0.5) > 0.1:
#         dx += (0.5 - person_center_x) * 0.5
#
#
#     # highest = min(highest,lowest - (elbow_right_x - elbow_left_x + 0.05) * height / width)
#     # ========= 合成新中心 =========
#     new_center_x = np.clip(person_center_x + dx, 0.1, 0.9)
#     # print(highest, lowest)
#     new_center_y = (highest + lowest)/2
#
#     person_center_x *= width
#     person_center_y *= height
#     new_center_x *= width
#     new_center_y *= height
#
#
#     scale = choose_scale(lowest - highest)
#     bbox_h = height / scale
#     bbox_w = width / scale
#
#     x1 = new_center_x - bbox_w / 2
#     y1 = new_center_y - bbox_h / 2
#     x2 = new_center_x + bbox_w / 2
#     y2 = new_center_y + bbox_h / 2
#
#     # 裁剪防止越界
#     x1, y1 = max(0, x1), max(0, y1)
#     x2, y2 = min(width, x2), min(height, y2)
#
#     return {
#         "target_center": (new_center_x, new_center_y),
#         "bbox":(x1, y1, x2, y2),
#         "scale": scale,
#     }



def compute_bbox(base_frame, keypoints, model, target_aspect_ratio=None):
    """
    结合 YOLOv5s (边界准确) 和 MediaPipe (姿态准确) 的裁切逻辑。
    优先保证：肢体完整性 (胳膊、膝盖、脚踝不被切)。
    """
    frame = base_frame.copy()
    h_img, w_img = frame.shape[:2]

    # 默认保持原图比例，或者指定如 9/16, 16/9
    if target_aspect_ratio is None:
        target_aspect_ratio = w_img / h_img

    # ================= 1. 获取 MediaPipe 的极限边界 =================
    # 包含了手腕、脚踝、膝盖的所有点
    mp_x1, mp_y1, mp_x2, mp_y2 = _get_mediapipe_bbox(keypoints, w_img, h_img)

    # ================= 2. 获取 YOLOv5 的检测边界 =================
    # YOLO 看到的通常比 MediaPipe 更“胖”一些（包含衣服）
    yolo_x1, yolo_y1, yolo_x2, yolo_y2 = _get_yolo_bbox(model, frame)

    # ================= 3. 计算“并集” (Union Box) =================
    # 取两者最宽的范围，确保绝对不切手、不切脚
    # 如果 YOLO 没检测到人，就完全信赖 MediaPipe
    if yolo_x1 is None:
        final_x1, final_y1, final_x2, final_y2 = mp_x1, mp_y1, mp_x2, mp_y2
    else:
        final_x1 = min(mp_x1, yolo_x1)
        final_y1 = min(mp_y1, yolo_y1)
        final_x2 = max(mp_x2, yolo_x2)
        final_y2 = max(mp_y2, yolo_y2)

    # ================= 4. 智能留白 (Padding) =================
    # 既然目标是“完整展示”，我们需要在极值边界外再加一点 buffer
    box_h = final_y2 - final_y1

    # 顶部留白：防止头顶太贴边 (Headroom)
    pad_top = box_h * 0.15
    # 底部留白：防止脚底太贴边
    pad_bottom = box_h * 0.05
    # 左右留白：防止挥手时手指贴边
    pad_x = (final_x2 - final_x1) * 0.1

    # 应用留白
    crop_x1 = max(0, final_x1 - pad_x)
    crop_x2 = min(w_img, final_x2 + pad_x)
    crop_y1 = max(0, final_y1 - pad_top)
    crop_y2 = min(h_img, final_y2 + pad_bottom)

    # ================= 5. 修正纵横比 (Aspect Ratio Fit) =================
    # 现在的 box 仅仅是包裹住了人，可能比例很奇怪（比如细长条）
    # 我们需要向外扩展背景，直到符合 target_aspect_ratio

    current_w = crop_x2 - crop_x1
    current_h = crop_y2 - crop_y1
    current_ratio = current_w / current_h

    # 目标中心点（以此为基准向外扩）
    # 这里做一个微调：中心点稍微上移一点点，视觉上更稳（胸口位置），而不是几何中心（肚脐）
    cx = (crop_x1 + crop_x2) / 2
    cy = (crop_y1 + crop_y2) / 2

    if current_ratio < target_aspect_ratio:
        # 当前太瘦 -> 增加宽度
        target_w = current_h * target_aspect_ratio
        delta_w = target_w - current_w
        crop_x1 -= delta_w / 2
        crop_x2 += delta_w / 2
    else:
        # 当前太胖 -> 增加高度
        target_h = current_w / target_aspect_ratio
        delta_h = target_h - current_h
        crop_y1 -= delta_h / 2
        crop_y2 += delta_h / 2

    # ================= 6. 最终边界处理 (Shift & Clip) =================
    # 如果向外扩充时超出了图片边界，我们需要平移框，尽量不要缩小框

    # 检查左界
    if crop_x1 < 0:
        crop_x2 += abs(crop_x1)  # 往右推
        crop_x1 = 0
    # 检查右界
    if crop_x2 > w_img:
        crop_x1 -= (crop_x2 - w_img)  # 往左推
        crop_x2 = w_img

    # 检查上界
    if crop_y1 < 0:
        crop_y2 += abs(crop_y1)
        crop_y1 = 0
    # 检查下界
    if crop_y2 > h_img:
        crop_y1 -= (crop_y2 - h_img)
        crop_y2 = h_img

    # 最后的安全截断（防止平移后还不够）
    x1, y1 = max(0, crop_x1), max(0, crop_y1)
    x2, y2 = min(w_img, crop_x2), min(h_img, crop_y2)

    # 计算最终中心和 Scale
    final_cx = (x1 + x2) / 2
    final_cy = (y1 + y2) / 2

    # Scale 定义：原图高度 / 裁切框高度
    # 意味着如果只截取了一半画面，画面就放大了2倍
    # scale = h_img / (y2 - y1) if (y2 - y1) > 0 else 1.0
    scale = round(h_img / (y2 - y1) if (y2 - y1) > 0 else 1.0, 1)

    return {
        "target_center": (final_cx, final_cy),
        "bbox": (int(x1), int(y1), int(x2), int(y2)),
        "scale": scale,
    }


def compute_bbox_standard(base_frame, keypoints, model, target_aspect_ratio=None):
    """
    结合 YOLOv5s 和 MediaPipe 的裁切逻辑。
    输出：归一化后的坐标 (0.0 - 1.0)
    """
    frame = base_frame.copy()
    h_img, w_img = frame.shape[:2]

    if target_aspect_ratio is None:
        target_aspect_ratio = w_img / h_img

    # 1. 获取边界 (原始像素坐标)
    mp_x1, mp_y1, mp_x2, mp_y2 = _get_mediapipe_bbox(keypoints, w_img, h_img)
    yolo_x1, yolo_y1, yolo_x2, yolo_y2 = _get_yolo_bbox_by_results(model)

    # 2. 计算并集
    if yolo_x1 is None:
        final_x1, final_y1, final_x2, final_y2 = mp_x1, mp_y1, mp_x2, mp_y2
    else:
        final_x1, final_y1 = min(mp_x1, yolo_x1), min(mp_y1, yolo_y1)
        final_x2, final_y2 = max(mp_x2, yolo_x2), max(mp_y2, yolo_y2)

    # 3. 智能留白
    box_h = final_y2 - final_y1
    pad_top, pad_bottom = box_h * 0.15, box_h * 0.05
    pad_x = (final_x2 - final_x1) * 0.1

    crop_x1, crop_x2 = max(0, final_x1 - pad_x), min(w_img, final_x2 + pad_x)
    crop_y1, crop_y2 = max(0, final_y1 - pad_top), min(h_img, final_y2 + pad_bottom)

    # 4. 修正纵横比
    current_w, current_h = crop_x2 - crop_x1, crop_y2 - crop_y1
    current_ratio = current_w / current_h

    if current_ratio < target_aspect_ratio:
        target_w = current_h * target_aspect_ratio
        delta_w = target_w - current_w
        crop_x1 -= delta_w / 2
        crop_x2 += delta_w / 2
    else:
        target_h = current_w / target_aspect_ratio
        delta_h = target_h - current_h
        crop_y1 -= delta_h / 2
        crop_y2 += delta_h / 2

    # 5. 边界平移与截断 (Shift & Clip)
    if crop_x1 < 0:
        crop_x2 += abs(crop_x1)
        crop_x1 = 0
    if crop_x2 > w_img:
        crop_x1 -= (crop_x2 - w_img);
        crop_x2 = w_img
    if crop_y1 < 0:
        crop_y2 += abs(crop_y1);
        crop_y1 = 0
    if crop_y2 > h_img:
        crop_y1 -= (crop_y2 - h_img);
        crop_y2 = h_img

    x1, y1 = max(0, crop_x1), max(0, crop_y1)
    x2, y2 = min(w_img, crop_x2), min(h_img, crop_y2)

    # 6. 计算最终中心 (像素)
    final_cx = (x1 + x2) / 2
    final_cy = (y1 + y2) / 2

    # 7. 计算缩放倍数
    scale = round(h_img / (y2 - y1) if (y2 - y1) > 0 else 1.0, 1)

    # ================= 归一化处理 =================
    return {
        # 中心点坐标 (x/w, y/h)
        "target_center": (round(final_cx / w_img, 4), round(final_cy / h_img, 4)),

        # 边界框 [x1/w, y1/h, x2/w, y2/h]
        "bbox": [
            round(x1 / w_img, 4),
            round(y1 / h_img, 4),
            round(x2 / w_img, 4),
            round(y2 / h_img, 4)
        ],

        # Scale 是比例值，本身就是归一化的，无需除以宽高
        "scale": scale,
    }


def _get_mediapipe_bbox(keypoints, w, h):
    """从关键点获取绝对坐标的 bbox"""
    # 筛选全身关键点 (不仅是头肩，还有四肢)
    # MediaPipe Pose landmarks:
    # 11-12: Shoulders, 13-14: Elbows, 15-16: Wrists
    # 23-24: Hips, 25-26: Knees, 27-28: Ankles, 29-30: Heels, 31-32: Foot index
    indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    xs = []
    ys = []
    for idx in indices:
        kp = keypoints[idx]
        # 只要可见性大于 0.5 或者 x,y 不为 0
        if kp.get('visibility', 1.0) > 0.3:
            xs.append(kp['x'])
            ys.append(kp['y'])

    if not xs:
        return 0, 0, w, h  # Fallback

    min_x, max_x = min(xs) * w, max(xs) * w
    min_y, max_y = min(ys) * h, max(ys) * h
    return min_x, max_y, max_x, max_y  # 注意这里有点笔误，修正如下:
    return min_x, min_y, max_x, max_y


def _get_yolo_bbox(model, frame):
    """运行 YOLOv5s 获取最大的人体 Box"""
    # 转 RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 推理
    results = model(img_rgb)

    # 解析结果：pandas format 比较好处理
    df = results.pandas().xyxy[0]

    # 筛选类别 (class 0 通常是 person, 具体看你的模型配置)
    people = df[df['class'] == 0]

    if people.empty:
        return None, None, None, None

    # 找到置信度最高，或者面积最大的人
    # 这里假设画面主体是面积最大的人
    people['area'] = (people['xmax'] - people['xmin']) * (people['ymax'] - people['ymin'])
    target = people.loc[people['area'].idxmax()]

    return target['xmin'], target['ymin'], target['xmax'], target['ymax']


def _get_yolo_bbox_by_results(results):
    """运行 YOLOv5s 获取最大的人体 Box"""


    # 解析结果：pandas format 比较好处理
    df = results.pandas().xyxy[0]

    # 筛选类别 (class 0 通常是 person, 具体看你的模型配置)
    people = df[df['class'] == 0]

    if people.empty:
        return None, None, None, None

    # 找到置信度最高，或者面积最大的人
    # 这里假设画面主体是面积最大的人
    people['area'] = (people['xmax'] - people['xmin']) * (people['ymax'] - people['ymin'])
    target = people.loc[people['area'].idxmax()]

    return target['xmin'], target['ymin'], target['xmax'], target['ymax']
def get_keypoints_bbox(keypoints, ids):
    xs, ys = [], []

    for i in ids:
        kp = keypoints.get(i)
        if kp and kp["visibility"] > 0.5:
            xs.append(kp["x"])
            ys.append(kp["y"])

    if not xs:
        return None

    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys)
    }

def apply_head_margin(bbox, keypoints):
    head_y = keypoints[0]["y"]
    shoulder_y = (keypoints[11]["y"] + keypoints[12]["y"]) / 2
    head_height = abs(shoulder_y - head_y)

    head_margin = head_height * 0.4

    bbox["y_min"] = min(bbox["y_min"], head_y - head_margin)
    return bbox

def expand_bbox_to_target_ratio(bbox, target_body_ratio=0.9):
    """
    target_body_ratio: 人体高度 / 画面高度
    """
    body_height = bbox["y_max"] - bbox["y_min"]

    target_frame_height = body_height / target_body_ratio
    extra = target_frame_height - body_height

    bbox["y_min"] -= extra * 0.5
    bbox["y_max"] += extra * 0.5

    return bbox

def compute_zoom_from_bbox(bbox):
    """
    返回 zoom 值（>1 表示放大）
    """
    bbox_height = bbox["y_max"] - bbox["y_min"]
    zoom = 1.0 / bbox_height
    return np.clip(zoom, 1.0, 2.5)


def compute_target_zoom(frame, keypoints, model):
    edge_frame = mask_hip_below(frame, keypoints)
    # BGR 转 RGB
    img_rgb = cv2.cvtColor(edge_frame, cv2.COLOR_BGR2RGB)

    # 进行推理
    results = model(img_rgb)

    bbox = get_box(edge_frame, results)

    if bbox is None:
        return 1.0

    bbox = apply_head_margin(bbox, keypoints)
    bbox = expand_bbox_to_target_ratio(bbox, target_body_ratio=0.9)

    zoom = compute_zoom_from_bbox(bbox)
    return zoom

def compute_bbox_from_center_and_scale(
    image_shape,
    center_x,
    center_y,
    scale,
    aspect_ratio=0.5,
    normalized=True
):
    """
    image_shape: (H, W, C)
    center_x, center_y: 中心点（归一化或像素）
    scale: 相对于图像高度的比例
    aspect_ratio: w / h
    normalized: 是否是归一化坐标
    """

    height, width = image_shape[:2]

    # 中心点 → 像素
    if normalized:
        cx = center_x * width
        cy = center_y * height
    else:
        cx, cy = center_x, center_y

    box_h = height * scale
    box_w = box_h * aspect_ratio

    x1 = int(max(0, cx - box_w / 2))
    y1 = int(max(0, cy - box_h / 2))
    x2 = int(min(width,  cx + box_w / 2))
    y2 = int(min(height, cy + box_h / 2))

    return x1, y1, x2, y2, int(cx), int(cy)


def get_center_point(keypoints):
    """
    根据关键点计算人物的中心点。这里取的是头部、肩膀和臀部的中间点
    """
    head_center_x = (keypoints[0]['x'] + keypoints[1]['x'] + keypoints[2]['x']) / 3
    shoulder_center_x = (keypoints[1]['x'] + keypoints[2]['x']) / 2
    # 计算人物的水平中心点
    person_center_x = (head_center_x + shoulder_center_x) / 2
    # 计算人物的垂直中心点（这里可以根据头部、肩膀和髋部的y坐标平均值来确定）
    person_center_y = (keypoints[0]['y'] + keypoints[1]['y'] + keypoints[2]['y'] + keypoints[25]['y'] + keypoints[26]['y']) / 5
    return person_center_x, person_center_y



def get_result(frame, keypoints):
    """
    根据关键点计算目标矩形框的位置和大小，并返回调整后的矩形框
    """
    height, width = frame.shape[:2]

    # 获取当前的中心点
    person_center_x, person_center_y = get_center_point(keypoints)

    # 图像中心点
    center_x = width // 2
    center_y = height // 2

    # 计算偏移量
    offset_x = center_x - person_center_x * width
    offset_y = center_y - person_center_y * height

    # 计算理想的矩形框大小（例如，基于膝盖和肩膀的距离来估算）
    shoulder_width = abs(keypoints[1]['x'] - keypoints[2]['x']) * width
    height_margin = abs(keypoints[0]['y'] - keypoints[25]['y']) * height  # 身体的垂直高度（从头到膝盖）

    # 计算矩形框的左上角和右下角
    left = int(max(0, offset_x))
    top = int(max(0, offset_y))
    right = int(min(width, left + shoulder_width))
    bottom = int(min(height, top + height_margin))

    return left, top, right, bottom


def suggest_orientation_multi(yolo_results, target_aspect_ratio=None):
    """
    根据 YOLO 检测到的所有人，判断整体适合横屏还是竖屏。

    Args:
        yolo_results: YOLOv5 的 pandas 结果 (df = results.pandas().xyxy[0])
    """
    # 筛选所有人
    people = yolo_results[0]

    if len(people) == 0:
        return "Portrait", "未检测到人物"

    # 1. 计算所有人构成的“大包围盒”
    all_x1 = people['xmin'].min()
    all_y1 = people['ymin'].min()
    all_x2 = people['xmax'].max()
    all_y2 = people['ymax'].max()

    group_w = all_x2 - all_x1
    group_h = all_y2 - all_y1
    group_ratio = group_w / group_h  # 注意这里用 W/H，大于 1 表示宽

    # 2. 决策逻辑
    if len(people) == 1:
        # 单人情况：回归姿态逻辑（此处简化为比例判断）
        return "Portrait" if group_ratio < 0.8 else "Landscape", "单人姿态适配"

    # 3. 多人核心逻辑
    if len(people) >= 2:
        # 如果人群宽度明显大于高度 (例如 2 人并排，比例通常会超过 1.2)
        if group_ratio > 1.2:
            return "Landscape", f"检测到 {len(people)} 人横向排布，建议横屏捕捉全员"

        # 如果人群比较“瘦长”（例如前后站位或拥抱）
        elif group_ratio < 0.85:
            return "Portrait", "人群构图紧凑，竖屏更具视觉重心"

        # 处于中间地带 (0.85 ~ 1.2)
        else:
            return "Landscape", "多人组合构图，建议使用横屏预留环境空间"