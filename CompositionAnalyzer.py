# crop_advisor.py
import cv2
import numpy as np

import mediapipe as mp

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

EDGE_MARGIN_RATIO = 0.05   # 5% 画面边缘安全区
MIN_VISIBILITY = 0.3
MAX_ZOOM = 1.8
MIN_ZOOM_TRIGGER = 1.05

# def analyze_crop_and_zoom(frame, keypoints):
#     print(keypoints)
#     """
#     分析是否需要移动、裁剪和缩放建议
#     参数:
#     - frame: 当前帧图像（OpenCV格式）
#     - keypoints: 人物的关键点列表，包含头部、肩膀、肘部等部位的坐标
#
#     返回:
#     - dict: 包含移动、裁剪和缩放建议的信息
#     """
#     height, width = frame.shape[:2]
#
#     if not keypoints:
#         return None
#
#     # 获取头部位置
#     head_x, head_y = keypoints[0]['x'] * width, keypoints[0]['y'] * height  # 头部位置
#     # 获取肩膀位置并计算肩膀宽度
#     shoulder_left_x, shoulder_left_y = keypoints[11]['x'] * width, keypoints[11]['y'] * height
#     shoulder_right_x, shoulder_right_y = keypoints[12]['x'] * width, keypoints[12]['y'] * height
#
#     shoulder_width = np.sqrt((shoulder_right_x - shoulder_left_x) ** 2 + (shoulder_right_y - shoulder_left_y) ** 2)
#
#     # 判断人物是否居中
#     center_x, center_y = width // 2, height // 2
#
#     # ===========================================
#     # 1️⃣ 给出移动建议
#     # ===========================================
#     move_suggestion = ""
#     if head_x < width * 0.3:
#         move_suggestion = "建议向右移动画面，人物偏左。"
#     elif head_x > width * 0.7:
#         move_suggestion = "建议向左移动画面，人物偏右。"
#
#     # ===========================================
#     # 2️⃣ 给出缩放建议
#     # ===========================================
#     zoom_suggestion = ""
#     if shoulder_width < width * 0.3:
#         zoom_suggestion = "建议放大，人物看起来太小。"
#     elif shoulder_width > width * 0.5:
#         zoom_suggestion = "建议缩小，人物占据太大空间。"
#
#     # ===========================================
#     # 3️⃣ 给出移动建议
#     # ===========================================
#     crop_suggestion = ""
#     if head_y < height * 0.3:
#         crop_suggestion = "建议向下移动，人物偏上。"
#     elif head_y > height * 0.7:
#         crop_suggestion = "建议向上移动，人物偏下。"
#
#     # ===========================================
#     # 4️⃣ 画面缩放建议（根据离边缘的距离）
#     # ===========================================
#     safe_margin_x = width * EDGE_MARGIN_RATIO  # 设定安全距离
#     safe_margin_y = height * EDGE_MARGIN_RATIO
#
#     max_required_scale = 1.0
#     risky_parts = set()
#
#     for kp in keypoints:
#         kp_id = kp.get("id")
#         if kp_id not in CRITICAL_JOINT_IDS:
#             continue
#         if kp.get("visibility", 0) < MIN_VISIBILITY:
#             continue
#
#         x, y = kp["x"] * width, kp["y"] * height
#         dist_to_edge = min(x, width - x, y, height - y)
#
#         if dist_to_edge < min(safe_margin_x, safe_margin_y):
#             scale = min(safe_margin_x, safe_margin_y) / max(dist_to_edge, 1.0)
#             max_required_scale = max(max_required_scale, scale)
#             risky_parts.add(CRITICAL_JOINT_IDS[kp_id])
#
#     if max_required_scale < MIN_ZOOM_TRIGGER:
#         return None
#
#     zoom_factor = round(min(max_required_scale * 1.1, MAX_ZOOM), 2)
#
#     # 返回所有建议
#     return {
#         "id": "移动与缩放",
#         "move_suggestion": move_suggestion,
#         "zoom_suggestion": zoom_suggestion,
#         "crop_suggestion": crop_suggestion,
#         "zoom_factor": zoom_factor,
#         "risky_parts": list(risky_parts),
#         "text": (
#             f"画面在 {', '.join(map(str, risky_parts))} 附近发生裁切，"
#             f"容易在关节处形成断裂感。"
#             f"{zoom_suggestion}\n"
#             f"{move_suggestion}\n"
#         )
#     }


# def analyze_crop_and_zoom(frame, keypoints):
#     """
#     分析是否存在不合理裁剪，并给出建议放大倍数
#
#     Args:
#         frame (np.ndarray): BGR 图像
#         keypoints (list[dict]): MediaPipe pose landmarks
#
#     Returns:
#         dict | None: suggestion
#     """
#     # print(keypoints)
#     if frame is None or not keypoints:
#         return None
#
#     h, w = frame.shape[:2]
#     safe_margin_x = w * EDGE_MARGIN_RATIO
#     safe_margin_y = h * EDGE_MARGIN_RATIO
#
#     max_required_scale = 1.0
#     risky_parts = set()
#
#     for kp in keypoints:
#         kp_id = kp.get("id")
#         if kp_id not in CRITICAL_JOINT_IDS:
#             continue
#         if kp.get("visibility", 0) < MIN_VISIBILITY:
#             continue
#
#         x = kp["x"] * w
#         y = kp["y"] * h
#
#         dist_to_edge = min(
#             x, w - x,
#             y, h - y
#         )
#
#         safe_dist = min(safe_margin_x, safe_margin_y)
#
#         if dist_to_edge < safe_dist:
#             scale = safe_dist / max(dist_to_edge, 1.0)
#             max_required_scale = max(max_required_scale, scale)
#             risky_parts.add(CRITICAL_JOINT_IDS[kp_id])
#
#     if max_required_scale < MIN_ZOOM_TRIGGER:
#         return None
#
#     zoom_factor = round(min(max_required_scale * 1.1, MAX_ZOOM), 2)
#
#     return {
#         "id": "画面裁剪",
#         "type": "framing",
#         "priority": 0,
#         "zoom_factor": zoom_factor,
#         "risky_parts": list(risky_parts),
#         "text": (
#             f"画面在 {', '.join(risky_parts)} 附近发生裁切，"
#             f"容易在关节处形成断裂感。"
#             f"建议将画面整体放大约 {zoom_factor} 倍，"
#             f"避免在可弯曲部位裁切，画面会更自然专业。"
#         )
#     }


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

    mp_pose = mp.solutions.pose
    # 提取肩部和髋部关键点
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

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


def analyze_crop_and_zoom(frame, keypoints, model):
    """
    分析拍照建议，包括头部留白、膝盖脚踝裁剪、胳膊显示、人物居中等
    参数:
    - frame: 当前帧图像（OpenCV格式）
    - keypoints: 人物的关键点列表，包含头部、肩膀、肘部、膝盖、脚踝等部位的坐标

    返回:
    - dict: 包含裁剪和缩放建议的信息
    """

    # print(keypoints)
    suggestions = []
    height, width = frame.shape[:2]

    # 获取关键点
    # head_y = keypoints[0]['y']  # 头部位置
    shoulder_left_y = keypoints[11]['y']  # 左肩
    shoulder_right_y = keypoints[12]['y']  # 右肩
    knee_left_y = keypoints[25]['y']  # 左膝
    knee_right_y = keypoints[26]['y']  # 右膝
    ankle_left_y = keypoints[27]['y']  # 左脚踝
    ankle_right_y = keypoints[28]['y']  # 右脚踝
    elbow_left_x = keypoints[13]['x']  # 左肘
    elbow_right_x = keypoints[14]['x']  # 右肘
    wrist_left_y = keypoints[15]['y']  # 左手腕
    wrist_right_y = keypoints[16]['y']  # 右手腕

    edge_frame = mask_hip_below(frame, keypoints)
    # BGR 转 RGB
    img_rgb = cv2.cvtColor(edge_frame, cv2.COLOR_BGR2RGB)

    # 进行推理
    results = model(img_rgb)

    head_y = get_highest_point(edge_frame, results)
    elbow_left_x, elbow_right_x = get_edge_point(frame, results)

    # 计算头部上方的留白（理想高度为头部的20%-30%）
    # print("头顶高度:", get_highest_point(edge_frame, results))
    head_height = abs(keypoints[0]['y'] - head_y)  # 头部高度
    head_margin = head_height * 0.4  # 留白高度（头部高度的20%）

    # print(head_height, head_margin)
    # 判断头部是否靠近画面顶部
    if head_y < head_margin:
        suggestions.append(
            {"id": "留白", "text": "建议向上移动，头部与顶部的留白不够", "need_modify": True}
        )
    else:
        suggestions.append(
            {"id": "留白", "text": "无需移动，留白合理", "need_modify": False}
        )
    # print(estimate_knee_height(keypoints))
    # print(knee_left_y, knee_right_y, ankle_left_y, ankle_right_y, height)
    # 判断膝盖和脚踝是否被裁剪
    knee_y = estimate_knee_height(keypoints)

    # if knee_y > 0.95:
    if 1 > knee_left_y > 0.95 or 1 > knee_right_y > 0.95:
        suggestions.append(
            {"id": "关节", "text": "建议向上移动，膝盖部分被裁剪", "need_modify": True}
        )
    elif 1.02 > ankle_left_y > 0.95 or 1.02 > ankle_right_y > 0.95:
        suggestions.append(
            {"id": "关节", "text": "建议向下移动，脚踝部分被裁剪", "need_modify": True}
        )
    else:
        suggestions.append(
            {"id": "关节", "text": "没有关节被裁剪，无需移动", "need_modify": False}
        )

    # 判断胳膊是否完全可见
    # if 0.02<elbow_left_x<0.98 and 0.02<elbow_right_x<0.98:
    #     suggestions.append(
    #         {"id": "胳膊", "text": "胳膊已完整露出，无需调整", "need_modify": False}
    #     )
    # else:
    #     suggestions.append(
    #         {"id": "胳膊", "text": "建议调整，胳膊部分不可见，可能需要缩放或调整角度", "need_modify": True}
    #     )
    if elbow_left_x<0.02 and elbow_right_x>0.98:
        suggestions.append(
            {"id": "胳膊", "text": "两侧胳膊均部分不可见，建议缩放", "need_modify": True}
        )
    elif elbow_left_x<0.02:
        suggestions.append(
            {"id": "胳膊", "text": "左侧胳膊部分不可见，建议左移", "need_modify": True}
        )
    elif elbow_right_x>0.98:
        suggestions.append(
            {"id": "胳膊", "text": "右侧胳膊部分不可见，建议右移", "need_modify": True}
        )
    else:
        suggestions.append(
            {"id": "胳膊", "text": "胳膊已完整露出，无需调整", "need_modify": False}
        )

    # 判断人物是否居中
    center_x = width // 2
    head_center_x = (keypoints[0]['x'] + keypoints[1]['x'] + keypoints[2]['x']) / 3
    shoulder_center_x = (keypoints[1]['x'] + keypoints[2]['x']) / 2
    person_center_x = (head_center_x + shoulder_center_x) / 2
    # print(head_center_x, shoulder_center_x, person_center_x, center_x)
    if abs(person_center_x - 0.5) > 0.1:
        suggestions.append(
            {"id": "中心", "text": "建议调整，人物偏离中心", "need_modify": True}
        )
    else:
        suggestions.append(
            {"id": "中心", "text": "人物居中良好", "need_modify": False}
        )

    # # 判断是否需要缩放（通过肩膀宽度来判断）
    # shoulder_width = abs(keypoints[1]['x'] - keypoints[2]['x'])  # 计算肩膀宽度
    # zoom_suggestion = ""
    # if shoulder_width < width * 0.2:
    #     zoom_suggestion = "建议放大，人物显得太小"
    # elif shoulder_width > width * 0.6:
    #     zoom_suggestion = "建议缩小，人物占据空间过大"

    return suggestions

