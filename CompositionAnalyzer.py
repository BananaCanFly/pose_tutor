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
    # shoulder_left_y = keypoints[11]['y']  # 左肩
    # shoulder_right_y = keypoints[12]['y']  # 右肩
    knee_left_y = keypoints[25]['y']  # 左膝
    knee_right_y = keypoints[26]['y']  # 右膝
    ankle_left_y = keypoints[27]['y']  # 左脚踝
    ankle_right_y = keypoints[28]['y']  # 右脚踝
    # elbow_left_x = keypoints[13]['x']  # 左肘
    # elbow_right_x = keypoints[14]['x']  # 右肘
    # wrist_left_y = keypoints[15]['y']  # 左手腕
    # wrist_right_y = keypoints[16]['y']  # 右手腕

    # 1. 先降分辨率（非常关键）

    frame = cv2.resize(frame, (320, 320))
    edge_frame = mask_hip_below(frame, keypoints)

    # 2. YOLO 可以直接吃 BGR（不需要手动转 RGB）
    # 3. 关闭梯度 + 指定类别 + 指定设备
    with torch.no_grad():
        results = model(edge_frame)

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
            {"id": "留白", "text": "⬆ 请向上移动一点（头顶空间不足）", "need_modify": True}
        )
    else:
        suggestions.append(
            {"id": "留白", "text": "✅ 头顶留白足够", "need_modify": False}
        )
    # print(estimate_knee_height(keypoints))
    # print(knee_left_y, knee_right_y, ankle_left_y, ankle_right_y, height)
    # 判断膝盖和脚踝是否被裁剪
    knee_y = estimate_knee_height(keypoints)

    # if knee_y > 0.95:
    if 1 > knee_left_y > 0.95 or 1 > knee_right_y > 0.95:
        suggestions.append(
            {"id": "关节", "text": "⬆ 请向上移动一点（膝盖部分被裁剪）", "need_modify": True}
        )
    elif 1.02 > ankle_left_y > 0.95 or 1.02 > ankle_right_y > 0.95:
        suggestions.append(
            {"id": "关节", "text": "⬇ 请向下移动一点（脚踝部分被裁剪）", "need_modify": True}
        )
    else:
        suggestions.append(
            {"id": "关节", "text": "✅ 关节完整显示", "need_modify": False}
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
            {"id": "胳膊", "text": "⬆/⬇ 缩放画面（两侧胳膊均部分不可见）", "need_modify": True}
        )
    elif elbow_left_x<0.02:
        suggestions.append(
            {"id": "胳膊", "text": "⬅ 请左移一点（左侧胳膊不可见）", "need_modify": True}
        )
    elif elbow_right_x>0.98:
        suggestions.append(
            {"id": "胳膊", "text": "➡ 请右移一点（右侧胳膊不可见）", "need_modify": True}
        )
    else:
        suggestions.append(
            {"id": "胳膊", "text": "✅ 胳膊完整显示", "need_modify": False}
        )

    # 判断人物是否居中
    center_x = width // 2
    head_center_x = (keypoints[0]['x'] + keypoints[1]['x'] + keypoints[2]['x']) / 3
    shoulder_center_x = (keypoints[1]['x'] + keypoints[2]['x']) / 2
    person_center_x = (head_center_x + shoulder_center_x) / 2
    # print(head_center_x, shoulder_center_x, person_center_x, center_x)
    if abs(person_center_x - 0.5) > 0.1:
        if person_center_x < 0.5:
            suggestions.append(
                {"id": "中心", "text": "➡ 请右移一点（人物偏左）", "need_modify": True}
            )
        else:
            suggestions.append(
                {"id": "中心", "text": "⬅ 请左移一点（人物偏右）", "need_modify": True}
            )
    else:
        suggestions.append(
            {"id": "中心", "text": "✅ 人物居中良好", "need_modify": False}
        )

    # # 判断是否需要缩放（通过肩膀宽度来判断）
    # shoulder_width = abs(keypoints[1]['x'] - keypoints[2]['x'])  # 计算肩膀宽度
    # zoom_suggestion = ""
    # if shoulder_width < width * 0.2:
    #     zoom_suggestion = "建议放大，人物显得太小"
    # elif shoulder_width > width * 0.6:
    #     zoom_suggestion = "建议缩小，人物占据空间过大"
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


import numpy as np
import mediapipe as mp
import cv2


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