import torch
from fastapi import FastAPI, File, UploadFile, Form
import cv2
import numpy as np
import mediapipe as mp
import time
import os

from realtime_extractor import PoseExtractor
from CompositionAnalyzer import compute_bbox, analyze_crop_and_zoom, compute_bbox_standard, suggest_orientation_multi, \
    compute_bbox_by_mode

# 确保有个文件夹存这些“失败”的图片
if not os.path.exists("debug_frames"):
    os.makedirs("debug_frames")

app = FastAPI()

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
from ultralytics import YOLO

# 加载模型 (yolov8n 是最轻量的)
yolo_model = YOLO('yolov8n.pt')

# yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.classes = [0]
yolo_model.conf = 0.4
yolo_model.iou = 0.5
yolo_model.max_det = 1

# @app.post("/analyze")
# async def analyze_pose(
#         file: UploadFile = File(...),
#         selected_pose: str = Form(...)
# ):
#     start_time = time.time()
#
#     # 1. 读取并解码图片
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
#     if image is None:
#         print("❌ [错误] 无法解码图片")
#         return {"success": False, "msg": "图像解码失败"}
#
#     h, w, _ = image.shape
#     # print(f"\n🔔 [新请求] 姿态类型: {selected_pose} | 分辨率: {w}x{h}")
#
#     # 2. 姿态检测
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # cv2.imshow('Check image', image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     results = pose.process(image_rgb)
#
#     response_data = {
#         "success": True,
#         "scale": 1.0,
#         "bbox": [0, 0, 0, 0],
#         "center": [0.5, 0.5],
#         "suggestions": ["未检测到人物"],
#         "user_points": [],
#     }
#
#     # 3. 如果检测到骨架
#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark
#
#         bbox_result = compute_bbox_standard(image, PoseExtractor().extract_from_frame(image), yolo_model)
#         camera_suggestion = analyze_crop_and_zoom(image, PoseExtractor().extract_from_frame(image),
#                                                   yolo_model)
#
#         # cached_bbox = {
#         #     "center": bbox_result["target_center"],
#         #     "bbox": bbox_result["bbox"],
#         #     "scale": bbox_result["scale"],
#         # }
#
#         # 整理返回内容
#         response_data["user_points"] = [[lm.x, lm.y] for lm in landmarks]
#         response_data["bbox"] = bbox_result["bbox"]
#         response_data["center"] = bbox_result["target_center"]
#         response_data["scale"] = bbox_result["scale"]
#         response_data["suggestions"] = camera_suggestion
#
#         # print(response_data)
#
#     else:
#         print("⚠️ [检测失败] 画面中没有发现人")
#
#     end_time = time.time()
#     # print(f"⏱️ [耗时] {(end_time - start_time) * 1000:.2f} ms")
#
#     return response_data
#
# @app.post("/analyze")
# async def get_photography_advice(
#         file: UploadFile = File(...)
# ):
#     # 1. 读取并解码图片
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
#     if frame is None:
#         print("❌ [错误] 无法解码图片")
#         return {"success": False, "msg": "图像解码失败"}
#
#     h_img, w_img = frame.shape[:2]
#
#     # --- 步骤 1: AI 运行 ---
#     # YOLO 检测 (多人/物体)
#     results = yolo_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     people_df = results.pandas().xyxy[0]
#     people_df = people_df[people_df['class'] == 0]  # 只看人
#
#     # MediaPipe 检测 (单人骨骼)
#     # 注意：如果是多人，这里通常取置信度最高的或者面积最大的
#     mp_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#     # --- 步骤 2: 逻辑计算 ---
#     advice_list = []
#
#     if not people_df.empty:
#         # A. 计算横竖屏建议
#         orientation, o_reason = suggest_orientation_multi(people_df)
#         advice_list.append(f"建议: {orientation} ({o_reason})")
#         return  {
#             "advice": advice_list,
#         }
#
#     return None


@app.post("/analyze")
async def analyze_pose(
        file: UploadFile = File(...),
        pose_type: str = Form("站立"),
        view_mode: str = Form("全身像")
):
    start_time = time.time()

    # 1. 读取并解码图片
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"success": False, "msg": "图像解码失败"}

    h_img, w_img, _ = image.shape

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- 步骤 1: 运行 AI 模型 ---
    # MediaPipe 姿态检测
    mp_results = pose.process(image_rgb)
    # YOLO 检测 (用于多人及横竖屏判定)
    # yolo_results = yolo_model(image_rgb)
    # 强制指定设备并显式调用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_model.to(device)
    yolo_results = yolo_model(image_rgb)
    result = yolo_results[0]

    yolo_box = None
    if len(result.boxes) > 0:
        # 1. 提取第一个检测到的物体坐标 [x1, y1, x2, y2]
        # .cpu().numpy() 将数据从 GPU 转到 CPU 数组
        box_data = result.boxes.xyxy[0].cpu().numpy()

        # 2. 赋值
        yolo_x1, yolo_y1, yolo_x2, yolo_y2 = box_data
        yolo_box = (yolo_x1/w_img, yolo_y1/h_img, yolo_x2/w_img, yolo_y2/h_img)

        print(f"检测到人: {yolo_box}")
    else:
        timestamp = int(time.time())
        filename = f"debug_frames/fail_{timestamp}.jpg"
        cv2.imwrite(filename, image)
        print(f"⚠️ 检测失败，图片已保存至: {filename}")
        # print("未检测到人")

    # --- 步骤 2: 初始化返回结构 ---
    response_data = {
        "success": True,
        "scale": 1.0,
        "bbox": [0.0, 0.0, 0.0, 0.0],
        "center": [0.5, 0.5],
        "suggestions": [],
        "user_points": [],
        "suggested_orientation": "portrait"  # 默认为竖屏，不触发动画
    }

    # --- 步骤 3: 核心逻辑计算 ---

    # A. 姿态点与构图计算 (如果有 MediaPipe 结果)
    if mp_results.pose_landmarks:
        landmarks = mp_results.pose_landmarks.landmark
        response_data["user_points"] = [[lm.x, lm.y] for lm in landmarks]

        # 调用你现有的工具函数计算 BBox 和 缩放建议
        # 注意：这里假设 PoseExtractor().extract_from_frame 返回符合要求的格式
        pose_data = PoseExtractor().extract_from_frame(image)
        # bbox_result = compute_bbox_standard(image, pose_data, yolo_model)
        bbox_result = compute_bbox_by_mode(image, pose_data, yolo_box, mode=view_mode)
        camera_advice = analyze_crop_and_zoom(image, pose_data, yolo_box)

        response_data["bbox"] = bbox_result["bbox"]  # [xmin, ymin, xmax, ymax] 归一化
        response_data["center"] = bbox_result["target_center"]
        response_data["scale"] = bbox_result["scale"]

        # 将原有建议转化为 SuggestionDetail 对象列表格式（匹配 Compose 端）
        response_data["suggestions"] = camera_advice

    else:
        response_data["suggestions"] = [{"id": "0", "text": "未检测到人物", "needModify": True}]

    # B. 横竖屏判定逻辑 (基于 YOLO 多人结果)
    # 调用你定义的建议函数
    # orientation, o_reason = suggest_orientation_multi(yolo_results)

    # 如果建议是横屏，则更新字段，触发前端“震动”和“箭头”
    # if orientation == "横屏":
    #     response_data["suggested_orientation"] = "landscape"
    #     response_data["suggestions"].append(
    #         {
    #         "id": "orient_01",
    #         "text": f"建议切换横屏: {o_reason}",
    #         "needModify": True
    #         }
    #     )
    # else:
    #     response_data["suggested_orientation"] = "portrait"

    response_data["suggested_orientation"] = "landscape"
    response_data["suggestions"].append(
        {
            "id": "orient_01",
            "text": f"建议切换横屏。",
            "needModify": True
        }
    )

    # --- 步骤 4: 性能统计与返回 ---
    process_time = (time.time() - start_time) * 1000
    # print(f"⏱️ 姿态分析耗时: {process_time:.2f} ms")

    print(response_data)

    return response_data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)