import cv2
import mediapipe as mp

import cv2
import numpy as np





# 初始化 Mediapipe Pose 模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化 OpenCV 视频捕捉
cap = cv2.VideoCapture(1)  # 打开摄像头

# 设置摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 360)  # 设置宽度为1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)  # 设置高度为720


# 捕获一帧图像
ret, frame = cap.read()

# 检查图像是否正确捕获
if not ret:
    print("无法访问摄像头")
    cap.release()
    exit()

# 获取图像的宽高
height, width = frame.shape[:2]
print(frame.shape)

# 保持宽高比，设定新的宽度
new_width = 200  # 设定一个新的宽度
aspect_ratio = width / height  # 计算宽高比
new_height = int(new_width / aspect_ratio)  # 计算新的高度

# 缩放图像，保持比例
resized_frame = cv2.resize(frame, (new_width, new_height))

# 打印图像的宽高（用于调试）
print(f"原始图像尺寸: {frame.shape[:2]}, 缩放后的图像尺寸: {resized_frame.shape[:2]}")

# 将图像从 BGR 转为 RGB
rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

# 处理图像，获取姿态关键点
results = pose.process(rgb_frame)

# 设置一个可视性阈值
visibility_threshold = 0.5  # 可调整的阈值，0.5表示可视的关键点

# 检查是否检测到关键点
if results.pose_landmarks:
    print("可视的关键点：")
    # 遍历每个关键点并筛选出可见的关键点
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        if landmark.visibility > visibility_threshold:
            # 获取每个关键点的位置
            x = int(landmark.x * resized_frame.shape[1])
            y = int(landmark.y * resized_frame.shape[0])

            # 打印可视关键点的信息
            print(f"ID: {idx}, 位置: ({x}, {y}), 可见性: {landmark.visibility}")

            # 在关键点位置绘制一个小圆点并显示编号
            cv2.circle(resized_frame, (x, y), 5, (0, 255, 0), -1)  # 绿色圆点
            cv2.putText(resized_frame, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

else:
    print("未能检测到姿势关键点")

# 显示结果图像
cv2.imshow("Pose Detection", resized_frame)

# 等待按键，按任意键退出
cv2.waitKey(0)
cv2.destroyAllWindows()

# 释放摄像头
cap.release()


#
# def get_highest_point(frame, model):
#     """
#     计算人物的最高点（边界框的顶部中心点）
#     """
#     # BGR 转 RGB
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # 进行推理
#     results = model(img_rgb)
#
#     # 获取检测结果（boxes, labels, scores）
#     boxes = results.xywh[0][:, :-2]  # 获取所有的边界框
#     scores = results.xywh[0][:, -2]  # 得到置信度
#     labels = results.xywh[0][:, -1]  # 类别名称
#
#     # 提取每个边界框的上下左右坐标
#     highest_point = None
#     for box, score, label in zip(boxes, scores, labels):
#         print(box, score, label)
#         if label != 0 or score < 0.5:  # 如果置信度低于0.5，忽略
#             continue
#
#         # 获取边界框的坐标 (x_center, y_center, width, height)
#         x_center, y_center, w, h = box
#
#         # 计算边界框的顶部中心点
#         top_y = y_center - h / 2  # 顶部的y坐标
#         top_x = x_center  # 顶部的x坐标与中心相同
#
#         # 记录最高点的位置
#         if highest_point is None or top_y < highest_point[1]:
#             highest_point = (top_x, top_y)
#
#     return highest_point
#
#
# import torch
# import cv2
# import numpy as np
#
# # 加载 YOLOv5 模型（使用预训练权重）
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#
# # 打开摄像头
# cap = cv2.VideoCapture(1)
#
# while True:
#     # 读取摄像头每一帧
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # BGR 转 RGB
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # 进行推理
#     results = model(img_rgb)
#
#     # 获取检测结果（boxes, labels, scores）
#     results.render()  # 绘制边界框到图像上
#     output_frame = results.ims[0]  # 获取带有边界框的结果
#
#     print(get_highest_point(frame, model))
#
#     # 转换回 BGR 以便于 OpenCV 显示
#     output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
#
#     # 显示实时视频流
#     cv2.imshow("YOLOv5 Real-time Camera Segmentation", output_frame)
#
#     # 按 'q' 键退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 释放资源
# cap.release()
# cv2.destroyAllWindows()
