# pose_analyzer.py
import numpy as np
import json
from pathlib import Path


class PoseAnalyzer:
    def __init__(self):
        """初始化姿势分析器"""
        self.standard_poses = {}
        self.load_standard_poses()

        # 定义身体部位分组
        self.body_parts = {
            "face": list(range(0, 11)),  # 0-10: 面部
            "shoulders": [11, 12],  # 肩膀
            "elbows": [13, 14],  # 手肘
            "wrists": [15, 16],  # 手腕
            "hands": list(range(17, 23)),  # 17-22: 手部细节
            "hips": [23, 24],  # 髋部
            "knees": [25, 26],  # 膝盖
            "ankles": [27, 28],  # 脚踝
            "feet": list(range(29, 33))  # 29-32: 脚部细节
        }

    def load_standard_poses(self):
        """加载所有标准姿势"""
        poses_folder = Path("standard_poses")

        if not poses_folder.exists():
            print("⚠️ 标准姿势文件夹不存在")
            print("请先运行: python pose_extractor.py")
            return

        # 加载所有JSON文件
        json_files = list(poses_folder.glob("*.json"))

        if len(json_files) == 0:
            print("⚠️ 没有找到姿势数据文件")
            print("请先运行: python pose_extractor.py")
            return

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    pose_data = json.load(f)

                # 提取姿势名称
                pose_name = json_file.stem
                self.standard_poses[pose_name] = {
                    "name": pose_name,
                    "keypoints": pose_data
                }
                print(f"✅ 加载姿势: {pose_name}")

            except Exception as e:
                print(f"❌ 加载姿势文件 {json_file} 时出错: {e}")

        print(f"\n📚 已加载 {len(self.standard_poses)} 个标准姿势")

    def compare_poses(self, user_keypoints, std_pose_name=None):
        """比较用户姿势和标准姿势"""
        if not self.standard_poses:
            return {"error": "没有标准姿势可供比较"}

        # 如果没有指定标准姿势，自动找到最相似的
        if std_pose_name is None:
            std_pose_name = self.find_most_similar(user_keypoints)
            # print(f"🤖 自动匹配到姿势: {std_pose_name}")

        if std_pose_name not in self.standard_poses:
            return {"error": f"标准姿势 '{std_pose_name}' 不存在"}

        # 获取标准姿势
        std_pose = self.standard_poses[std_pose_name]["keypoints"]

        # 计算差异
        differences = self.calculate_differences(user_keypoints, std_pose)

        # 计算分数
        score = self.calculate_score(differences)

        # 生成建议
        suggestions = self.generate_suggestions(differences)

        return {
            "standard_pose": std_pose_name,
            "score": score,
            "differences": differences,
            "suggestions": suggestions,
            "is_good": score >= 70,
            "detailed_analysis": self.get_detailed_analysis(differences)
        }

    def find_most_similar(self, user_keypoints):
        """找到最相似的标准姿势"""
        if not self.standard_poses:
            return None

        best_match = None
        best_similarity = 0

        # 将用户关键点转换为numpy数组
        user_array = self.keypoints_to_array(user_keypoints)

        for pose_name, pose_data in self.standard_poses.items():
            std_array = self.keypoints_to_array(pose_data["keypoints"])

            # 确保数组长度一致
            min_len = min(len(user_array), len(std_array))
            if min_len == 0:
                continue

            # 计算相似度（使用前min_len个点）
            similarity = self.calculate_similarity(
                user_array[:min_len],
                std_array[:min_len]
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pose_name

        return best_match

    def keypoints_to_array(self, keypoints):
        """将关键点列表转换为numpy数组"""
        points = []
        for kp in keypoints:
            # 只使用x, y坐标
            points.append([kp['x'], kp['y']])
        return np.array(points)

    def calculate_similarity(self, pose1, pose2):
        """计算两个姿势的相似度"""
        if len(pose1) != len(pose2) or len(pose1) == 0:
            return 0

        # 计算加权欧氏距离
        distances = np.linalg.norm(pose1 - pose2, axis=1)

        # 对不同部位给予不同权重
        weights = np.ones(len(pose1))

        # 重要部位权重更高
        important_indices = [11, 12, 13, 14, 15, 16, 23, 24]  # 肩膀、手肘、手腕、髋部
        for idx in important_indices:
            if idx < len(weights):
                weights[idx] = 2.0

        weighted_distances = distances * weights
        avg_distance = np.mean(weighted_distances)

        # 距离越小，相似度越高
        similarity = 1.0 / (1.0 + avg_distance * 10)
        return similarity

    def calculate_differences(self, user_kps, std_kps):
        """计算关键点差异"""
        differences = {}

        # 使用所有可用的关键点
        min_len = min(len(user_kps), len(std_kps))

        for i in range(min_len):
            # 跳过不可见的点
            if user_kps[i]['visibility'] < 0.1 or std_kps[i]['visibility'] < 0.1:
                continue

            user_pos = [user_kps[i]['x'], user_kps[i]['y']]
            std_pos = [std_kps[i]['x'], std_kps[i]['y']]

            # 计算位置差异
            diff_x = user_pos[0] - std_pos[0]
            diff_y = user_pos[1] - std_pos[1]
            distance = np.sqrt(diff_x ** 2 + diff_y ** 2)

            # 获取部位名称
            part_name = self.get_body_part_name(i)

            differences[f"point_{i}_{part_name}"] = {
                "index": i,
                "part": part_name,
                "user_position": user_pos,
                "standard_position": std_pos,
                "diff_x": diff_x,
                "diff_y": diff_y,
                "distance": distance,
                "needs_adjustment": distance > 0.08,  # 调整容忍度阈值
                "visibility": min(user_kps[i]['visibility'], std_kps[i]['visibility'])
            }

        return differences

    def get_body_part_name(self, index):
        """根据索引获取身体部位名称"""
        for part_name, indices in self.body_parts.items():
            if index in indices:
                return part_name

        # 中文映射
        chinese_names = {
            "face": "面部",
            "shoulders": "肩膀",
            "elbows": "手肘",
            "wrists": "手腕",
            "hands": "手部",
            "hips": "髋部",
            "knees": "膝盖",
            "ankles": "脚踝",
            "feet": "脚部"
        }

        return "other" if index >= 33 else "body"

    def calculate_score(self, differences):
        """计算姿势得分（0-100）"""
        if not differences:
            return 0

        # 按部位分组计算
        part_scores = {}

        for joint_name, joint_data in differences.items():
            part = joint_data.get("part", "other")

            if part not in part_scores:
                part_scores[part] = []

            # 计算该点的分数（距离越小分数越高）
            point_score = max(0, 100 - joint_data["distance"] * 300)
            part_scores[part].append(point_score)

        # 计算加权总分
        total_score = 0
        total_weight = 0

        # 不同部位的权重
        part_weights = {
            "shoulders": 1.5,
            "hips": 1.5,
            "elbows": 1.2,
            "knees": 1.2,
            "wrists": 1.0,
            "ankles": 1.0,
            "face": 0.8,
            "hands": 0.5,
            "feet": 0.5
        }

        for part, scores in part_scores.items():
            if scores:
                part_avg = np.mean(scores)
                weight = part_weights.get(part, 0.5)
                total_score += part_avg * weight
                total_weight += weight

        if total_weight == 0:
            return 0

        final_score = total_score / total_weight
        return round(min(100, final_score), 1)


    def generate_suggestions(self, differences):
        """根据关键点偏差生成可执行动作指令"""
        suggestions = []

        # 按部位分组
        part_differences = {}
        for joint_name, joint_data in differences.items():
            if joint_data["needs_adjustment"]:
                part = joint_data["part"]
                part_differences.setdefault(part, []).append(joint_data)

        # 中文部位映射
        part_translations = {
            "face": "面部",
            "shoulders": "肩膀",
            "elbows": "手肘",
            "wrists": "手腕",
            "hands": "手部",
            "hips": "髋部",
            "knees": "膝盖",
            "ankles": "脚踝",
            "feet": "脚部"
        }

        # 为每个部位生成动作指令
        for part, diffs in part_differences.items():
            if not diffs:
                continue

            chinese_part = part_translations.get(part, part)
            avg_diff_x = np.mean([d["diff_x"] for d in diffs])
            avg_diff_y = np.mean([d["diff_y"] for d in diffs])
            avg_distance = np.mean([d["distance"] for d in diffs])
            if avg_distance < 0.1:
                continue

            suggestion = {"id": part, "text": ""}

            # 根据方向生成箭头指令
            if abs(avg_diff_y) > abs(avg_diff_x) * 1.5:
                # 垂直移动
                if avg_diff_y > 0:
                    suggestion["text"] = f"⬆ 请向上移动一点（{chinese_part}偏低）"
                else:
                    suggestion["text"] = f"⬇ 请向下移动一点（{chinese_part}偏高）"
            elif abs(avg_diff_x) > abs(avg_diff_y) * 1.5:
                # 水平移动
                if avg_diff_x > 0:
                    suggestion["text"] = f"⬅ 请向左移动一点（{chinese_part}偏右）"
                else:
                    suggestion["text"] = f"➡ 请向右移动一点（{chinese_part}偏左）"
            else:
                # 对角方向
                if avg_diff_x > 0 and avg_diff_y > 0:
                    suggestion["text"] = f"↖ 请向左上移动一点（{chinese_part}偏右下）"
                elif avg_diff_x < 0 and avg_diff_y > 0:
                    suggestion["text"] = f"↗ 请向右上移动一点（{chinese_part}偏左下）"
                elif avg_diff_x > 0 and avg_diff_y < 0:
                    suggestion["text"] = f"↙ 请向左下移动一点（{chinese_part}偏右上）"
                else:
                    suggestion["text"] = f"↘ 请向右下移动一点（{chinese_part}偏左上）"

            # 补充可执行动作
            if part == "shoulders":
                suggestion["text"] += "，肩膀轻微放松后展"
            elif part == "hips":
                suggestion["text"] += "，保持骨盆中立，重心移到一侧腿"
            elif part == "face":
                suggestion["text"] += "，下巴微抬，眼神看向镜头上方3cm"

            suggestions.append(suggestion)

        return suggestions

    def get_detailed_analysis(self, differences):
        """获取详细的身体部位分析"""
        analysis = {}

        # 按部位统计
        for joint_name, joint_data in differences.items():
            part = joint_data.get("part", "other")

            if part not in analysis:
                analysis[part] = {
                    "total_points": 0,
                    "points_need_adjustment": 0,
                    "avg_distance": 0,
                    "max_distance": 0
                }

            analysis[part]["total_points"] += 1
            analysis[part]["avg_distance"] += joint_data["distance"]
            analysis[part]["max_distance"] = max(
                analysis[part]["max_distance"],
                joint_data["distance"]
            )

            if joint_data["needs_adjustment"]:
                analysis[part]["points_need_adjustment"] += 1

        # 计算平均值
        for part in analysis:
            if analysis[part]["total_points"] > 0:
                analysis[part]["avg_distance"] /= analysis[part]["total_points"]
                analysis[part]["accuracy_rate"] = (
                                                          1 - analysis[part]["points_need_adjustment"] / analysis[part][
                                                      "total_points"]
                                                  ) * 100

        return analysis



if __name__ == "__main__":
    print("=" * 60)
    print("🧘 AI姿势教练 - 姿势分析器")
    print("=" * 60)

    analyzer = PoseAnalyzer()

    if not analyzer.standard_poses:
        print("❌ 没有可用的标准姿势数据")
        print("请先运行: python pose_extractor.py")
    else:
        print(f"✅ 已成功加载 {len(analyzer.standard_poses)} 个标准姿势")
        print("可用的姿势:")
        for pose_name in analyzer.standard_poses.keys():
            print(f"  - {pose_name}")
        print("\n🎉 姿势分析器已准备好！")
        print("下一步: streamlit run app.py")