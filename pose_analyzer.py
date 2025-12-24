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
        """根据差异生成改进建议"""
        suggestions = []

        # 按部位分组
        part_differences = {}

        for joint_name, joint_data in differences.items():
            if joint_data["needs_adjustment"]:
                part = joint_data["part"]
                if part not in part_differences:
                    part_differences[part] = []
                part_differences[part].append(joint_data)

        # 中文部位名称映射
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

        # 为每个部位生成建议
        for part, diffs in part_differences.items():
            if not diffs:
                continue

            chinese_part = part_translations.get(part, part)

            # 计算该部位的平均偏移方向
            avg_diff_x = np.mean([d["diff_x"] for d in diffs])
            avg_diff_y = np.mean([d["diff_y"] for d in diffs])
            avg_distance = np.mean([d["distance"] for d in diffs])

            if avg_distance < 0.1:
                continue  # 差异太小，不生成建议

            # 生成建议
            suggestion = {
                "id": part,  # 使用部位作为唯一标识符
                "text": "",  # 建议内容
            }

            if abs(avg_diff_y) > abs(avg_diff_x) * 1.5:
                # 垂直方向差异更大
                if avg_diff_y > 0:
                    suggestion["text"] = f"你的{chinese_part}整体位置偏高啦，试着微微放低并后移重心，能让体态更舒展自然~"  # 加入“重心转移”（书里核心技巧）
                else:
                    suggestion["text"]  = f"你的{chinese_part}整体位置偏低啦，轻轻抬高并让身体微侧（避开正对镜头），比例会更协调~"  # 加入“身体微侧避僵硬”（书里基础原则）
            elif abs(avg_diff_x) > abs(avg_diff_y) * 1.5:
                # 水平方向差异更大
                if avg_diff_x > 0:
                    suggestion["text"]  = f"你的{chinese_part}整体偏右啦，轻轻向左调整，同时让手臂与身体留些空隙（避免紧贴显宽），平衡感会更好~"  # 加入“负空间”（书里避误区技巧）
                else:
                    suggestion["text"]  = f"你的{chinese_part}整体偏左啦，轻轻向右调整，搭配肩部微微放松下沉，体态会更舒展协调~"  # 加入“肩颈放松”（书里面部+身体摆姿）
            else:
                # 对角方向差异
                if avg_diff_x > 0 and avg_diff_y > 0:
                    suggestion["text"]  = f"你的{chinese_part}整体偏右上方啦，向左下方调整的同时，让重心移到后脚，能让体态更稳更协调~"  # 加入“重心转移”
                elif avg_diff_x < 0 and avg_diff_y > 0:
                    suggestion["text"]  = f"你的{chinese_part}整体偏左上方啦，往右下方向调整，同时让身体微侧15°（避开正对镜头的僵硬感），姿态会更自然~"  # 加入“身体微侧”
                elif avg_diff_x > 0 and avg_diff_y < 0:
                    suggestion["text"]  = f"你的{chinese_part}整体偏右下方啦，向左上方调整，搭配腿部微微弯曲（创造曲线感），状态会更松弛好看~"  # 加入“曲线创造”（书里女士美姿）
                else:
                    suggestion["text"]  = f"你的{chinese_part}整体偏左下方啦，往右上方向调整，同时轻抬下巴（避免双下巴），整体体态会更精致~"  # 加入“下巴调整”（书里面部摆姿）


            # 添加补充建议到每个部位的建议中
            if part == "shoulders":
                suggestion["text"] += " 试着轻轻放松肩膀并微微后展，让手臂与身体留些空隙（避免紧贴显宽），整个人会更松弛自然~"
            elif part == "hips":
                suggestion["text"] += " 试着保持骨盆中立，同时让重心移到一侧腿上（避免僵硬），还能悄悄弱化臀部的视觉宽度~"
            elif part == "face":
                suggestion["text"] += " 保持面部自然放松，轻抬下巴并让眼神看向镜头上方3cm（更灵动不生硬），状态会更精致好看~"
            suggestions.append(suggestion)


        return suggestions[:5]  # 最多返回5条建议

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


# def test_pose_analyzer():
#     """测试姿势分析器"""
#     print("=" * 60)
#     print("🧪 姿势分析器测试")
#     print("=" * 60)
#
#     analyzer = PoseAnalyzer()
#
#     if not analyzer.standard_poses:
#         print("❌ 没有可用的标准姿势数据")
#         return
#
#     print(f"\n📚 可用的标准姿势: {list(analyzer.standard_poses.keys())}")
#
#     # 用第一个标准姿势模拟用户姿势（加一些噪声）
#     first_pose_name = list(analyzer.standard_poses.keys())[0]
#     std_keypoints = analyzer.standard_poses[first_pose_name]["keypoints"]
#
#     print(f"\n🔬 测试姿势: {first_pose_name}")
#     print(f"关键点数量: {len(std_keypoints)}")
#
#     # 创建模拟的用户姿势（添加一些随机差异）
#     import random
#     user_keypoints = []
#     for kp in std_keypoints:
#         user_keypoints.append({
#             "id": kp["id"],
#             "x": kp["x"] + random.uniform(-0.08, 0.08),  # 添加随机噪声
#             "y": kp["y"] + random.uniform(-0.08, 0.08),
#             "z": kp["z"],
#             "visibility": kp["visibility"]
#         })
#
#     print("\n📊 开始姿势对比分析...")
#
#     # 进行分析
#     result = analyzer.compare_poses(user_keypoints, first_pose_name)
#
#     if "error" in result:
#         print(f"❌ 分析出错: {result['error']}")
#         return
#
#     print(f"\n📈 分析结果:")
#     print(f"  🎯 标准姿势: {result['standard_pose']}")
#     print(f"  📊 得分: {result['score']}/100")
#     print(f"  ✅ 是否合格: {'是' if result['is_good'] else '否'}")
#
#     if result['suggestions']:
#         print(f"\n💡 改进建议:")
#         for i, suggestion in enumerate(result['suggestions'], 1):
#             print(f"  {i}. {suggestion}")
#     else:
#         print(f"\n🎉 姿势完美！")
#
#     # 显示详细分析
#     if 'detailed_analysis' in result:
#         print(f"\n🔍 详细部位分析:")
#         for part, data in result['detailed_analysis'].items():
#             print(f"  {part}: 准确率{data.get('accuracy_rate', 0):.1f}%")
#
#     print(f"\n✅ 测试完成")


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