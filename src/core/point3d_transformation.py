import itertools
import numpy as np
from scipy.spatial.transform import Rotation as R
# from core.检查照片缺失 import photo_folder
from data_manager import DataManager
import os
from datetime import datetime
from pathlib import Path


def detect_missing_pose(photo_folder,
                        expected_photos_per_pose=5,
                        max_time_gap=12):
    photos = []
    missing_indices = []
    current_pose_index = 0
    missing_item_count = 0
    #扫描照片文件夹，获取时间排序的照片列表
    for filename in os.listdir(photo_folder):
        if filename.endswith('.JPG'):
            filepath = os.path.join(photo_folder, filename)
            photo_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            photos.append((filename, photo_time))
    photos.sort(key=lambda x: x[1])

    for i in range(0,
                   len(photos) - expected_photos_per_pose,
                   expected_photos_per_pose):
        # 当前组和下一组
        current_group = photos[i:i + expected_photos_per_pose]
        next_group = photos[i + expected_photos_per_pose:i +
                            2 * expected_photos_per_pose]

        # 确保当前组和下一组都有5张照片
        if len(current_group) == expected_photos_per_pose and len(
                next_group) == expected_photos_per_pose:
            # 获取当前组的最后一张照片和下一组的第一张照片的时间
            current_group_end_time = current_group[-1][1]
            next_group_start_time = next_group[0][1]

            # 计算两组之间的时间差
            time_diff = (next_group_start_time -
                         current_group_end_time).total_seconds()

            # 检查时间差是否超过最大允许时间
            if time_diff > max_time_gap:
                print(
                    f"Time gap between path point {current_pose_index} and {current_pose_index + 1} is > {max_time_gap}s:"
                )
                print(
                    f"    Last photo in point {current_pose_index}: {current_group[-1][0]} - {current_group_end_time}"
                )
                print(
                    f"    First photo in point {current_pose_index + 1}: {next_group[0][0]} - {next_group_start_time}"
                )
                missing_item_count += 1
                missing_index = current_pose_index + missing_item_count
                missing_indices.append(missing_index)
        current_pose_index += 1  # 递增到下一个路径点
    print(f"缺失姿态数：{missing_item_count}")
    print(f"缺失姿态索引{missing_indices}")
    return missing_indices


def pose_transform_matirx():
    euler = np.array([-20, -10, 0, 10, 20])
    #欧拉角组合
    euler_combos = list(itertools.product(euler, repeat=3))
    # print(euler_combos)

    #生成网格点
    grid_size = 50
    grid_extend = [-2, -1, 0, 1, 2]
    grid = np.array([(i * grid_size, j * grid_size) for i in grid_extend
                     for j in grid_extend])
    grid_reshaped = grid.reshape(len(grid_extend), len(grid_extend), 2)
    for i in range(len(grid_extend)):
        if i % 2 == 1:  # 奇数列从上到下
            grid_reshaped[i, :] = grid_reshaped[i, ::-1]
    grid_3d = np.insert(grid_reshaped, 2, 0, axis=2)
    grid_flatten = grid_3d.reshape(-1, 3)
    #倒序的网格点
    reverse_grid = grid_flatten[::-1]
    # print(reverse_grid)

    #从欧拉角到变换矩阵
    rotations = [
        R.from_euler('zyx', angles, degrees=True) for angles in euler_combos
    ]

    T_list = []
    for rotation in rotations:
        R_matrix = rotation.as_matrix()

        i = rotations.index(rotation)  #第i种姿态
        T = np.eye(4)
        T[:3, :3] = R_matrix
        if i % 2 == 0:
            translation = grid_flatten
        else:
            translation = reverse_grid
        for g in translation:
            T[:3, 3] = g
            T_list.append(T.copy())
    print(f"共生成{len(T_list)}种变换矩阵")
    return T_list


def object_coord_calculate(T, p_t):
    """
    说明：计算某一位姿下四个角点在基坐标系下的3d坐标
    T：该位姿下TOOL相对于BASE的变换矩阵，即TOOL在BASE中的表示
    p_t:TOOL坐标系中四个角点的坐标
    """
    #将3D点转换为齐次坐标
    origin_points_h = np.hstack([p_t, np.ones((p_t.shape[0], 1))])
    # 使用矩阵乘法进行坐标变换
    #去掉齐次坐标中的最后一列，转换回普通的3D坐标
    object_points = np.dot(T, origin_points_h.T).T
    return object_points[:, :3]


def process_coords():
    #1，筛选未被拍摄的点位
    BASE_DIR = Path(__file__)
    photo_folder = BASE_DIR.parent.parent.parent / 'data' / 'images'
    #之后应该使用75%的运行速度，执行下面这一条，而不是手动输入。本次手动是因为操作失误导致某些姿态之间转换时间太长。
    # missing_indecies = detect_missing_pose(photo_folder)
    missing_indecies = [51, 126, 722, 1414, 2203]
    corner_pts_base = []
    #2，计算每个位姿的变换矩阵
    T_pose = pose_transform_matirx()
    T_pose_filtered = [
        item for i, item in enumerate(T_pose) if i not in missing_indecies
    ]

    #3，计算四个角点在基坐标系中的坐标
    #左上，右上，右下，左下
    corner_pts = np.array([[90, 90, 0], [-90, 90, 0], [-90, -90, 0],
                           [90, -90, 0]])
    for T in T_pose_filtered:
        pts = object_coord_calculate(T, corner_pts)
        corner_pts_base.append(pts)

    #4,检索角点提取成功的id，把该id对应的索引的3d坐标存储到data
    data_path = Path(__file__).parent.parent / 'output' / 'data.pkl'
    data = DataManager.load(data_path)
    keys = list(data.pixel_coords.keys())
    for key in keys:
        try:
            # 将key转换为整数索引
            idx = int(key)
            
            # ✅ 检查索引是否在有效范围内
            if 0 <= idx < len(corner_pts_base):
                data.add_3d_points(key, corner_pts_base[idx])
                
            else:
                print(f"  ⚠ 警告: key {key} 对应的索引 {idx} 超出范围")
                
                
        except (ValueError, IndexError) as e:
            print(f"  ⚠ 处理 key {key} 时出错: {e}")
            

    if data.world_coords:
        data.save(data_path)
        print(f"3d点坐标已保存，共有{len(data.world_coords)}组")
    else:
        print("没有点坐标可保存")
    return data
# data = DataManager.load("data.pkl")
# indexs = data.pixel_coords.keys()
# T_list_final = [filtered_list[int(idx)] for idx in indexs]#根据存取数据的id，选取成功识别了像素点的位置，
# print(len(T_list_final))

# def object_points_calculate(T_ref, T_pose, point_board):
#     """
#     说明：计算某一位姿下棋盘格点在参考坐标系下的3d坐标
#     T_ref:参考坐标系相对于base的变换矩阵
#     T_pose该位姿下tcp相对于base的变换矩阵
#     point_board：棋盘格点在tcp坐标系里的坐标
#     """
#     # 将3D点转换为齐次坐标
#     origin_points_h = np.hstack(
#         [point_board, np.ones((point_board.shape[0], 1))])
#     # 使用矩阵乘法进行坐标变换
#     #去掉齐次坐标中的最后一列，转换回普通的3D坐标
#     object_points_base_inv = np.dot(T_pose, origin_points_h.T)
#     object_points_o1 = np.dot(np.linalg.inv(T_ref), object_points_base_inv).T
#     return object_points_o1[:, :3]
