import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R
from data_manager import DataManager
from pathlib import Path


def cam_calib(w,
              h,
              square_size,
              criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        100, 0.0001),
              path=None,
              outpath=None):
    """
    说明：用来获得相机内参
    参数：
    w,h 棋盘格内角点个数
    square_size:棋盘格尺寸
    criteria：检测角点准则
    path：检测图像路径
    output_path:标记了格点后的图像

    """
    # 新增，统一所有路径以当前脚本所在目录为基准
    BASE_DIR = Path(__file__)

    if path is None:
        # path = os.path.join(BASE_DIR.parent.parent / 'data', 'img4calib')
        path = BASE_DIR.parent.parent.parent / 'data' / 'image4calib'
    else:
        path = os.path.join(BASE_DIR,
                            path) if not os.path.isabs(path) else path

    if outpath is None:
        # outpath = os.path.join(BASE_DIR.parent.parent / 'data',
        #                        'chessboard_corner')
        outpath = BASE_DIR.parent.parent.parent / 'data' / 'chessboard_corner'
    else:
        outpath = os.path.join(
            BASE_DIR, outpath) if not os.path.isabs(outpath) else outpath

    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到图片目录: {path}")
    list_path = os.listdir(path)

    # 准备棋盘格的世界坐标点
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objp *= square_size

    grays = []
    objpoints = []  # 世界坐标点
    imgpoints = []  # 图像像素点

    # 遍历图像并检测棋盘角点
    for file in list_path:
        file_name = os.path.join(path, file)
        img = cv2.imread(file_name)
        scale = 0.5  # 50%缩小
        img_resize = cv2.resize(
            img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

        gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        # gray_resize = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))
        print('image pixel', gray.shape[::-1])
        grays.append(gray)

        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        print(f"Chessboard found in {file}: {ret}")

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria)
            imgpoints.append(corners2)

            # 可视化检测到的角点（用 matplotlib 替代 OpenCV）
            img_with_corners = img_resize.copy()
            cv2.drawChessboardCorners(img_with_corners, (w, h), corners2, ret)
            cv2.imwrite(f'{outpath}\\{file}', img_with_corners)
            #使用 matplotlib 显示
            # plt.figure(figsize=(8, 6))
            # plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))  # 转换为 RGB 格式
            # plt.scatter(corners2[:, 0, 0], corners2[:, 0, 1], marker='o', c='red', s=10, label="Detected Corners")  # `s`控制点大小
            # plt.title(f"Detected Corners: {file}")
            # plt.rcParams['lines.linewidth'] = 2  # 设置线条宽度
            # plt.axis("off")  # 隐藏坐标轴
            # plt.show()

    # 相机标定
    print("\nStarting camera calibration...")

    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        grays[0].shape[::-1],
        None,
        None,
        flags=cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4
        | cv2.CALIB_FIX_K5)

    # 计算重投影误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                          camera_matrix, distortion_coeffs)
        error = cv2.norm(imgpoints[i].squeeze(), imgpoints2.squeeze(),
                         cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("\n总平均误差 (Mean Reprojection Error):", mean_error / len(objpoints))

    new_camera_matrix = camera_matrix.copy()
    new_camera_matrix[0, 0] /= scale  # fx
    new_camera_matrix[1, 1] /= scale  # fy
    new_camera_matrix[0, 2] /= scale  # cx
    new_camera_matrix[1, 2] /= scale  # cy
    # 打印结果
    print("相机矩阵 (Camera Matrix):\n", new_camera_matrix)
    print("\n畸变系数 (Distortion Coefficients):\n", distortion_coeffs)
    # 保存标定结果
    np.savez(BASE_DIR.parent.parent / 'output' / 'camera_matrix.npz',
             camera_matrix=new_camera_matrix,
             distortion_coeffs=distortion_coeffs,
             rvecs=rvecs,
             tvecs=tvecs)

    
    return new_camera_matrix, distortion_coeffs
