import cv2
import numpy as np
import cv2


def cam_solve( object_points, image_points, camera_matrix,
              distortion_coeffs):
    """调用pnp函数，返回rvec,tvec"""
    _, rvec, tvec, inliners = cv2.solvePnPRansac(object_points,
                                                 image_points.squeeze(),
                                                 camera_matrix,
                                                 distortion_coeffs,
                                                 reprojectionError=8.0,
                                                 confidence=0.99,
                                                 iterationsCount=150)
    if not _:
        print("PnP 求解失败")
    else:
        return rvec, tvec

def project_points(rvec, tvec, object_points, camera_matrix, distortion_coeffs):
    """使用 Rodrigues 公式计算旋转矩阵并投影 3D 点"""
    img_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion_coeffs)
    return img_points.reshape(-1, 2)

def reprojection_error(params, object_points, image_points_all, camera_matrix, distortion_coeffs):
    """
    计算所有相机位姿的重投影误差
    params: 优化变量 (20 个相机的 rvec 和 tvec)
    """
    num_cameras = len(image_points_all)
    errors = []

    for i in range(num_cameras):
        rvec = params[i * 6: i * 6 + 3].reshape(3, 1)  # 旋转向量
        tvec = params[i * 6 + 3: i * 6 + 6].reshape(3, 1)  # 平移向量
        
        # 投影 3D 点
        img_points_proj = project_points(rvec, tvec, object_points, camera_matrix, distortion_coeffs)
        
        # 计算误差
        error = (img_points_proj - image_points_all[i]).ravel()  # 误差拉平成 1 维数组
        errors.append(error)
    
    return np.concatenate(errors)  # 合并误差


