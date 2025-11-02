import numpy as np
import cv2


def warp_matrix(image, pts_src, output_size):
    """
    对四边形区域进行透视变换，使其变成正方形。
    
    参数:
    - image: 原图
    - pts_src: 四个角点 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    - output_size: 目标正方形尺寸 (宽, 高)
    
    返回:
    - warped_img: 透视变换后的正方形图像
    - M: 变换矩阵 (用于逆变换)
    """
    # 定义正方形目标点 (左上、右上、右下左下)
    pts_dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1],
    ],
                       dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(np.array(pts_src, dtype=np.float32),
                                    pts_dst)

    return M

def act_warp(image,M,output_size):
    warped_img = cv2.warpPerspective(image, M, output_size)

    return warped_img
