import numpy as np
import cv2
import os
from pathlib import Path

def corner_pixel_extract(file_path):
    """
    检测样品的四个角点，并返回角点的像素坐标
    file是照片的名字
    file_path是照片所处文件夹的路径
    output_path是绘制四边形后照片的保存路径
    """
    points = []
    sorted_points = []
    img = cv2.imread(file_path)
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 应用高斯模糊（减少噪声）
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 二值化处理
    retval, threshold_img = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    # 形态学操作，去除小噪点
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, kernel)
    # 轮廓检测
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        peri = cv2.arcLength(contour, True)  # 计算轮廓周长
        # print(peri)
        if peri > 1000:
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)  # 轮廓逼近

            if len(approx) == 4:  # 只保留四边形
                print("检测到四边形，顶点坐标：")

                for point in approx:
                    x, y = point[0]
                    points.append([x, y])
                    print(f"({x}, {y})")
                    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # 标记四边形顶点（红色）
                cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)  # 绿色绘制四边形
                # 保存检测结果

                
                base_dir = Path(__file__).parent
                output_path = base_dir.parent.parent / 'data'/'draw_corner'
                cv2.imwrite(output_path/f'{file_path.name}')
                print(f"处理完成，结果已保存: {output_path}")
                sum_sorted = sorted(points, key=lambda p: p[0] + p[1])

                # 按照 x - y 的值排序（右上角点的 x-y 最大，左下角点 x-y 最小）
                diff_sorted = sorted(points, key=lambda p: p[0] - p[1])

                # 识别四个角点
                top_left = sum_sorted[0]  # 左上角点 (x+y 最小)
                bottom_right = sum_sorted[-1]  # 右下角点 (x+y 最大)
                top_right = diff_sorted[-1]  # 右上角点 (x-y 最大)
                bottom_left = diff_sorted[0]  # 左下角点 (x-y 最小)
                sorted_points = [
                    top_left, top_right, bottom_right, bottom_left
                ]
    return sorted_points
