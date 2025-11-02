from camera_calib import cam_calib
import cv2
from pathlib import Path
from scipy.optimize import least_squares 
from solvepnp_ba import cam_solve, project_points,reprojection_error
from data_manager import DataManager  # 确保 DataManager 可用
from point3d_transformation import object_points_calculate 


#获取像素坐标
data = DataManager.load("data.pkl") # 加载之前存的 data
image_points = [data.pixel_coords[i] for i in range(50)]

#计算3d点的坐标及相机的位姿


# 获取当前脚本所在的目录
base_dir = Path(__file__).parent
img_folder = base_dir / "image4calib"
output_path = base_dir / "chessboard_corner"

#标定相机内参
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100,
            0.0001)
camera_matrix, distortion_coeffs = cam_calib(w=9,
                                             h=11,
                                             square_size=45,
                                             criteria=criteria,
                                             path=img_folder,
                                             outpath=output_path)

# 初始化优化变量 (20 组 rvec 和 tvec)
initial_params = []
for i in range(20):
    initial_params.extend(rvec_list[i].flatten())  # 旋转向量
    initial_params.extend(tvec_list[i].flatten())  # 平移向量

initial_params = np.array(initial_params)

# 进行优化
result = least_squares(reprojection_error, initial_params, 
                       args=(object_points, image_points_all, camera_matrix, distortion_coeffs),
                       method='lm')  # Levenberg-Marquardt 方法

# 提取优化后的 rvec 和 tvec
optimized_rvecs = []
optimized_tvecs = []

for i in range(20):
    rvec = result.x[i * 6: i * 6 + 3].reshape(3, 1)
    tvec = result.x[i * 6 + 3: i * 6 + 6].reshape(3, 1)
    optimized_rvecs.append(rvec)
    optimized_tvecs.append(tvec)

print("优化后的相机位姿计算完成！")