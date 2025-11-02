
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from data_manager import DataManager


class ExtrinsicCalibrator:
    """相机外参标定器（眼在手外）"""
    
    def __init__(self, camera_matrix, distortion_coeffs):
        """
        参数：
            camera_matrix: 相机内参矩阵 (3×3)
            distortion_coeffs: 畸变系数
        """
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.extrinsic_matrix = None
        self.rvec = None
        self.tvec = None
    
    def compute_initial_estimate(self, pixel_coords_all, world_coords_all):
        """
        用多组PnP求解获得初始外参估计
        
        参数：
            pixel_coords_all: 所有像素坐标 {id: (N,2)}
            world_coords_all: 所有3D坐标 {id: (N,3)}
        
        返回：
            T_initial: 初始外参矩阵 (4×4)
        """
        print("\n" + "=" * 60)
        print("步骤4.1: 计算初始外参估计")
        print("=" * 60)
        
        # 随机选取若干组数据进行PnP求解
        sample_size = min(20, len(pixel_coords_all))
        sample_keys = list(pixel_coords_all.keys())[:sample_size]
        
        rvecs_list = []
        tvecs_list = []
        
        print(f"\n使用 {sample_size} 组数据进行初始估计...")
        
        for i, key in enumerate(sample_keys):
            pixel_coords = np.array(pixel_coords_all[key], dtype=np.float32)
            world_coords = np.array(world_coords_all[key], dtype=np.float32)
            
            # PnP求解
            success, rvec, tvec = cv2.solvePnP(
                world_coords,
                pixel_coords,
                self.camera_matrix,
                self.distortion_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                rvecs_list.append(rvec.flatten())
                tvecs_list.append(tvec.flatten())
                
                if (i + 1) % 5 == 0:
                    print(f"  进度: {i + 1}/{sample_size}")
        
        print(f"\n成功求解 {len(rvecs_list)} 组PnP")
        
        # 取平均（旋转向量用平均后归一化，平移直接平均）
        rvec_mean = np.mean(rvecs_list, axis=0)
        tvec_mean = np.mean(tvecs_list, axis=0)
        
        # 构建初始变换矩阵
        R_matrix = cv2.Rodrigues(rvec_mean)[0]
        T_initial = np.eye(4)
        T_initial[:3, :3] = R_matrix
        T_initial[:3, 3] = tvec_mean
        
        print(f"\n初始外参估计:")
        print(f"  旋转向量: {rvec_mean}")
        print(f"  平移向量: {tvec_mean}")
        print(f"  变换矩阵:\n{T_initial}")
        
        return T_initial, rvec_mean, tvec_mean
    
    def reprojection_error(self, params, pixel_coords_all, world_coords_all):
        """
        计算重投影误差（目标函数）
        
        参数：
            params: [rx, ry, rz, tx, ty, tz] (6个参数)
            pixel_coords_all: 所有像素坐标
            world_coords_all: 所有3D坐标
        
        返回：
            errors: 重投影误差数组 (N×2)
        """
        # 解析参数
        rvec = params[:3]
        tvec = params[3:6]
        
        # 转换为旋转矩阵
        R_matrix = cv2.Rodrigues(rvec)[0]
        
        errors = []
        
        for key in pixel_coords_all.keys():
            pixel_coords = np.array(pixel_coords_all[key], dtype=np.float32)
            world_coords = np.array(world_coords_all[key], dtype=np.float32)
            
            # 世界坐标 → 相机坐标
            camera_coords = (R_matrix @ world_coords.T).T + tvec
            
            # 投影到像素平面
            projected, _ = cv2.projectPoints(
                camera_coords,
                np.zeros(3),  # 相机坐标系中不需要旋转
                np.zeros(3),  # 不需要平移
                self.camera_matrix,
                self.distortion_coeffs
            )
            projected = projected.reshape(-1, 2)
            
            # 计算误差
            error = pixel_coords - projected
            errors.extend(error.flatten())
        
        return np.array(errors)
    
    def optimize_extrinsic(self, initial_params, pixel_coords_all, world_coords_all):
        """
        优化外参
        
        参数：
            initial_params: 初始参数 [rx, ry, rz, tx, ty, tz]
            pixel_coords_all: 所有像素坐标
            world_coords_all: 所有3D坐标
        
        返回：
            optimized_params: 优化后的参数
        """
        print("\n" + "=" * 60)
        print("步骤4.2: 非线性优化")
        print("=" * 60)
        
        print(f"\n优化设置:")
        print(f"  数据组数: {len(pixel_coords_all)}")
        print(f"  总点数: {len(pixel_coords_all) * 4}")
        print(f"  初始参数: {initial_params}")
        
        # 非线性最小二乘优化
        result = least_squares(
            fun=self.reprojection_error,
            x0=initial_params,
            args=(pixel_coords_all, world_coords_all),
            method='trf',  # Trust Region Reflective算法
            verbose=2,
            max_nfev=100  # 最大迭代次数
        )
        
        print(f"\n优化完成:")
        print(f"  迭代次数: {result.nfev}")
        print(f"  成功: {result.success}")
        print(f"  最终代价: {result.cost:.6f}")
        
        optimized_params = result.x
        print(f"  优化后参数: {optimized_params}")
        
        return optimized_params, result
    
    def evaluate_accuracy(self, params, pixel_coords_all, world_coords_all):
        """
        评估标定精度
        
        参数：
            params: 外参参数 [rx, ry, rz, tx, ty, tz]
            pixel_coords_all: 所有像素坐标
            world_coords_all: 所有3D坐标
        
        返回：
            stats: 精度统计信息
        """
        print("\n" + "=" * 60)
        print("步骤4.3: 精度评估")
        print("=" * 60)
        
        rvec = params[:3]
        tvec = params[3:6]
        R_matrix = cv2.Rodrigues(rvec)[0]
        
        all_errors = []
        error_per_pose = {}
        
        for key in pixel_coords_all.keys():
            pixel_coords = np.array(pixel_coords_all[key], dtype=np.float32)
            world_coords = np.array(world_coords_all[key], dtype=np.float32)
            
            # 投影
            camera_coords = (R_matrix @ world_coords.T).T + tvec
            projected, _ = cv2.projectPoints(
                camera_coords,
                np.zeros(3),
                np.zeros(3),
                self.camera_matrix,
                self.distortion_coeffs
            )
            projected = projected.reshape(-1, 2)
            
            # 计算欧氏距离误差
            error = np.linalg.norm(pixel_coords - projected, axis=1)
            all_errors.extend(error)
            error_per_pose[key] = np.mean(error)
        
        all_errors = np.array(all_errors)
        
        # 统计
        stats = {
            'mean_error': np.mean(all_errors),
            'std_error': np.std(all_errors),
            'max_error': np.max(all_errors),
            'min_error': np.min(all_errors),
            'median_error': np.median(all_errors),
            'rmse': np.sqrt(np.mean(all_errors**2))
        }
        
        print(f"\n重投影误差统计（像素）:")
        print(f"  平均误差: {stats['mean_error']:.3f}")
        print(f"  标准差: {stats['std_error']:.3f}")
        print(f"  中位数: {stats['median_error']:.3f}")
        print(f"  最大误差: {stats['max_error']:.3f}")
        print(f"  最小误差: {stats['min_error']:.3f}")
        print(f"  RMSE: {stats['rmse']:.3f}")
        
        # 找出误差最大的几个姿态
        sorted_poses = sorted(error_per_pose.items(), key=lambda x: x[1], reverse=True)
        print(f"\n误差最大的10个姿态:")
        for i, (key, error) in enumerate(sorted_poses[:10], 1):
            print(f"  {i}. ID {key}: {error:.3f} 像素")
        
        return stats, error_per_pose
    
    def calibrate(self, pixel_coords_all, world_coords_all):
        """
        完整的标定流程
        
        参数：
            pixel_coords_all: 所有像素坐标字典
            world_coords_all: 所有3D坐标字典
        
        返回：
            T_extrinsic: 相机外参矩阵 (4×4)
            stats: 精度统计
        """
        print("\n" + "=" * 70)
        print(" " * 20 + "相机外参标定")
        print("=" * 70)
        
        # 1. 初始估计
        T_initial, rvec_init, tvec_init = self.compute_initial_estimate(
            pixel_coords_all, 
            world_coords_all
        )
        initial_params = np.hstack([rvec_init, tvec_init])
        
        # 2. 非线性优化
        optimized_params, result = self.optimize_extrinsic(
            initial_params,
            pixel_coords_all,
            world_coords_all
        )
        
        # 3. 构建最终外参矩阵
        rvec_final = optimized_params[:3]
        tvec_final = optimized_params[3:6]
        R_final = cv2.Rodrigues(rvec_final)[0]
        
        T_extrinsic = np.eye(4)
        T_extrinsic[:3, :3] = R_final
        T_extrinsic[:3, 3] = tvec_final
        
        self.extrinsic_matrix = T_extrinsic
        self.rvec = rvec_final
        self.tvec = tvec_final
        
        # 4. 精度评估
        stats, error_per_pose = self.evaluate_accuracy(
            optimized_params,
            pixel_coords_all,
            world_coords_all
        )
        
        # 5. 显示最终结果
        print("\n" + "=" * 60)
        print("最终标定结果")
        print("=" * 60)
        print(f"\n旋转向量 (rvec):")
        print(f"  {rvec_final}")
        print(f"\n平移向量 (tvec, 单位: mm):")
        print(f"  {tvec_final}")
        print(f"\n旋转矩阵:")
        print(R_final)
        print(f"\n完整外参矩阵 (4×4):")
        print(T_extrinsic)
        
        # 转换为欧拉角（便于理解）
        euler_angles = R.from_matrix(R_final).as_euler('xyz', degrees=True)
        print(f"\n欧拉角 (XYZ顺序, 度):")
        print(f"  Rx: {euler_angles[0]:.3f}°")
        print(f"  Ry: {euler_angles[1]:.3f}°")
        print(f"  Rz: {euler_angles[2]:.3f}°")
        
        print("=" * 60)
        
        return T_extrinsic, stats


def calibrate_camera_extrinsic(data=None):
    """
    相机外参标定主流程
    
    参数：
        data: DataManager对象（可选）
    
    返回：
        data: 更新后的DataManager
        T_extrinsic: 相机外参矩阵
        stats: 精度统计
    """
    # 1. 加载数据
    if data is None:
        data_path = Path(__file__).parent.parent / 'output' / 'data.pkl'
        data = DataManager.load(data_path)
    
    # 2. 检查数据完整性
    print("\n数据检查:")
    print(f"  相机内参: {'已有' if data.camera_matrix is not None else '缺失'}")
    print(f"  像素坐标: {len(data.pixel_coords)} 组")
    print(f"  3D坐标: {len(data.world_coords)} 组")
    
    if data.camera_matrix is None:
        raise ValueError("❌ 缺少相机内参！请先执行步骤1：内参标定")
    
    if not data.pixel_coords or not data.world_coords:
        raise ValueError("❌ 缺少2D-3D对应点！请先执行步骤2和3")
    
    # 3. 检查keys是否匹配
    pixel_keys = set(data.pixel_coords.keys())
    world_keys = set(data.world_coords.keys())
    common_keys = pixel_keys & world_keys
    
    print(f"  匹配的数据组: {len(common_keys)}")
    
    if len(common_keys) < 10:
        raise ValueError(f"❌ 匹配数据太少（{len(common_keys)}组）！至少需要10组")
    
    # 4. 准备数据
    pixel_coords_all = {k: data.pixel_coords[k] for k in common_keys}
    world_coords_all = {k: data.world_coords[k] for k in common_keys}
    
    # 5. 执行标定
    calibrator = ExtrinsicCalibrator(
        data.camera_matrix,
        data.distortion_coeffs
    )
    
    T_extrinsic, stats = calibrator.calibrate(
        pixel_coords_all,
        world_coords_all
    )
    
    # 6. 保存结果
    # 可以扩展DataManager添加extrinsic_matrix字段
    data.add_extrinsic_matrix(T_extrinsic)
    
    # 保存到文件
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    np.savez(
        output_dir / 'camera_extrinsic.npz',
        extrinsic_matrix=T_extrinsic,
        rvec=calibrator.rvec,
        tvec=calibrator.tvec,
        reprojection_stats=stats
    )
    
    print(f"\n✅ 外参已保存到: {output_dir / 'camera_extrinsic.npz'}")
    
    return data


# ============================================================
# 使用示例
# ============================================================

if __name__ == '__main__':
    try:
        data = calibrate_camera_extrinsic()
        
        print("\n" + "=" * 70)
        print("🎉 相机外参标定完成！")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
