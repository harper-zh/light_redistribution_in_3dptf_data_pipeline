from operator import imod
from data_manager import DataManager
from camera_calib import cam_calib
from pathlib import Path
from act_img import process_images
from point3d_transformation import process_coords
from extrinsic_calibration import calibrate_camera_extrinsic


def main():
    """
    执行相机内外参标定、获取所有数据并存储
    """
    #一，加载数据
    data = DataManager.load(
        Path(__file__).parent.parent / 'output' / 'data.pkl')
    print(f"  相机内参: {'已有' if data.camera_matrix is not None else '未标定'}")
    #二，获取相机内参
    if data.camera_matrix is None:
        camera_matrix, distortion_coeffs = cam_calib(
            w=9,
            h=11,
            square_size=45,
        )
        data.add_camera_matrix(camera_matrix, distortion_coeffs)
        print("内参标定完成并已存储到 DataManager")
    else:
        camera_matrix, distortion_coeffs = data.camera_matrix, data.distortion_coeffs
    #三，获取像素角点
    if data.pixel_coords is None:
        data = process_images()
        pixel_coords = data.pixel_coords
        print("像素角点检测及坐标提取已完成")
    else:
        pixel_coords = data.pixel_coords
        print("像素角点及坐标：已有")
    #四，获取3d角点
    if not data.world_coords:
        data = process_coords()
        world_coords = data.world_coords
        print("3d角点坐标计算完成")
    else:
        world_coords = data.world_coords
        print("3d角点坐标已有")
        # print(list(world_coords.items()))
    #五，计算相机外参
    if data.extrinsic_matrix is None:
        data = calibrate_camera_extrinsic()
        extrinsic_matrix = data.extrinsic_matrix
        print(f"相机外参计算完成，外参为：{extrinsic_matrix}")

    else:
        extrinsic_matrix = data.extrinsic_matrix
        print(f"相机外参已有，外参为：{extrinsic_matrix}")
        
    return data


if __name__ == '__main__':
    data = None

    try:
        data = main()
    except KeyboardInterrupt:
        print("\n⚠️ 检测到 Ctrl+C，中断前先保存数据...")
    finally:
        if data:  # 只有 data 不为空才执行保存

            data.save(Path(__file__).parent.parent/'output'/'data.pkl')
            print("✅ 数据已保存！")
        else:
            print("❌ 没有数据可保存，可能 main() 没有执行完成。")
