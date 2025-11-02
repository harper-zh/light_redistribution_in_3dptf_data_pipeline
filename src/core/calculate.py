# import pickle
# from data_manager import DataManager  # 确保 DataManager 可用

# data = DataManager.load("data.pkl") # 加载之前存的 data

# print(len(data.pixel_coords))  # 检查数据
# diagnose.py
import numpy as np
from pathlib import Path
from data_manager import DataManager

def diagnose_calibration_data():
    """诊断标定数据"""
    print("=" * 60)
    print("标定数据诊断")
    print("=" * 60)
    
    data_path = Path(__file__).parent.parent / 'output' / 'data.pkl'
    data = DataManager.load(data_path)
    
    # 1. 相机内参
    print("\n1. 相机内参:")
    print(data.camera_matrix)
    print(f"   fx = {data.camera_matrix[0, 0]:.2f}")
    print(f"   fy = {data.camera_matrix[1, 1]:.2f}")
    print(f"   cx = {data.camera_matrix[0, 2]:.2f}")
    print(f"   cy = {data.camera_matrix[1, 2]:.2f}")
    
    # 2. 像素坐标范围
    print("\n2. 像素坐标统计:")
    all_pixels = []
    for coords in data.pixel_coords.values():
        all_pixels.extend(coords)
    all_pixels = np.array(all_pixels)
    
    print(f"   总点数: {len(all_pixels)}")
    print(f"   X范围: [{all_pixels[:, 0].min():.1f}, {all_pixels[:, 0].max():.1f}]")
    print(f"   Y范围: [{all_pixels[:, 1].min():.1f}, {all_pixels[:, 1].max():.1f}]")
    print(f"   均值: ({all_pixels[:, 0].mean():.1f}, {all_pixels[:, 1].mean():.1f})")
    
    # 3. 3D坐标范围
    print("\n3. 3D坐标统计:")
    all_world = []
    for coords in data.world_coords.values():
        all_world.extend(coords)
    all_world = np.array(all_world)
    
    print(f"   总点数: {len(all_world)}")
    print(f"   X范围: [{all_world[:, 0].min():.1f}, {all_world[:, 0].max():.1f}]")
    print(f"   Y范围: [{all_world[:, 1].min():.1f}, {all_world[:, 1].max():.1f}]")
    print(f"   Z范围: [{all_world[:, 2].min():.1f}, {all_world[:, 2].max():.1f}]")
    print(f"   均值: ({all_world[:, 0].mean():.1f}, {all_world[:, 1].mean():.1f}, {all_world[:, 2].mean():.1f})")
    
    # 4. 查看具体数据
    print("\n4. 第一组数据详情:")
    first_key = sorted(data.pixel_coords.keys(), key=int)[0]
    
    print(f"   ID: {first_key}")
    print(f"   像素坐标:")
    pixel_coords = np.array(data.pixel_coords[first_key])
    for i, pc in enumerate(pixel_coords):
        print(f"     点{i+1}: ({pc[0]:.1f}, {pc[1]:.1f})")
    
    print(f"   3D坐标:")
    world_coords = np.array(data.world_coords[first_key])
    for i, wc in enumerate(world_coords):
        print(f"     点{i+1}: ({wc[0]:.1f}, {wc[1]:.1f}, {wc[2]:.1f})")
    
    # 5. 检查角点在TCP坐标系的定义
    print("\n5. 请确认角点在TCP坐标系的定义:")
    print("   你在 step3 中定义的角点坐标是:")
    corner_pts_tcp = np.array([[90, 90, 0], [-90, 90, 0], [-90, -90, 0], [90, -90, 0]])
    print(corner_pts_tcp)
    print("   单位是: mm? cm? m?")
    
    # 6. 计算单应性检验
    print("\n6. 尺度一致性检验:")
    print("   如果3D坐标单位是mm，那么:")
    print(f"     相机到物体的距离约: {all_world[:, 2].mean():.1f} mm = {all_world[:, 2].mean()/1000:.2f} m")
    print(f"   如果fx={data.camera_matrix[0,0]:.1f}是以像素为单位，")
    print(f"   那么1mm在距离{all_world[:, 2].mean():.1f}mm处，应该对应:")
    print(f"     {data.camera_matrix[0,0] / all_world[:, 2].mean():.3f} 像素")
    
    # 7. 外参结果
    if hasattr(data, 'extrinsic_matrix') and data.extrinsic_matrix is not None:
        print("\n7. 当前外参:")
        print(data.extrinsic_matrix)
        print(f"   平移向量: {data.extrinsic_matrix[:3, 3]}")
        print(f"   平移距离: {np.linalg.norm(data.extrinsic_matrix[:3, 3]):.1f}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    diagnose_calibration_data()
# import piexif
# from PIL import Image
# from pathlib import Path

# # 📌 设定文件夹路径
# base_dir = Path(__file__).parent
# img_with_exif = base_dir / "exif_img"

# images = sorted(img_with_exif.glob("*.JPG"))  # 按名称排序，确保同组图片顺序一致

# # 📌 设定新的曝光时间（你可以调整）
# exposure_values = [1/200, 1/800, 1/400, 1/100, 1/50]  # 五张图的曝光时间 (秒)

# # 📌 按 5 张一组修改 EXIF
# for i in range(0, len(images), 5):
#     group = images[i:i+5]  # 取 5 张图片
#     if len(group) < 5:
#         print(f"⚠️ 图片不足 5 张，跳过 {group}")
#         continue
    
#     print(f"🔹 处理图片组: {[img.name for img in group]}")
    
#     for img_path, new_exposure in zip(group, exposure_values):
#         try:
#             # 读取图片及 EXIF 数据
#             img = Image.open(img_path)
#             exif_dict = piexif.load(img.info.get("exif", b""))

#             # 🔄 修改曝光时间（Exif Tag 0x829A: ExposureTime）
#             exif_dict["Exif"][piexif.ExifIFD.ExposureTime] = (int(new_exposure * 1e6), int(1e6))  # (分子, 分母)

#             # 保存修改后的 EXIF 数据
#             exif_bytes = piexif.dump(exif_dict)
#             img.save(img_path, "jpeg", exif=exif_bytes)
#             print(f"✅ 修改 {img_path.name} 的曝光时间为 {new_exposure}s")

#         except Exception as e:
#             print(f"❌ 处理 {img_path.name} 时出错: {e}")
