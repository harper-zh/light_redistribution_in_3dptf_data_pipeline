# from pathlib import Path
# from data_manager import DataManager
# from camera_calib import cam_calib
# from pixel_processing import corner_pixel_extract
# from image_cut_warp import warp_matrix, act_warp
# import cv2
# import numpy as np
# from PIL import Image
# import piexif
# # 获取当前脚本所在的目录
# base_dir = Path(__file__).parent

# # 定义输入和输出文件夹
# img4calib = base_dir / "image4calib"
# chessboard_corner = base_dir / "chessboard_corner"
# img4pnp = base_dir / "image4pnp"
# draw_corner = base_dir / "draw_corner"
# warped_img = base_dir / "warped_img"
# imgmassive = base_dir / "images"
# img_with_exif = base_dir / "exif_img"
# # 确保输出文件夹存在
# chessboard_corner.mkdir(exist_ok=True)
# draw_corner.mkdir(exist_ok=True)
# warped_img.mkdir(exist_ok=True)


# def save_corner_pixel(img):
#     pixel = corner_pixel_extract(img)
#     if pixel:
#         pts = np.array(pixel, dtype=np.float32)
#         if pts.shape != (4, 2):  # 确保有 4 个点
#             print(
#                 f"Error: Expected 4 points, but got {pts.shape}. Skipping {img.name}."
#             )
#             return
#         else:
#             id = img.stem.split("_")[0]
#             print("id是:", id)
#             data.add_pixel_coords(id, pixel)
#             return pts


# def transform_img(match, ):
#     #读取原图像EXIF数据
#     original_pil = Image.open(match)
#     exif_dict = piexif.load(original_pil.info.get("exif", b""))

#     image = cv2.imread(str(match))
#     if image is None:
#         print(f"Error: Image {match} could not be loaded.")
#         return  # 跳过处理

#     warp_matrix = warp_matrix(image, pts, output_size=(800, 800))
#     #保存变换矩阵数据
#     data.add_warp_matrix(id, warp_matrix)
#     # warped = act_warp(image, warp_matrix, output_size=(800, 800))
#     # if warped is None:
#     #     print(f"Error: Warped image is None for {match}. Skipping save.")
#     #     return

#     # output_path = warped_img / match.name
#     # exif_path = img_with_exif / match.name
#     # cv2.imwrite(str(output_path), warped)
#     # # 用 PIL 重新加载 OpenCV 保存的图片
#     # warped_pil = Image.open(output_path)

#     # # 重新写入 EXIF 数据
#     # exif_bytes = piexif.dump(exif_dict)
#     # warped_pil.save(exif_path, "jpeg", exif=exif_bytes)
#     # print(" 继承 EXIF 数据！")




# def act_img():
#     # 1. 初始化数据管理器
#     path = Path(__file__).parent.parent / 'output' / 'data.pkl'
#     data = DataManager.load(path
#         )
#     imgs4pixel = list(imgmassive.glob("*_0*.JPG"))
#     imgs4warp = list(imgmassive.glob("*.JPG"))

#     for img in imgs4pixel:

#         pts = save_corner_pixel(img)
#         if pts:
#             # 切割图像计算，透视变换矩阵
#             match_imgs = [
#                 match for match in imgs4warp if match.stem.split("_")[0] == id
#             ]
#             print(match_imgs)

#             for match in match_imgs:
#                 transform_img(match)
#         else:
#             print("没有提取到角点")
#             continue
#     if data.pixel_coords and data.warp_matrices:
#         data.save(path)


# if  __name__ == "__main__":
#     data = None

#     try:
#         data = main()
#     except KeyboardInterrupt:
#         print("\n⚠️ 检测到 Ctrl+C，中断前先保存数据...")
#     finally:
#         if data:  # 只有 data 不为空才执行保存
#             data.save("data.pkl")
#             print("✅ 数据已保存！")
#         else:
#             print("❌ 没有数据可保存，可能 main() 没有执行完成。")
from pathlib import Path
from data_manager import DataManager
from camera_calib import cam_calib
from pixel_processing import corner_pixel_extract
from image_cut_warp import warp_matrix as compute_warp_matrix  # ✅ 重命名避免冲突
from image_cut_warp import act_warp
import cv2
import numpy as np
from PIL import Image
import piexif

# 获取图像所在的目录
base_dir = Path(__file__).parent
img_dir = base_dir.parent.parent / 'data'
# 定义输入和输出文件夹
img4calib = img_dir / "image4calib"
chessboard_corner = img_dir / "chessboard_corner"
img4pnp = img_dir / "image4pnp"
draw_corner = img_dir / "draw_corner"
warped_img = img_dir / "warped_img"
imgmassive = img_dir / "images"
img_with_exif = img_dir / "exif_img"

# 确保输出文件夹存在
chessboard_corner.mkdir(exist_ok=True)
draw_corner.mkdir(exist_ok=True)
warped_img.mkdir(exist_ok=True)
img_with_exif.mkdir(exist_ok=True)  # ✅ 添加这个


def save_corner_pixel(img, data_manager):
    """
    提取并保存角点像素坐标
    
    Args:
        img: 图像路径 (Path对象)
        data_manager: DataManager 对象
        
    Returns:
        tuple: (id, pts) 或 (None, None)
    """
    pixel = corner_pixel_extract(img)
    
    if not pixel:
        print(f"⚠ 未检测到角点: {img.name}")
        return None, None
    
    pts = np.array(pixel, dtype=np.float32)
    
    if pts.shape[0] != 4:  # ✅ 检查点数
        print(f"❌ 错误: 需要4个点，但得到{pts.shape[0]}个点: {img.name}")
        return None, None
    
    # 提取图像ID
    img_id = img.stem.split("_")[0]
    print(f"✓ 检测到角点，图像ID: {img_id}")
    
    # 保存到 DataManager
    data_manager.add_pixel_coords(img_id, pixel)
    
    return img_id, pts


def transform_img(img_path, pts, img_id, data_manager):
    """
    计算透视变换矩阵并保存
    
    Args:
        img_path: 图像路径
        pts: 角点坐标
        img_id: 图像ID
        data_manager: DataManager 对象
        
    Returns:
        bool: 是否成功
    """
    # 读取图像
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"❌ 无法加载图像: {img_path}")
        return False
    
    # 计算变换矩阵
    try:
        matrix = compute_warp_matrix(image, pts, output_size=(800, 800))  # ✅ 使用重命名的函数
        
        # 保存变换矩阵
        data_manager.add_warp_matrix(img_id, matrix)
        print(f"  ✓ 已保存变换矩阵: {img_path.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 计算变换矩阵失败: {img_path.name}, 错误: {e}")
        return False


def save_warped_image_with_exif(img_path, pts, output_dir, exif_output_dir):
    """
    可选：保存变换后的图像并保留EXIF数据
    
    Args:
        img_path: 原图像路径
        pts: 角点坐标
        output_dir: 输出目录
        exif_output_dir: 带EXIF的输出目录
    """
    try:
        # 读取原图像EXIF数据
        original_pil = Image.open(img_path)
        exif_dict = piexif.load(original_pil.info.get("exif", b""))
        
        # 读取并变换图像
        image = cv2.imread(str(img_path))
        matrix = compute_warp_matrix(image, pts, output_size=(800, 800))
        warped = act_warp(image, matrix, output_size=(800, 800))
        
        if warped is None:
            print(f"⚠ 变换失败: {img_path.name}")
            return False
        
        # 保存变换后的图像
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), warped)
        
        # 用 PIL 重新加载并添加 EXIF
        warped_pil = Image.open(output_path)
        exif_bytes = piexif.dump(exif_dict)
        exif_path = exif_output_dir / img_path.name
        warped_pil.save(exif_path, "jpeg", exif=exif_bytes)
        
        print(f"  ✓ 已保存变换图像(含EXIF): {img_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ 保存图像失败: {img_path.name}, 错误: {e}")
        return False


def process_images():
    """
    主处理流程：提取角点并计算变换矩阵
    """
    print("=" * 60)
    print("图像处理流程：提取角点和计算变换矩阵")
    print("=" * 60)
    
    # 1. 加载 DataManager
    data_path = Path(__file__).parent.parent / 'output' / 'data.pkl'
    data = DataManager.load(data_path)
    
    # 2. 获取图像列表
    imgs4pixel = list(imgmassive.glob("*_0*.JPG"))  # ✅ 排序，便于追踪
    imgs4warp = list(imgmassive.glob("*.JPG"))
    
    print(f"\n找到 {len(imgs4pixel)} 张用于角点检测的图像")
    print(f"找到 {len(imgs4warp)} 张总图像")
    
    if not imgs4pixel:
        print("❌ 没有找到图像文件！")
        return data
    
    # 3. 处理每张图像
    processed_count = 0
    failed_count = 0
    
    for img in imgs4pixel:
        print(f"\n处理图像: {img.name}")
        
        # 3.1 提取角点
        img_id, pts = save_corner_pixel(img, data)
        
        if pts is None:
            failed_count += 1
            continue
        
        # 3.2 找到同一ID的所有图像
        match_imgs = [
            match for match in imgs4warp 
            if match.stem.split("_")[0] == img_id
        ]
        
        print(f"  找到 {len(match_imgs)} 张匹配图像")
        
        # 3.3 对每张匹配图像计算变换矩阵
        for match in match_imgs:
            success = transform_img(match, pts, img_id, data)
            if success:
                processed_count += 1
                
                # 可选：保存变换后的图像
                # save_warped_image_with_exif(match, pts, warped_img, img_with_exif)
    
    # 4. 汇总统计
    print("\n" + "=" * 60)
    print("处理完成")
    print(f"  成功处理: {processed_count} 张图像")
    print(f"  失败: {failed_count} 张图像")
    print(f"  保存的像素坐标: {len(data.pixel_coords)} 组")
    print(f"  保存的变换矩阵: {len(data.warp_matrices)} 个")
    print("=" * 60)
    
    # 5. 保存数据
    if data.pixel_coords or data.warp_matrices:
        data.save(data_path)
        print(f"✅ 数据已保存")
    else:
        print("⚠ 没有数据可保存")
    
    return data


