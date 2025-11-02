import pickle
from data_manager import DataManager  # 确保 DataManager 可用

data = DataManager.load("data.pkl") # 加载之前存的 data

print(len(data.pixel_coords))  # 检查数据

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
