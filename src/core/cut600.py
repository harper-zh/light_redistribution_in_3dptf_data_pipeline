from pathlib import Path
import cv2
import os
# 获取当前脚本所在的目录
base_dir = Path(__file__).parent
output_folder = base_dir / "hdr_new"
input_folder = "I:\\print_light\\hdr_new"
# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 处理每张 HDR 图片
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".hdr"):  # 只处理 HDR 文件
        img_path = os.path.join(input_folder, filename)
        
        # 读取 HDR 图片 (cv2.IMREAD_ANYDEPTH 保持 HDR 格式)
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)

        if img is None:
            print(f"无法读取 {filename}，跳过...")
            continue

        # 确保图片尺寸正确
        h, w = img.shape[:2]
        if h != 800 or w != 800:
            print(f"{filename} 尺寸不匹配 (当前: {w}x{h})，跳过...")
            continue

        # 计算裁剪区域（中心 640）
        x_start, y_start = (w - 640) // 2, (h - 640) // 2
        x_end, y_end = x_start + 640, y_start + 640
        cropped_img = img[y_start:y_end, x_start:x_end]

        # 保存裁剪后的图片
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, cropped_img)
        print(f"已保存: {output_path}")

print("所有 HDR 图片处理完成！")