import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# 📌 设定路径
# dir_path = "/Volumes/ZY/print_light/exif_img"
# hdr_folder = "/Volumes/ZY/print_light/hdr"

dir_path = "/Volumes/ZY/print_light/pre"
hdr_folder = "/Volumes/ZY/print_light/pre"
ldr_images = os.listdir(dir_path)
# 📌 预处理：按编号分组，避免重复遍历
image_groups = defaultdict(list)

for image in ldr_images:
    try:
        idx = int(image.split("_")[0])  # 解析文件编号
        image_groups[idx].append(os.path.join(dir_path, image))
    except ValueError:
        print(f"⚠️ 跳过无效文件: {image}")

# 📌 定义 HDR 生成函数
def generate_hdr(idx, image_paths):
    if len(image_paths) < 5:  # HDR 需要 5 张不同曝光的图片
        print(f"⚠️ 跳过 {idx}, 图片不足 5 张")
        return

    hdr_path = os.path.join(hdr_folder, f"{idx}.hdr")
    command = f"hdrgen {' '.join(image_paths)} -o {hdr_path} -r response_function.rsp -a -e -f -g"

    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ 生成 HDR: {hdr_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 生成失败 {idx}: {e}")

# 📌 使用多线程并行处理（加速）
MAX_WORKERS = 8  # 线程数，可调整
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for idx, image_paths in image_groups.items():
        executor.submit(generate_hdr, idx, image_paths)

print("🚀 HDR 生成任务完成！")
