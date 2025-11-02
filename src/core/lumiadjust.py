import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

hdr_folder = "/Volumes/ZY/print_light/hdr"
output_path = "/Volumes/ZY/print_light/hdr_cf"
hdr_images = os.listdir(hdr_folder)

for hdr in hdr_images:
    try:
        img_path = os.path.join(hdr_folder,hdr)
        img_cf_path = os.path.join(output_path,f'{hdr}_cf.hdr')
        command = f"pcomb -s 0.9 {img_path} > {img_cf_paths}"
        subprocess.run(command,shell=True, check=True)
        print(f"已调整亮度,{img_path}")
    except subprocess.CalledProcessError as e:
        print(f"调整失败，{e}")

print("finish adjustment!")
