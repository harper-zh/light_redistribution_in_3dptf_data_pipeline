import os
import time
dir_path = 'H:\\DCIM\\101CANON'
list_path = os.listdir(dir_path)


for i in range(0,len(list_path)):
    group = i // 5 
    num = i % 5 
    
    path = os.path.join(dir_path, list_path[i])
    new_path = os.path.join(dir_path, f'{group + 1555}_{num}.JPG')
    # new_path = os.path.join(dir_path, f'{group}_{num}.JPG')

    

    try:
        os.rename(path, new_path)
    except PermissionError:
        print(f"Skipping {path}: file is in use.")
        time.sleep(1) 