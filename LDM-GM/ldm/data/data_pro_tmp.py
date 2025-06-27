import numpy as np
import os
from tqdm import tqdm

paths = [
            "/home/zhanglu/datasets/shell/shell_samples_80_80_80",
            "/home/zhanglu/datasets/truss/truss_samples_80_80_80",
        ]
save_dirs = [
    "/home/zhanglu/datasets/shell/shell_80_80_80",
    "/home/zhanglu/datasets/truss/truss_80_80_80",
]
for save_dir in save_dirs:
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

sample_paths = [os.path.join(path, name) for path in paths for name in sorted(os.listdir(path))]
paths = sample_paths

for path in tqdm(paths):

    if 'truss' in path:
        save_path = os.path.join(save_dirs[1], os.path.basename(path))
    elif 'shell' in path:
        save_path = os.path.join(save_dirs[0], os.path.basename(path))
    else:
        print("[ERROR] not truss or shell!")

    sample = np.load(path)
    np.savez(save_path, C=sample['C'], vf=sample['vf'], sdf=sample['sdf'])
