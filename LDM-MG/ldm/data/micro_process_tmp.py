import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D



def check_C(sample_C, ther=0.1, ther_abs_C11=0.0, ther_abs_C12=1.0e-5, ther_abs_C44=1.0e-4):

    if sample_C[0,0] > 1 or sample_C[1,1] > 1 or sample_C[2,2] > 1 or sample_C[3, 3] > 1 or sample_C[4, 4] > 1 or sample_C[5, 5] > 1:
        return False

    C_44_avg = (sample_C[3, 3] + sample_C[4, 4] + sample_C[5, 5]) / 3.0
    if C_44_avg < ther_abs_C44:
        return False
    if np.abs(sample_C[3, 3] -C_44_avg) / C_44_avg > ther or np.abs(sample_C[4, 4] -C_44_avg) / C_44_avg > ther or np.abs(sample_C[5, 5] -C_44_avg) / C_44_avg > ther:
        return False

    C_11_avg = (sample_C[0, 0] + sample_C[1, 1] + sample_C[2, 2]) / 3.0
    if C_11_avg < ther_abs_C11:
        return False
    if np.abs(sample_C[0, 0] -C_11_avg) / C_11_avg > ther or np.abs(sample_C[1, 1] -C_11_avg) / C_11_avg > ther or np.abs(sample_C[2, 2] -C_11_avg) / C_11_avg > ther:
        return False

    C_12_avg = (sample_C[0, 1] + sample_C[1, 2] + sample_C[0, 2] + sample_C[1, 0] + sample_C[2, 1] + sample_C[2, 0]) / 6.0
    if C_12_avg < ther_abs_C12:
        return False
    if np.abs(sample_C[0, 1] -C_12_avg) / C_12_avg > ther or np.abs(sample_C[1, 2] -C_12_avg) / C_12_avg > ther or np.abs(sample_C[0, 2] -C_12_avg) / C_12_avg > ther:
        return False
    if np.abs(sample_C[1, 0] -C_12_avg) / C_12_avg > ther or np.abs(sample_C[2, 1] -C_12_avg) / C_12_avg > ther or np.abs(sample_C[2, 0] -C_12_avg) / C_12_avg > ther:
        return False

    return True

# paths = [
#     "/home/zhanglu/datasets/shell/C_s_vf.npz",
#     "/home/zhanglu/datasets/truss/C_s_vf.npz",
# ]
# labels = np.concatenate([np.load(path)["C_vf"] for path in paths], axis=0)
# # print("warning", labels.shape)
# sample_paths = np.concatenate([np.load(path)["paths"] for path in paths])
#
# print(len(sample_paths))
#
# for ind, path in enumerate(sample_paths):
#     voxel = np.load(path)["voxel_data"]
#     sdf = np.load(path.replace("_vox", "_sdf"))["voxel_data"]
#     label = labels[ind]
#
#     # print(ind, label.shape, voxel.shape, sdf.shape, path)
#
#     np.savez(path.replace("_vox", "_samples"), C=np.reshape(label[:36], (6, 6)), vf=label[36], sdf=sdf, voxel=voxel)
#     print("{} saved!".format(path.replace("_vox", "_samples")))

paths = [
            "/mnt/zhanglu/inverse/datasets/shell/shell_80_80_80/",
            "/mnt/zhanglu/inverse/datasets/truss/truss_80_80_80/",
        ]
flag_hist = "st_all_clean"
sample_paths = [os.path.join(path, name) for path in paths for name in sorted(os.listdir(path))]

# f = open("./dataset_config/clean_in_stage1_thresh_0.3_vf_0.2_0.25_all_samples.txt", "r")
# lines = f.readlines()
# sample_paths = [line.split("\n")[0] for line in lines]
# flag_hist = "clean_in_stage1_thresh_0.3_vf_0.2_0.25_log"


# print("warning", paths.shape)
# len_samples = len(sample_paths)
# inds = list(range(len_samples))
# random.shuffle(inds)
# train_inds = inds[:int(len_samples * 0.8)]
# test_inds = inds[int(len_samples * 0.8):]
# train_paths = [sample_paths[i] for i in train_inds]
# test_paths = [sample_paths[i] for i in test_inds]

paths = sample_paths  #[:500]  #[:30926]  #[:6926]  #[:50]

C_max = np.load(paths[0])["C"] - 100.0
C_min = np.load(paths[0])["C"] + 100.0
C_sq = np.zeros(C_max.shape, dtype=C_max.dtype)

# print(C_max)
# print(C_min)

C11_list = []
C12_list = []
C44_list = []
# C11_ori_list = []
# C12_ori_list = []
# C44_ori_list = []
clean_paths = []
vf_list = []

bad_case_num = 0

ther = 0.1
ther_abs_C11 = 0.0
ther_abs_C12 = 1.0e-5
ther_abs_C44 = 1.0e-4

for path in tqdm(paths):
    sample_C = np.load(path)["C"]
    vf = np.load(path)["vf"]

    # if vf < 0.2 or vf > 0.3:
    #     continue
    if vf < 0.085:
        continue

    if not check_C(sample_C, ther=0.1, ther_abs_C11=0.0, ther_abs_C12=1.0e-5, ther_abs_C44=1.0e-4):
        bad_case_num += 1
        continue

    clean_paths.append(path)
    # C11_list.append((28 * sample_C[0, 0]) + (np.log(sample_C[0, 0])) + 3.5)
    # C12_list.append((78 * sample_C[0, 1]) + 0.6 * (np.log(sample_C[0, 1])) + 2.9)
    # C44_list.append((63 * sample_C[3, 3]) + (np.log(sample_C[3, 3])) + 5.3)
    C11_list.append(sample_C[0, 0])
    C12_list.append(sample_C[0, 1])
    C44_list.append(sample_C[3, 3])
    # if C12_list[-1] < 0.000009:
    #     print(path)
    #     print(sample_C)
    C_max = np.maximum(C_max, sample_C)
    C_min = np.minimum(C_min, sample_C)
    # C_sq = (C_sq + sample_C ** 2.0)

    vf_list.append(vf)

print("C11 MAX MIN", max(C11_list), min(C11_list))
print("C12 MAX MIN", max(C12_list), min(C12_list))
print("C44 MAX MIN", max(C44_list), min(C44_list))

print(C_max)
print(C_min)
# print(np.sqrt(C_sq / len(clean_paths)))
print(bad_case_num)

np.savez("./dataset_config/C_vf_{}".format(flag_hist),
         C11=np.concatenate([np.reshape(item, [1, ]) for item in C11_list], axis=0),
         C12=np.concatenate([np.reshape(item, [1, ]) for item in C12_list], axis=0),
         C44=np.concatenate([np.reshape(item, [1, ]) for item in C44_list], axis=0),
         vf=np.concatenate([np.reshape(item, [1, ]) for item in vf_list], axis=0))

# np.savez("./dataset_config/C_min_max_{}".format(flag_hist), min=C_min, max=C_max, std=np.sqrt(C_sq / len(clean_paths)))
#
# # hist image
# plt.hist(C11_list, bins=100)
# plt.savefig("./dataset_config/{}_C11_HIST.png".format(flag_hist))
# plt.cla()
# plt.hist(C12_list, bins=100)
# plt.savefig("./dataset_config/{}_C12_HIST.png".format(flag_hist))
# plt.cla()
# plt.hist(C44_list, bins=100)
# plt.savefig("./dataset_config/{}_C44_HIST.png".format(flag_hist))
# plt.cla()
# plt.hist(vf_list, bins=100)
# plt.savefig("./dataset_config/{}_vf_HIST.png".format(flag_hist))

# scatter vf-C
flag_vf_C_scatter = "C11" #  "vf-c11"   "vf-c12"   "vf-c44"
dict_vf_scatter = {"C11": C11_list, "C12": C12_list, "C44": C44_list}
C_list = dict_vf_scatter[flag_vf_C_scatter]

g_pairs = [
    [0.128, 0.0441],
    [0.2107, 0.0863],
    [0.216, 0.0839],
    [0.24, 0.0945],
    [0.156, 0.0569],
    [0.1456, 0.054435],
    [0.14, 0.0531],
    [0.15, 0.0598]
]
g_vf = [item[0] for item in g_pairs]
g_c = [item[1] for item in g_pairs]
g_paths = [
    "/mnt/zhanglu/inverse/ldm/test_results_st_all_log_1_ddim_generated_contidion_nn_all.npz",
    "/mnt/zhanglu/inverse/ldm/test_results_st_all_log_ddim_generated_contidion_nn_all.npz",
    ]
c_rand = np.concatenate([np.load(g_path)["res"][:, 6:, :] for g_path in g_paths], axis=0)
vf_rand = np.concatenate([np.load(g_path)["vf"] for g_path in g_paths], axis=0)
print(c_rand.shape, vf_rand.shape)
g_c_1 = []
g_vf_1 = []
for i in range(vf_rand.shape[0]):
    if not check_C(c_rand[i], ther=0.1, ther_abs_C11=1.0e-5, ther_abs_C12=1.0e-5, ther_abs_C44=1.0e-4):
        continue
    if vf_rand[i] < 0.085:
        continue
    g_c_1.append(c_rand[i, 0, 0]* 1.0)
    g_vf_1.append(vf_rand[i])
g_c_1.extend(g_c)
g_vf_1.extend(g_vf)

# upbound_g_c, upbound_g_vf = get_upbound(g_vf_1, g_c_1)

plt.figure(figsize=(16, 10))
# plt.scatter(g_vf, g_c, color='brown', s=2, marker='^')
plt.scatter(g_vf_1, g_c_1, color='brown', s=2, marker='^', label="Generated")
plt.scatter(vf_list, C_list, color='g', s=2, marker='o', label="Dataset", alpha=0.7)

plt.legend()
plt.xlabel('volume fraction')
plt.ylabel(flag_vf_C_scatter)
plt.title('{}-vf Scatter'.format(flag_vf_C_scatter))

plt.savefig("./dataset_config/vf-{}-{}.png".format(flag_vf_C_scatter, flag_hist))



# # 3D scatter image
# C11_array = np.asarray(C11_list)
# # print(C11_array.shape, C11_array.dtype, np.max(C11_array), np.min(C11_array))
# C12_array = np.asarray(C12_list)
# C44_array = np.asarray(C44_list)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(C44_array, C11_array, C12_array, s=0.1)
# ax.set_zlabel('C12', fontdict={'size': 15, 'color': 'red'})
# ax.set_ylabel('C11', fontdict={'size': 15, 'color': 'red'})
# ax.set_xlabel('C44', fontdict={'size': 15, 'color': 'red'})
# # plt.show()
# plt.savefig("scatter.png")

# print(len(paths), len(clean_paths))



# # split trainset/valset
# len_samples = len(clean_paths)
# inds = list(range(len_samples))
# random.shuffle(inds)
# train_f = 0.985
# train_inds = inds[:int(len_samples * train_f)]
# test_inds = inds[int(len_samples * train_f):]
# train_paths = [clean_paths[i] for i in train_inds]
# test_paths = [clean_paths[i] for i in test_inds]
# print("[INFO] train set include {} samples, test set include {} samples!".format(len(train_paths), len(test_paths)))
#
# with open("./dataset_config/{}_train.txt".format(flag_hist), 'w') as f:
#     for i in train_paths:
#         f.write(i+'\n')
#
# with open("./dataset_config/{}_test.txt".format(flag_hist), 'w') as f:
#     for i in test_paths:
#         f.write(i+'\n')
#
# with open("./dataset_config/{}_all_samples.txt".format(flag_hist), 'w') as f:
#     for i in clean_paths:
#         f.write(i+'\n')
#
# train_samples = len(test_paths)
# train_samples_num = len(train_paths)
# inds = list(range(train_samples_num))
# random.shuffle(inds)
# train_inds = inds[:train_samples]
# train_paths = [train_paths[i] for i in train_inds]
#
# with open("./dataset_config/{}_train_sample_{}.txt".format(flag_hist, train_samples), 'w') as f:
#     for i in train_paths:
#         f.write(i+'\n')


# def calculate_x(v2, G1, G2, K2, K1):
#     v1 = 1 - v2
#     a = 6 * (K2 + 2 * G2) * v2
#     b = 5 * G2 * (3 * K2 + 4 * G2)
#     c = a / b
#     d = 1 / (G1 - G2)
#     e = v1 / (d + c)
#     f = G2 + e
#
#     g = 3 * v2 / (3 * K2 + 4 * G2)
#     h = 1 / (K1 - K2)
#     i = v1 / (h + g)
#     j = i + K2
#
#     x = 9 * j * f / (3 * j + f)
#     return x
#
# # 定义v2的取值范围
# v2_values = np.linspace(0.05, 0.4, 100)
#
#
# E1 = 1e-5
# E2 = 1
# nu1 = 0.3
# nu2 = 0.3
#
# def compute_K_G(E, nu):
#     K = E / (3 * (1 - 2 * nu))  # 体积模量
#     G = E / (2 * (1 + nu))     # 剪切模量
#     return K, G
#
# # 计算对应的x值
# K1, G1 = compute_K_G(E1, nu1)
# K2, G2 = compute_K_G(E2, nu2)
#
# x_values = [calculate_x(v2, G1, G2, K2, K1) for v2 in v2_values]

