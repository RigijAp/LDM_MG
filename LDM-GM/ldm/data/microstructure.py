import os, yaml, pickle, shutil, tarfile, glob
import random

import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset

import taming.data.utils as tdu
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
from taming.data.imagenet import ImagePaths
import itertools
from random import sample, shuffle
from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light


def synset2idx(path_to_yaml="data/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v,k) for k,v in di2s.items())


class ImageNetBase(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self.keep_orig_class_label = self.config.get("keep_orig_class_label", False)
        self.process_images = True  # if False we skip loading & processing images and self.data contains filepaths
        self._prepare()
        self._prepare_synset_to_human()
        self._prepare_idx_to_synset()
        self._prepare_human_to_integer_label()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        raise NotImplementedError()

    def _filter_relpaths(self, relpaths):
        ignore = set([
            "n06596364_9591.JPEG",
        ])
        relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
        if "sub_indices" in self.config:
            indices = str_to_indices(self.config["sub_indices"])
            synsets = give_synsets_from_indices(indices, path_to_yaml=self.idx2syn)  # returns a list of strings
            self.synset2idx = synset2idx(path_to_yaml=self.idx2syn)
            files = []
            for rpath in relpaths:
                syn = rpath.split("/")[0]
                if syn in synsets:
                    files.append(rpath)
            return files
        else:
            return relpaths

    def _prepare_synset_to_human(self):
        SIZE = 2655750
        URL = "https://heibox.uni-heidelberg.de/f/9f28e956cd304264bb82/?dl=1"
        self.human_dict = os.path.join(self.root, "synset_human.txt")
        if (not os.path.exists(self.human_dict) or
                not os.path.getsize(self.human_dict)==SIZE):
            download(URL, self.human_dict)

    def _prepare_idx_to_synset(self):
        URL = "https://heibox.uni-heidelberg.de/f/d835d5b6ceda4d3aa910/?dl=1"
        self.idx2syn = os.path.join(self.root, "index_synset.yaml")
        if (not os.path.exists(self.idx2syn)):
            download(URL, self.idx2syn)

    def _prepare_human_to_integer_label(self):
        URL = "https://heibox.uni-heidelberg.de/f/2362b797d5be43b883f6/?dl=1"
        self.human2integer = os.path.join(self.root, "imagenet1000_clsidx_to_labels.txt")
        if (not os.path.exists(self.human2integer)):
            download(URL, self.human2integer)
        with open(self.human2integer, "r") as f:
            lines = f.read().splitlines()
            assert len(lines) == 1000
            self.human2integer_dict = dict()
            for line in lines:
                value, key = line.split(":")
                self.human2integer_dict[key] = int(value)

    def _load(self):
        with open(self.txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines()
            l1 = len(self.relpaths)
            self.relpaths = self._filter_relpaths(self.relpaths)
            print("Removed {} files from filelist during filtering.".format(l1 - len(self.relpaths)))

        self.synsets = [p.split("/")[0] for p in self.relpaths]
        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

        unique_synsets = np.unique(self.synsets)
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        if not self.keep_orig_class_label:
            self.class_labels = [class_dict[s] for s in self.synsets]
        else:
            self.class_labels = [self.synset2idx[s] for s in self.synsets]

        with open(self.human_dict, "r") as f:
            human_dict = f.read().splitlines()
            human_dict = dict(line.split(maxsplit=1) for line in human_dict)

        self.human_labels = [human_dict[s] for s in self.synsets]

        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "human_label": np.array(self.human_labels),
        }

        if self.process_images:
            self.size = retrieve(self.config, "size", default=256)
            self.data = ImagePaths(self.abspaths,
                                   labels=labels,
                                   size=self.size,
                                   random_crop=self.random_crop,
                                   )
        else:
            self.data = self.abspaths


class ImageNetTrain(ImageNetBase):
    NAME = "ILSVRC2012_train"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "a306397ccf9c2ead27155983c254227c0fd938e2"
    FILES = [
        "ILSVRC2012_img_train.tar",
    ]
    SIZES = [
        147897477120,
    ]

    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        super().__init__(**kwargs)

    def _prepare(self):
        if self.data_root:
            self.root = os.path.join(self.data_root, self.NAME)
        else:
            cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)

        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        self.expected_length = 1281167
        self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop",
                                    default=True)
        if not tdu.is_prepared(self.root):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    assert atpath == path

                print("Extracting {} to {}".format(path, datadir))
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                print("Extracting sub-tars.")
                subpaths = sorted(glob.glob(os.path.join(datadir, "*.tar")))
                for subpath in tqdm(subpaths):
                    subdir = subpath[:-len(".tar")]
                    os.makedirs(subdir, exist_ok=True)
                    with tarfile.open(subpath, "r:") as tar:
                        tar.extractall(path=subdir)

            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            tdu.mark_prepared(self.root)


class ImageNetValidation(ImageNetBase):
    NAME = "ILSVRC2012_validation"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5"
    VS_URL = "https://heibox.uni-heidelberg.de/f/3e0f6e9c624e45f2bd73/?dl=1"
    FILES = [
        "ILSVRC2012_img_val.tar",
        "validation_synset.txt",
    ]
    SIZES = [
        6744924160,
        1950000,
    ]

    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.data_root = data_root
        self.process_images = process_images
        super().__init__(**kwargs)

    def _prepare(self):
        if self.data_root:
            self.root = os.path.join(self.data_root, self.NAME)
        else:
            cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        self.expected_length = 50000
        self.random_crop = retrieve(self.config, "ImageNetValidation/random_crop",
                                    default=False)
        if not tdu.is_prepared(self.root):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    assert atpath == path

                print("Extracting {} to {}".format(path, datadir))
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                vspath = os.path.join(self.root, self.FILES[1])
                if not os.path.exists(vspath) or not os.path.getsize(vspath)==self.SIZES[1]:
                    download(self.VS_URL, vspath)

                with open(vspath, "r") as f:
                    synset_dict = f.read().splitlines()
                    synset_dict = dict(line.split() for line in synset_dict)

                print("Reorganizing into synset folders")
                synsets = np.unique(list(synset_dict.values()))
                for s in synsets:
                    os.makedirs(os.path.join(datadir, s), exist_ok=True)
                for k, v in synset_dict.items():
                    src = os.path.join(datadir, k)
                    dst = os.path.join(datadir, v)
                    shutil.move(src, dst)

            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            tdu.mark_prepared(self.root)



class MicroStructure(Dataset):
    def __init__(self, size=None,
                 degradation=None, downscale_f=4, min_crop_f=0.5, max_crop_f=1.,
                 random_crop=True, mask=False, norm=False, log=False, c_vf=False, c_noise=0.0,
                 log_param=[1.0, 1.0, 1.0, 3.5, 4.8, 4.7, 20.0, 63.0, 63.0]):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        # paths = [
        #     "/home/zhanglu/datasets/shell/shell_samples_80_80_80",
        #     "/home/zhanglu/datasets/truss/truss_samples_80_80_80",
        # ]
        # sample_paths = [os.path.join(path, name) for path in paths for name in os.listdir(path)]
        # # print("warning", paths.shape)
        # len_samples = len(sample_paths)
        # inds = list(range(len_samples))
        # random.shuffle(inds)
        # train_inds = inds[:int(len_samples * 0.8)]
        # test_inds = inds[int(len_samples * 0.8):]
        # train_inds = [425, 235, 15, 141, 878, 641, 77, 937, 561, 722, 744, 250, 70, 752, 273, 477, 896, 723, 338, 377, 553, 934, 239, 179, 439, 995, 318, 932, 851, 255, 990, 957, 1026, 706, 288, 67, 73, 353, 527, 286, 1108, 200, 354, 926, 140, 781, 580, 587, 1081, 392, 230, 361, 802, 129, 1111, 135, 494, 257, 914, 537, 236, 879, 543, 946, 188, 191, 125, 299, 573, 627, 226, 905, 860, 152, 329, 1116, 506, 840, 495, 794, 339, 717, 1019, 901, 434, 657, 745, 66, 1038, 890, 364, 341, 809, 206, 715, 991, 466, 740, 640, 903, 290, 1051, 397, 53, 82, 701, 1121, 528, 692, 348, 214, 633, 127, 418, 540, 481, 907, 522, 96, 414, 85, 1092, 685, 898, 368, 925, 383, 588, 312, 5, 285, 994, 876, 426, 176, 1099, 121, 758, 766, 162, 158, 1070, 252, 448, 45, 264, 128, 822, 237, 969, 110, 1127, 1022, 60, 714, 1075, 27, 690, 78, 608, 118, 779, 468, 841, 197, 93, 924, 29, 154, 615, 238, 955, 892, 658, 97, 1112, 718, 207, 441, 1074, 20, 885, 12, 9, 803, 560, 325, 464, 1041, 1018, 74, 843, 1126, 469, 873, 961, 242, 895, 76, 259, 768, 806, 730, 193, 664, 465, 107, 501, 449, 613, 89, 1027, 987, 912, 530, 531, 220, 269, 287, 1016, 195, 568, 315, 57, 331, 294, 1054, 153, 642, 221, 148, 831, 771, 874, 691, 454, 1024, 1035, 785, 921, 727, 228, 49, 703, 539, 100, 461, 199, 688, 445, 930, 241, 274, 183, 493, 356, 668, 815, 1078, 308, 490, 84, 787, 1050, 196, 367, 174, 310, 592, 268, 984, 1001, 694, 330, 245, 486, 229, 644, 942, 827, 217, 854, 296, 674, 606, 1052, 112, 384, 711, 1069, 167, 38, 1115, 556, 993, 732, 566, 770, 159, 966, 743, 816, 427, 1031, 328, 891, 320, 906, 513, 978, 748, 297, 311, 160, 837, 511, 458, 665, 243, 696, 1032, 499, 453, 87, 1034, 496, 23, 731, 249, 979, 923, 667, 756, 820, 857, 126, 42, 525, 406, 814, 821, 2, 68, 476, 1063, 1055, 783, 321, 247, 407, 981, 887, 371, 13, 293, 409, 1045, 380, 420, 10, 936, 612, 276, 21, 595, 953, 599, 90, 864, 832, 103, 263, 764, 350, 326, 963, 122, 988, 799, 136, 853, 138, 479, 61, 952, 651, 198, 451, 1091, 472, 102, 773, 1072, 313, 507, 246, 267, 403, 131, 823, 1, 388, 845, 248, 533, 56, 99, 295, 710, 712, 360, 581, 786, 894, 659, 492, 604, 398, 1005, 428, 734, 185, 1087, 6, 1002, 1013, 725, 630, 671, 660, 277, 1082, 1010, 720, 645, 922, 655, 1068, 1053, 1088, 866, 1094, 258, 144, 800, 754, 997, 203, 390, 289, 807, 105, 678, 760, 233, 415, 485, 471, 629, 550, 1124, 1097, 548, 463, 51, 86, 850, 681, 443, 877, 750, 444, 474, 1077, 478, 600, 31, 33, 769, 751, 1006, 917, 621, 459, 192, 382, 1106, 1011, 516, 218, 828, 618, 777, 189, 839, 209, 374, 394, 1095, 900, 517, 515, 1039, 504, 782, 536, 502, 927, 1109, 109, 498, 859, 262, 1120, 95, 792, 591, 373, 232, 834, 119, 933, 1037, 844, 423, 55, 359, 562, 497, 632, 565, 344, 11, 157, 1029, 412, 847, 182, 410, 637, 535, 281, 542, 817, 305, 998, 721, 401, 584, 108, 856, 336, 793, 958, 842, 178, 17, 385, 928, 700, 413, 106, 939, 819, 811, 307, 977, 941, 610, 909, 929, 166, 54, 457, 669, 1043, 381, 899, 1064, 682, 133, 98, 1123, 39, 343, 594, 992, 1007, 810, 1044, 470, 999, 378, 151, 869, 938, 1058, 387, 512, 210, 609, 366, 693, 520, 623, 130, 598, 101, 272, 719, 808, 607, 747, 534, 1093, 1080, 519, 438, 739, 982, 680, 161, 146, 1103, 314, 212, 170, 1073, 626, 292, 333, 736, 673, 508, 8, 761, 571, 524, 597, 362, 14, 1008, 830, 1012, 648, 663, 585, 582, 46, 1040, 614, 0, 1119, 956, 62, 500, 889, 180, 134, 1101, 698, 875, 650, 728, 586, 557, 962, 1036, 523, 36, 647, 1028, 729, 943, 861, 596, 870, 256, 1059, 115, 143, 935, 156, 973, 968, 724, 225, 22, 960, 825, 971, 559, 505, 738, 1090, 275, 791, 355, 202, 661, 797, 213, 620, 532, 918, 882, 120, 316, 656, 886, 702, 484, 433, 1056, 124, 555, 475, 526, 602, 65, 616, 147, 172, 279, 684, 687, 868, 399, 431, 1125, 58, 260, 391, 652, 1083, 3, 996, 64, 980, 976, 194, 450, 835, 622, 765, 986, 638, 549, 16, 510, 589, 376, 408, 948, 456, 303, 686, 35, 322, 251, 375, 1017, 603, 904, 483, 437, 753, 1042, 838, 59, 944, 1025, 323, 965, 165, 1117, 796, 1122, 215, 741, 951, 404, 503, 1107, 317, 389, 1086, 132, 653, 675, 1014, 319, 759, 967, 762, 224, 666, 544, 858, 679, 446, 601, 546, 227, 634, 775, 324, 436, 1030, 1066, 163, 94, 358, 137, 949, 334, 104, 1033, 778, 617, 44, 48, 789, 813, 583, 1000, 726, 865, 301, 749, 985, 699, 848, 139, 19, 186, 945, 880, 41, 173, 915, 947, 713, 639, 975, 300, 369, 695, 625, 405, 521, 1096, 184, 92, 798, 636, 150, 578, 370, 1060, 282, 81, 826, 416, 902, 480, 649, 577, 1021, 253, 1105, 871, 801, 265, 37, 7, 1114, 788, 579, 893, 123, 473, 635, 113, 379, 514, 149, 1079, 270, 222, 1118, 91, 707, 284, 551, 261, 567, 774, 489, 683, 836, 429]
        # test_inds = [43, 396, 164, 349, 280, 309, 824, 357, 417, 327, 950, 298, 1102, 619, 467, 689, 18, 795, 959, 509, 1046, 884, 63, 897, 231, 88, 1057, 271, 304, 545, 538, 181, 569, 737, 552, 972, 611, 491, 283, 716, 430, 83, 432, 1003, 964, 670, 177, 30, 697, 812, 208, 411, 888, 767, 145, 804, 1023, 386, 346, 4, 278, 1104, 460, 818, 482, 1076, 908, 240, 1047, 554, 705, 575, 345, 80, 676, 974, 1015, 254, 440, 855, 746, 572, 340, 142, 646, 25, 111, 1061, 846, 52, 563, 1071, 570, 881, 790, 1067, 419, 435, 302, 558, 1084, 763, 654, 919, 849, 983, 709, 71, 422, 829, 704, 631, 1089, 169, 833, 424, 913, 1062, 351, 780, 1048, 24, 337, 863, 204, 518, 488, 347, 211, 187, 1098, 605, 940, 402, 954, 541, 155, 75, 395, 1004, 772, 32, 708, 190, 911, 116, 455, 26, 462, 332, 47, 805, 852, 114, 69, 920, 342, 784, 205, 291, 672, 487, 1009, 244, 219, 306, 529, 447, 1113, 624, 590, 757, 223, 677, 574, 564, 79, 872, 201, 168, 234, 400, 1065, 662, 365, 372, 335, 916, 989, 363, 970, 1110, 643, 576, 1100, 742, 442, 352, 755, 72, 421, 117, 735, 883, 1049, 40, 862, 1020, 175, 216, 50, 931, 452, 28, 910, 547, 393, 266, 733, 1085, 776, 867, 628, 34, 171, 593]
        # print("train id", train_inds)
        # print("test id", test_inds)
        # self.norm = norm
        # self.flag_path = flag_path
        # self.paths = self.get_base()  #[:500]
        # sample_max = np.load("/home/zhanglu/ldm/dataset_config/C_min_max.npz")["max"]
        # self.sample_min = np.load("/home/zhanglu/ldm/dataset_config/C_min_max.npz")["min"]
        # self.sample_range = sample_max - self.sample_min
        self.sample_std = np.load("./dataset_config/C_min_max.npz")["std"] if norm else None
        self.mask = None if not mask else np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        if log:

            # # vf all
            # w_11 = 1.0
            # w_12 = 0.6
            # w_44 = 1.0
            #
            # c_11_b = 3.5
            # c_12_b = 2.9
            # c_44_b = 5.3
            #
            # c_11_w = 28.0
            # c_12_w = 78.0
            # c_44_w = 63.0

            # # vf 0.2 ~0.25
            # w_11 = 1.0
            # w_12 = 1.0
            # w_44 = 1.0
            #
            # c_11_b = 3.5
            # c_12_b = 4.8
            # c_44_b = 4.7
            #
            # c_11_w = 20.0
            # c_12_w = 63.0
            # c_44_w = 63.0

            w_11, w_12, w_44, c_11_b, c_12_b, c_44_b, c_11_w, c_12_w, c_44_w = log_param
            # print(w_11, w_12, w_44, c_11_b, c_12_b, c_44_b, c_11_w, c_12_w, c_44_w)

            self.w = np.array([[w_11, w_12, w_12, 1.0, 1.0, 1.0],
                               [w_12, w_11, w_12, 1.0, 1.0, 1.0],
                               [w_12, w_12, w_11, 1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0, w_44, 1.0, 1.0],
                               [1.0, 1.0, 1.0, 1.0, w_44, 1.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, w_44]])
            self.log_b = np.array([[c_11_b, c_12_b, c_12_b, 0.0, 0.0, 0.0],
                                   [c_12_b, c_11_b, c_12_b, 0.0, 0.0, 0.0],
                                   [c_12_b, c_12_b, c_11_b, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, c_44_b, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, c_44_b, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, c_44_b]])
            self.log_w = np.array([[c_11_w, c_12_w, c_12_w, 0.0, 0.0, 0.0],
                                   [c_12_w, c_11_w, c_12_w, 0.0, 0.0, 0.0],
                                   [c_12_w, c_12_w, c_11_w, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, c_44_w, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, c_44_w, 0.0],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, c_44_w]])
        else:
            self.log_w = None
        self.c_vf = c_vf
        self.c_n = c_noise



    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        example = np.load(self.paths[i])
        # print(example)
        # voxel = np.load(example['path'])["voxel_data"]
        # sdf = np.load(example['path'].replace("_vox", "_sdf"))["voxel_data"]
        # label = example["label"]
        # C = np.reshape(label[:36], (6, 6))
        # vf = label[36]
        # example["C"] = C  #* 20.0
        # example["vf"] = vf
        # example["voxel"] = np.expand_dims(voxel, axis=-1)
        # example["sdf"] = np.expand_dims(sdf, axis=-1)
        # print(voxel.shape, C.shape, sdf.shape, vf)
        # print(np.max(voxel), np.min(voxel))
        # print(np.max(sdf), np.min(sdf))
        # print(np.max(C), np.min(C))
        # voxel = np.expand_dims(example["voxel"], axis=-1)
        sdf = np.expand_dims(example["sdf"], axis=-1)
        # print(example["C"].shape, self.sample_min.shape, self.sample_range.shape)
        # C = (example["C"] - self.sample_min) / self.sample_range
        C = example["C"] / self.sample_std if (self.sample_std is not None) else example["C"]
        # print(C[0, 5])
        # print(np.log(C[0, 5]))
        if self.c_n > 0:
            noise = np.random.normal(1, self.c_n, size=C.shape)
            # print(np.max(noise), np.min(noise))
            noise = np.clip(noise, 0.9, 1.1)
            # print(C.shape, noise.shape, C.size)
            C = C * noise

        C = C * self.log_w + np.log(np.abs(C)) * self.w + self.log_b if (self.log_w is not None) else C
        C = C * self.mask if (self.mask is not None) else C

        C = np.concatenate([np.reshape(C, [1, -1]), np.reshape(example["vf"], [1, -1])], axis=1) if self.c_vf else C

        # print(C)
        # print(example["C"].shape, example["vf"], example["voxel"].shape, example["sdf"].shape, np.max(example["voxel"]), np.min(example["voxel"]), np.max(example["sdf"]), np.min(example["sdf"]))
        # print(C.shape, sdf.shape)
        # print("C")
        # print(C)
        # print("sdf")
        # print(sdf[0:5, 0:5, 5, 0], np.max(sdf), np.min(sdf))
        # print(C)
        return {"C": C, "sdf": sdf, "vf":example["vf"]}


# flag_path = "clean_in_stage1_thresh_0.3_vf_0.2_0.25"  #"clean_in_stage1_thresh_0.1_vf_0.25_0.32"


class MSTrain(MicroStructure):
    def __init__(self, flag_path, ori_str="", rpl_str="", **kwargs):
        super().__init__(**kwargs)
        self.flag_path = flag_path
        self.ori_str = ori_str
        self.rpl_str = rpl_str
        self.paths = self.get_base()  # [:500]

    def get_base(self):
        print(os.getcwd())
        f = open("{}_all_samples.txt".format(self.flag_path), "r")
        lines = f.readlines()
        example_paths = [line.split("\n")[0] for line in lines]
        if len(self.ori_str) > 0:
            example_paths = [item.replace(self.ori_str, self.rpl_str) for item in example_paths]
        return example_paths


class MSValidation(MicroStructure):
    def __init__(self, flag_path=None, ori_str="", rpl_str="", **kwargs):
        super().__init__(**kwargs)
        self.flag_path = flag_path
        self.ori_str = ori_str
        self.rpl_str = rpl_str
        self.paths = self.get_base()  # [:500]

    def get_base(self):
        f = open("{}_test.txt".format(self.flag_path), "r")
        lines = f.readlines()
        example_paths = [line.split("\n")[0] for line in lines]
        if len(self.ori_str) > 0:
            example_paths = [item.replace(self.ori_str, self.rpl_str) for item in example_paths]
        return example_paths
