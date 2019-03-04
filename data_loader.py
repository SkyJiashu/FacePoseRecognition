import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys
import csv
import imutils
import argparse
import glob as gb 
import torch
import random

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

# def _find_classes(self, dir):
#     """
#     Finds the class folders in a dataset.
#     Args:
#         dir (string): Root directory path.
#     Returns:
#         tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
#     Ensures:
#         No class is a subdirectory of another.
#     """
#     if sys.version_info >= (3, 5):
#         # Faster and available in Python 3.5 and above
#         classes = [d.name for d in os.scandir(dir) if d.is_dir()]
#     else:
#         classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
#     classes.sort()
#     class_to_idx = {classes[i]: i for i in range(len(classes))}
#     return classes, class_to_idx


# MY_000001_IEU+00_PD+00_EN_A0_D0_T0_BB_M0_R1_S0	 141 288 217 288
# MY_000001_IEU+00_PD+22_EN_A0_D0_T0_BB_M0_R1_S0	 170 344 241 343
# MY_000001_IEU+00_PD+45_EN_A0_D0_T0_BB_M0_R1_S0	 199 333 260 330
# MY_000001_IEU+00_PD+67_EN_A0_D0_T0_BB_M0_R1_S0	 230 282 260 280
# MY_000001_IEU+00_PD-22_EN_A0_D0_T0_BB_M0_R1_S0	 117 287 187 288
# MY_000001_IEU+00_PD-45_EN_A0_D0_T0_BB_M0_R1_S0	 99 313 157 315
# MY_000001_IEU+00_PD-67_EN_A0_D0_T0_BB_M0_R1_S0	 97 329 129 329
# MY_000001_IEU+00_PM+00_EN_A0_D0_T0_BB_M0_R1_S0	 141 222 217 220
# MY_000001_IEU+00_PM+22_EN_A0_D0_T0_BB_M0_R1_S0	 169 275 242 275
# MY_000001_IEU+00_PM+45_EN_A0_D0_T0_BB_M0_R1_S0	 201 264 259 264
# MY_000001_IEU+00_PM+67_EN_A0_D0_T0_BB_M0_R1_S0	 229 210 261 211
# MY_000001_IEU+00_PM-22_EN_A0_D0_T0_BB_M0_R1_S0	 117 220 188 218
# MY_000001_IEU+00_PM-45_EN_A0_D0_T0_BB_M0_R1_S0	 99 249 157 245
# MY_000001_IEU+00_PM-67_EN_A0_D0_T0_BB_M0_R1_S0	 96 263 130 259
# MY_000001_IEU+00_PU+00_EN_A0_D0_T0_BB_M0_R1_S0	 143 206 214 206
# MY_000001_IEU+00_PU+22_EN_A0_D0_T0_BB_M0_R1_S0	 173 258 239 260
# MY_000001_IEU+00_PU+45_EN_A0_D0_T0_BB_M0_R1_S0	 202 246 257 250
# MY_000001_IEU+00_PU+67_EN_A0_D0_T0_BB_M0_R1_S0	 227 189 263 194
# MY_000001_IEU+00_PU-22_EN_A0_D0_T0_BB_M0_R1_S0	 118 204 186 204
# MY_000001_IEU+00_PU-45_EN_A0_D0_T0_BB_M0_R1_S0	 100 232 157 230
# MY_000001_IEU+00_PU-67_EN_A0_D0_T0_BB_M0_R1_S0	 94 243 132 241

def make_dataset(dir,extensions):
    images = []
    for image in gb.glob(dir+"*.jpg"):

        dir = image[image.rindex("/")+1:]
        # print(dir)
        labels = dir
        # print(labels)
        labels = labels[labels.index("_")+1:]
        labels = labels[labels.index("_")+1:]
        labels = labels[labels.index("_")+1:]
        labels = labels[labels.index("_")+1:]
        labels = labels[0:5]
        # print(labels)
        # target = []
        # for i in range(0,33):
        #     target.append(0)
        
        # if(labels == "PD+00"):
        #     target[0] = 1
        # elif(labels == "PD+22"):
        #     target[1] = 1
        # elif(labels == "PD+45"):
        #     target[2] = 1
        # elif(labels == "PD+67"):
        #     target[3] = 1
        # elif(labels == "PD-22"):
        #     target[4] = 1
        # elif(labels == "PD-45"):
        #     target[5] = 1
        # elif(labels == "PD-67"):
        #     target[6] = 1
        # elif(labels == "PM+00"):
        #     target[7] = 1
        # elif(labels == "PM+22"):
        #     target[8] = 1
        # elif(labels == "PM+45"):
        #     target[9] = 1
        # elif(labels == "PM+67"):
        #     target[10] = 1
        # elif(labels == "PM-22"):
        #     target[11] = 1
        # elif(labels == "PM-45"):
        #     target[12] = 1
        # elif(labels == "PM-67"):
        #     target[13] = 1
        # elif(labels == "PU+00"):
        #     target[14] = 1
        # elif(labels == "PU+22"):
        #     target[15] = 1
        # elif(labels == "PU+45"):
        #     target[16] = 1
        # elif(labels == "PU+67"):
        #     target[17] = 1
        # elif(labels == "PU-22"):
        #     target[18] = 1
        # elif(labels == "PU-45"):
        #     target[19] = 1
        # elif(labels == "PU-67"):
        #     target[20] = 1
        # elif(labels == "PU-15"):
        #     target[21] = 1
        # elif(labels == "PU-30"):
        #     target[22] = 1
        # elif(labels == "PU+15"):
        #     target[23] = 1
        # elif(labels == "PU+30"):
        #     target[24] = 1
        # elif(labels == "PM-15"):
        #     target[25] = 1
        # elif(labels == "PM-30"):
        #     target[26] = 1
        # elif(labels == "PM+15"):
        #     target[27] = 1
        # elif(labels == "PM+30"):
        #     target[28] = 1
        # elif(labels == "PD-15"):
        #     target[29] = 1
        # elif(labels == "PD-30"):
        #     target[30] = 1
        # elif(labels == "PD+15"):
        #     target[31] = 1
        # elif(labels == "PD+30"):
        #     target[32] = 1
        # else:
        #     print("Unknown :"+labels)

        target = []

        if(labels == "PD+00"):
            target = [0]
        elif(labels == "PD+22"):
            target = [1]
        elif(labels == "PD+45"):
            target = [2]
        elif(labels == "PD+67"):
            target = [3]
        elif(labels == "PD-22"):
            target = [4]
        elif(labels == "PD-45"):
            target = [5]
        elif(labels == "PD-67"):
            target = [6]
        elif(labels == "PM+00"):
            target = [7]
        elif(labels == "PM+22"):
            target = [8]
        elif(labels == "PM+45"):
            target = [9]
        elif(labels == "PM+67"):
            target = [10]
        elif(labels == "PM-22"):
            target = [11]
        elif(labels == "PM-45"):
            target = [12]
        elif(labels == "PM-67"):
            target = [13]
        elif(labels == "PU+00"):
            target = [14]
        elif(labels == "PU+22"):
            target = [15]
        elif(labels == "PU+45"):
            target = [16]
        elif(labels == "PU+67"):
            target = [17]
        elif(labels == "PU-22"):
            target = [18]
        elif(labels == "PU-45"):
            target = [19]
        elif(labels == "PU-67"):
            target = [20]
        elif(labels == "PU-15"):
            target = [21]
        elif(labels == "PU-30"):
            target = [22]
        elif(labels == "PU+15"):
            target = [23]
        elif(labels == "PU+30"):
            target = [24]
        elif(labels == "PM-15"):
            target = [25]
        elif(labels == "PM-30"):
            target = [26]
        elif(labels == "PM+15"):
            target = [27]
        elif(labels == "PM+30"):
            target = [28]
        elif(labels == "PD-15"):
            target = [29]
        elif(labels == "PD-30"):
            target = [30]
        elif(labels == "PD+15"):
            target = [31]
        elif(labels == "PD+30"):
            target = [32]
        else:
            print("Unknown :"+labels)
        # print(dir+labels)
        # print(target)
        
        # if (label1 == "P"):
        #     target1 = [1]
        # else:
        #     target1 = [0]
        # print(target)
        # image = Image.open(image)
        item = (image, target)
        images.append(item)
    # for target in sorted(class_to_idx.keys()):
    #     d = os.path.join(dir, target)
    #     if not os.path.isdir(d):
    #         continue

    #     for root, _, fnames in sorted(os.walk(d)):
    #         for fname in sorted(fnames):
    #             if has_file_allowed_extension(fname, extensions):
    #                 path = os.path.join(root, fname)
    #                 item = (path, class_to_idx[target]) # ex.(path, label1, label2)
    #                 images.append(item)
    random.shuffle(images)
    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):

        # classes, class_to_idx = self._find_classes(label_root)

        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        # self.classes = classes
        # self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    # def _find_classes(self, dir):
    #     """
    #     Finds the class folders in a dataset.
    #     Args:
    #         dir (string): Root directory path.
    #     Returns:
    #         tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    #     Ensures:
    #         No class is a subdirectory of another.
    #     """     

    #     with open(dir) as csvfile:
    #         csv_reader = csv.reader(csvfile)
    #         classes = next(csv_reader)
    #         # classes = classes.split(",") No need, it is a list of str
    #     classes.sort()
    #     class_to_idx = {classes[i]: i for i in range(len(classes))}

    #     print(classes)


    #     print("\n\n\n\n")
    #     print(class_to_idx)

    #     return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # sample , target1, target2 
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # print(path)


        return sample, torch.FloatTensor(target)

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str





def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    return pil_loader(path)
    # if get_image_backend() == 'accimage':
    #     return accimage_loader(path)
    # else:
    #     return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None):
        super(ImageFolder, self).__init__(root, loader = default_loader, extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif'],
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples




# label_root = "/home/jiashu/Desktop/Attr.csv"