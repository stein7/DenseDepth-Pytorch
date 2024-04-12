import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from PIL import Image
import torch.nn as nn
import torch

'''
<<ReadMe>>
NYU-V2의 mat 확장자 dataset을 그대로 사용하는 경우
전처리 방식을 찾아서 활용 -> 가능했으나, 전처리의 차이가 발생하여 training이 원활히 되지않음.

'''
class TransposeDepthInput(object):
    def __call__(self, depth):
        depth = depth.transpose((2, 0, 1))
        depth = torch.from_numpy(depth)
        depth = depth.view(1, depth.shape[0], depth.shape[1], depth.shape[2])
        depth = nn.functional.interpolate(depth, size = (55, 74), mode='bilinear', align_corners=False)
        depth = torch.log(depth)
        # depth = (depth - depth.min())/(depth.max() - depth.min())
        return depth[0]

rgb_data_transforms = transforms.Compose([
    transforms.Resize((228, 304)),    # Different for Input Image & Depth Image
    transforms.ToTensor(),
])


depth_data_transforms = transforms.Compose([
    TransposeDepthInput(),
])

input_for_plot_transforms = transforms.Compose([
    transforms.Resize((55, 74)),    # Different for Input Image & Depth Image
    transforms.ToTensor(),
])


class NYUDataset(Dataset):
    def calculate_mean(self, images):
        mean_image = np.mean(images, axis=0)
        return mean_image

    def __init__(self, filename, type, rgb_transform = None, depth_transform = None):
        f = h5py.File(filename, 'r')
        # images_data = copy.deepcopy(f['images'][0:1449])
        # depths_data = copy.deepcopy(f['depths'][0:1449])
        # merged_data = np.concatenate((images_data, depths_data.reshape((1449, 1, 640, 480))), axis=1)

        # np.random.shuffle(merged_data)
        # images_data = merged_data[:,0:3,:,:]
        # depths_data = merged_data[:,3:4,:,:]

        images_data = f['images'][0:1449]
        depths_data = f['depths'][0:1449]

        if type == "training":
            # self.images = images_data[0:1024]
            # self.depths = depths_data[0:1024]
            self.images = images_data[0:1024]
            self.depths = depths_data[0:1024]
        elif type == "validation":
            self.images = images_data[1024:1248]
            self.depths = depths_data[1024:1248]
            # self.images = images_data[1024:1072]
            # self.depths = depths_data[1024:1072]
        elif type == "test":
            self.images = images_data[1248:]
            self.depths = depths_data[1248:]
            # self.images = images_data[0:32]
            # self.depths = depths_data[0:32]
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.mean_image = self.calculate_mean(images_data[0:1449])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # image = (image - self.mean_image)/np.std(image)
        image = image.transpose((2, 1, 0))
        # image = (image - image.min())/(image.max() - image.min())
        # image = image * 255
        # image = image.astype('uint8')
        image = Image.fromarray(image)
        if self.rgb_transform:
            image = self.rgb_transform(image)

        depth = self.depths[idx]
        depth = np.reshape(depth, (1, depth.shape[0], depth.shape[1]))
        depth = depth.transpose((2, 1, 0))
        if self.depth_transform:
            depth = self.depth_transform(depth)
        sample = {'image': image, 'depth': depth}
        return sample



import random
_check_pil = lambda x: isinstance(x, Image.Image)
_check_np_img = lambda x: isinstance(x, np.ndarray)
from itertools import permutations

class RandomHorizontalFlip(object):
    def __call__(self, sample):

        img, depth = sample["image"], sample["depth"]

        if not _check_pil(img):
            raise TypeError("Expected PIL type. Got {}".format(type(img)))
        if not _check_pil(depth):
            raise TypeError("Expected PIL type. Got {}".format(type(depth)))

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {"image": img, "depth": depth}

class RandomChannelSwap(object):
    def __init__(self, probability):

        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):

        image, depth = sample["image"], sample["depth"]

        if not _check_pil(image):
            raise TypeError("Expected PIL type. Got {}".format(type(image)))
        if not _check_pil(depth):
            raise TypeError("Expected PIL type. Got {}".format(type(depth)))

        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(
                image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])]
            )

        return {"image": image, "depth": depth}

class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]

        image = self.to_tensor(image)

        depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:
            depth = self.to_tensor(depth).float() * 1000

        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {"image": image, "depth": depth}

    def to_tensor(self, pic):
        if not (_check_pil(pic) or _check_np_img(pic)):
            raise TypeError(
                "pic should be PIL Image or ndarray. Got {}".format(type(pic))
            )

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


def getNoTransform(is_test=False):
    return transforms.Compose([ToTensor(is_test=is_test)])


def getDefaultTrainTransform():
    return transforms.Compose(
        [ToTensor()]
    )
    return transforms.Compose(
        [RandomHorizontalFlip(), RandomChannelSwap(0.5), ToTensor()]
    )
    # File "/home/sslunder0/project/NNPROJ/DenseDepth-Pytorch/densedepth/DH_process_v2.py", line 105, in __call__
    # img, depth = sample["images"], sample["depths"]
    # TypeError: 'Image' object is not subscriptable

# '/home/sslunder0/project/NNPROJ/dataset/nyu_depth_v2_labeled.mat'
def DH_getTrainingTestingData(path, batch_size):
    train_loader = torch.utils.data.DataLoader(NYUDataset( path, 
                                                        'training', 
                                                            rgb_transform = getDefaultTrainTransform(), 
                                                            depth_transform = getDefaultTrainTransform()),
                                                            batch_size = batch_size, 
                                                            shuffle = False, num_workers = 5)

    val_loader = torch.utils.data.DataLoader(NYUDataset( path,
                                                        'validation', 
                                                            rgb_transform = rgb_data_transforms, 
                                                            depth_transform = depth_data_transforms), 
                                                            batch_size = batch_size, 
                                                            shuffle = False, num_workers = 5)

    test_loader = torch.utils.data.DataLoader(NYUDataset( path,
                                                        'test', 
                                                            rgb_transform = getNoTransform(), 
                                                            depth_transform = getNoTransform()), 
                                                            batch_size = batch_size, 
                                                            shuffle = False, num_workers = 5)
    
    return train_loader, test_loader