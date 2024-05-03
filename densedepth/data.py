from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch

import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import random
from itertools import permutations

_check_pil = lambda x: isinstance(x, Image.Image)

_check_np_img = lambda x: isinstance(x, np.ndarray)


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


def loadZipToMem(zip_file):
    # Load zip file into memory
    print("Loading dataset zip file...", end="")
    from zipfile import ZipFile

    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list(
        (
            row.split(",")
            for row in (data["data/nyu2_train.csv"]).decode("utf-8").split("\n")
            if len(row) > 0
        )
    )

    from sklearn.utils import shuffle

    nyu2_train = shuffle(nyu2_train, random_state=0)

    # if True: nyu2_train = nyu2_train[:40]

    print("Loaded ({0}).".format(len(nyu2_train)))
    return data, nyu2_train

def loadZipToMem_test(zip_file):
    # Load zip file into memory
    print("Loading dataset zip file...", end="")
    from zipfile import ZipFile

    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_test = list(
        (
            row.split(",")
            for row in (data["data/nyu2_test.csv"]).decode("utf-8").split("\n")
            if len(row) > 0
        )
    )

    from sklearn.utils import shuffle

    #nyu2_test = shuffle(nyu2_test, random_state=0)

    # if True: nyu2_train = nyu2_train[:40]

    print("Loaded ({0}).".format(len(nyu2_test)))
    return data, nyu2_test

class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1]]))
        sample = {"image": image, "depth": depth}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)


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
        [RandomHorizontalFlip(), RandomChannelSwap(0.5), ToTensor()]
    )


def getTrainingTestingData(path, batch_size):
    data, nyu2_train = loadZipToMem(path)

    transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
    transformed_testing = depthDatasetMemory(data, nyu2_train, transform=getNoTransform())

    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, batch_size, shuffle=False)


def load_testloader(path, batch_size=1):

    #data, nyu2_train = loadZipToMem(path)
    data, nyu2_test = loadZipToMem_test(path)
    transformed_testing = depthDatasetMemory(
        data, nyu2_test, transform=getNoTransform()
    )
    return DataLoader(transformed_testing, batch_size, shuffle=False)


#######################################################################

def DH_load_test_data(zip_path, batch_size=1):
    from zipfile import ZipFile
    
    input_zip = ZipFile(zip_path)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}

    transformed_testing = DH_depthDataset(data, 
                                          transform=transforms.Compose([DH_ToTensor()]))
    
    return DataLoader(transformed_testing, batch_size=batch_size, shuffle=True)
class DH_ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]

        image = self.to_tensor(image)
        depth = self.to_tensor(depth)
        #depth = depth.resize((320, 240)) #Do not resize to test depth images

        # if self.is_test:
        #     depth = self.to_tensor(depth).float() / 1000
        # else:
        #     depth = self.to_tensor(depth).float() * 1000

        # # put in expected range
        # depth = torch.clamp(depth, 10, 1000)

        return {"image": image, "depth": depth}
    def to_tensor(self, pic):
        if not (_check_pil(pic) or _check_np_img(pic)):
            raise TypeError(
                "pic should be PIL Image or ndarray. Got {}".format(type(pic))
            )
        if isinstance(pic, np.ndarray):
            if len(pic.shape) == 2: #depth 
                img = torch.from_numpy(pic)
                return img.float()
            else: #color
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

class DH_depthDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        # image = Image.open(BytesIO(self.data['eigen_test_rgb.npy'][idx]))
        # depth = Image.open(BytesIO(self.data['eigen_test_depth.npy'][idx]))
        # crop = torch.tensor(np.load(BytesIO(self.data['eigen_test_crop.npy'])))
        # sample = {"image": image, "depth": depth}
        # if self.transform:
        #     sample = self.transform(sample)
            
        #debug
        np_rgb = np.load(BytesIO(self.data['eigen_test_rgb.npy']))[idx]
        np_depth = np.load(BytesIO(self.data['eigen_test_depth.npy']))[idx]
        np_sample = {"image": np_rgb, "depth": np_depth}
        crop = torch.tensor(np.load(BytesIO(self.data['eigen_test_crop.npy'])))
        #sample['image'].shape
        #np_rgb[idx].shape
        if self.transform:
            sample = self.transform(np_sample)
            
        return sample, crop
    def __len__(self):
        return np.load(BytesIO(self.data['eigen_test_rgb.npy'])).shape[0]


######################## mat파일 dataloading (Eigen) ###################################



import h5py
def DH_loadMatToMemory(mat_file):
    # h5py를 사용하여 .mat 파일 로드
    with h5py.File(mat_file, 'r') as file:
        print(list(file.keys()))
        images = np.array(file['images'])
        depths = np.array(file['depths'])
        # 데이터를 (이미지, 깊이) 튜플의 리스트로 변환
        nyu2_train = [(i, i) for i in range(images.shape[0])] # 인덱스 사용
    return {'images': images, 'depths': depths}, nyu2_train
def DH_getTrainingTestingData(path, batch_size):
    data, nyu2_train = DH_loadMatToMemory(path)

    transformed_training = DH_depthDatasetMemory(
        data, nyu2_train, transform=getDefaultTrainTransform()
    )
    transformed_testing = DH_depthDatasetMemory(
        data, nyu2_train, transform=getNoTransform()
    )

    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(
        transformed_testing, batch_size, shuffle=False
    )
class DH_depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform

    def __getitem__(self, idx):
        # 이미지와 깊이 인덱스를 가져옵니다.
        image_idx, depth_idx = self.nyu_dataset[idx]
        # 실제 이미지와 깊이 데이터를 로드합니다.
        image = self.data['images'][image_idx, :, :, :]
        depth = self.data['depths'][depth_idx, :, :]
        # numpy 배열을 PIL 이미지로 변환
        image = Image.fromarray(np.uint8(image.transpose(2, 1, 0)))
        depth = Image.fromarray(np.uint8(depth.transpose(1, 0)))
        sample = {"image": image, "depth": depth}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)