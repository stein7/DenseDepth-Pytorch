from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch

import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import random
from itertools import permutations


''' ReadMe
전처리방식을 이해하기위해, mini 데이터셋 만들어서 독립적으로 돌리면서 테스트한 코드
'''


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
            depth = self.to_tensor(depth).float() * 1000 # 이부분에서 imshow의 왜곡발생함

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

    #nyu2_train = shuffle(nyu2_train, random_state=0)

    # if True: nyu2_train = nyu2_train[:40]

    print("Loaded ({0}).".format(len(nyu2_train)))
    return data, nyu2_train


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

def imshow(sample):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(F.to_pil_image(sample['image']).convert("RGB"))
    # axes[1].imshow(F.to_pil_image(sample['depth']).convert("RGB"))
    
    plt.figure()
    plt.imshow(F.to_pil_image(sample['image']).convert("RGB"))
    plt.figure()
    plt.imshow(F.to_pil_image(sample['depth']).convert("RGB"))
    plt.show()
def getTrainingTestingData(path, batch_size):
    data, nyu2_train = loadZipToMem(path)

    transformed_training = depthDatasetMemory(
        data, nyu2_train, transform=getDefaultTrainTransform()
    )
    transformed_testing = depthDatasetMemory(
        data, nyu2_train, transform=getNoTransform()
    )

    return DataLoader(transformed_training, batch_size, shuffle=False), DataLoader(
        transformed_testing, batch_size, shuffle=False
    )   
        
if __name__ == "__main__":
    trainloader, testloader = getTrainingTestingData("/home/sslunder0/project/NNPROJ/dataset/nyu_minidata.zip", 5)
    
    for idx, batch in enumerate(trainloader):
        
        print(batch["image"].shape)
        print(batch["depth"].shape)
        
# import torch
# ...

# #1. arg parsing 
# parser = arg.ArgumentParser("--arg", type=str, help="descriptions")
# args = parser.parse_args()

# #2. dataloading
# trainloader, testloader = getTrainingTestingData(args.data, batch_size=args.batch)

# #3. model, optimizer instance or load
# model, optimizer, start_epoch = init_or_load_model()

# l1_criterion == nn.L1Loss()

# #4. training Loop
# for epoch in range(args.epochs):
#     model.train()
#     model = model.to(device)
    
#     for idx, batch in enumerate(trainloader):
#         optimizer.zero_grad()
        
#         image_x = torch.Tensor(batch["image"]).to(device)
#         depth_y = torch.Tensor(batch["depth"]).to(device)
#         normalized_depth_y = DepthNorm(depth_y)
        
#         preds = model(image_x)
        
#         l1_loss = l1_criterion(preds, normalized_depth_y)
    
#         net_loss.backward()
#         optimizer.step()
# #5. etc logging, write, model save, 

