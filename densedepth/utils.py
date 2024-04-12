import numpy as np
import matplotlib
import matplotlib.cm as cm
from PIL import Image

import torch

def evaluation(pred, gt):
    average_relative_error = torch.mean( torch.abs(gt - pred) / gt )
    root_mean_square = torch.sqrt( torch.mean( (gt - pred) ** 2 ) )
    # 최종출력 픽셀이 0 혹은 음수가 나올 수 있으므로 clamp를 수행
    log_rms = torch.mean( torch.abs( torch.log10( torch.clamp(gt, min=1e-6) ) - torch.log10( torch.clamp(pred, min=1e-6) ) ) )
    
    threshold = torch.max( torch.abs(gt / pred) , torch.abs(pred / gt) )
    accuracy = torch.mean( (threshold < 1.25).float() )
    accuracy2 = torch.mean( (threshold < 1.25 ** 2).float() )
    accuracy3 = torch.mean( (threshold < 1.25 ** 3).float() )
    
    return average_relative_error, root_mean_square, log_rms, accuracy, accuracy2, accuracy3

def DepthNorm(depth, max_depth=1000.0):
    return max_depth / depth


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def colorize(value, vmin=10, vmax=1000, cmap="binary"):

    value = value.cpu().numpy()[0, :, :]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0

    cmapper = cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def load_from_checkpoint(ckpt, model, optimizer, epochs, loss_meter=None):

    checkpoint = torch.load(ckpt)
    ckpt_epoch = epochs - (checkpoint["epoch"] + 1)
    if ckpt_epoch <= 0:
        raise ValueError(
            "Epochs provided: {}, epochs completed in ckpt: {}".format(
                epochs, checkpoint["epoch"] + 1
            )
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])

    return model, optimizer, ckpt_epoch


def init_or_load_model(
    depthmodel,
    enc_pretrain,
    epochs,
    lr,
    ckpt=None,
    device=torch.device("cuda:0"),
    loss_meter=None,
):

    if ckpt is not None:
        checkpoint = torch.load(ckpt)

    model = depthmodel(encoder_pretrained=enc_pretrain)

    if ckpt is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if ckpt is not None:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])

    start_epoch = 0
    if ckpt is not None:
        start_epoch = checkpoint["epoch"] + 1
        if start_epoch <= 0:
            raise ValueError(
                "Epochs provided: {}, epochs completed in ckpt: {}".format(
                    epochs, checkpoint["epoch"] + 1
                )
            )

    return model, optimizer, start_epoch


def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(
            np.asarray(Image.open(file).resize((640, 480)), dtype=float) / 255, 0, 1
        ).transpose(2, 0, 1)

        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)
