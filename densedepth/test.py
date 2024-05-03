import os
import argparse as arg
import shutil

import torch

import numpy as np
import cv2
from PIL import Image
from glob import glob

from data import load_testloader


from losses import ssim as ssim_criterion
from losses import depth_loss as gradient_criterion

from utils import colorize, DepthNorm, AverageMeter, load_images, \
                    evaluation, tensor_to_image, image_to_tensor, \
                    DH_evaluation, DH_scale_up
from data import getTrainingTestingData, \
                DH_load_test_data

def main():

    parser = arg.ArgumentParser(description="Test the model that has been trained")
    parser.add_argument("--checkpoint", "-c", type=str, help="path to checkpoint")
    parser.add_argument("--device", "-d", type=str, default="cuda")
    parser.add_argument(
        "--data", type=str, default="examples/", help="Path to dataset zip file"
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="plasma",
        help="Colormap to be used for the predictions",
    )

    args = parser.parse_args()

    if len(args.checkpoint) and not os.path.isfile(args.checkpoint):
        raise FileNotFoundError("{} no such file".format(args.checkpoint))

    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    print("Using device: {}".format(device))

    #trainloader, testloader = getTrainingTestingData(args.data, batch_size=4)
    #testloader = load_testloader(args.data)
    DH_testloader = DH_load_test_data(zip_path=args.data, batch_size=3)
    
    # Initializing the model and loading the pretrained model
    #from model import DenseDepth
    from DH_model import DenseDepth
    model = DenseDepth(encoder_pretrained=False)
    
    ckpt = torch.load(args.checkpoint, map_location=torch.device(device))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print("model load from checkpoint complete ...")

    # # Get Test Images
    # img_list = glob(args.data + "*.png")

    # # making processed image directory
    # try:
    #     os.mkdir("examples/processed/")
    # except FileExistsError:
    #     shutil.rmtree("examples/processed/")
    #     os.mkdir("examples/processed/")
    #     pass

    # save_path = "examples/processed/"

    # Set model to eval mode
    model.eval()

    # print(f"Number of images detected: {len(img_list)}")
    # Begin testing loop
    print("Begin Test Loop ...")
    
    depth_scores = torch.zeros((6, DH_testloader.dataset.__len__()))
    for idx, (batch, crop) in enumerate(DH_testloader):
        crop = crop[0]
        image_x = torch.Tensor(batch['image']).to(device)
        depth_y = torch.Tensor(batch['depth']).to(device=device)
        #normalized_depth_y = DepthNorm(depth_y)
        
        with torch.no_grad():
            preds = model(image_x)
            # Mirror image estimate
            preds_flip = model(torch.flip(image_x, [2]))
            
            preds = torch.clamp( DepthNorm(preds, max_depth=1000), 10, 1000 ) / 1000
            preds_flip = torch.clamp( DepthNorm(preds_flip, max_depth=1000), 10, 1000 ) / 1000
            
            # Up-Scaling Step
            preds_y = DH_scale_up(2, preds).squeeze(1) * 10
            preds_y_flip = DH_scale_up(2, preds_flip).squeeze(1) * 10
            
            # Cropping Step [:, 20:460, 24:616]
            true_y = depth_y[:, crop[0]:crop[1]+1, crop[2]:crop[3]+1]
            preds_y = preds_y[:, crop[0]:crop[1]+1, crop[2]:crop[3]+1] 
            preds_y_flip = preds_y_flip[:, crop[0]:crop[1]+1, crop[2]:crop[3]+1]
            
            for j in range(len(preds_y)):
                preds_avg = 0.5 * ( preds_y[j] + torch.flip(preds_y_flip[j], dims=[0]) )
        
                errors = DH_evaluation(preds_avg, true_y[j])

                for k in range(len(errors)):
                    depth_scores[k][(idx * DH_testloader.batch_size) + j] = errors[k]
    e = depth_scores.mean(axis=1)
    print("(⬆) {:>10}, {:>10}, {:>10}, (⬇) {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
    print("(⬆) {:10.4f}, {:10.4f}, {:10.4f}, (⬇) {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))

    ##############  아래는 커스텀 단일  ############
    # total_abs_rel, total_rmse, total_log10_rms, total_acc1, total_acc2, total_acc3 = 0, 0, 0, 0, 0, 0
    # batch_count = 0
    # for idx, batch in enumerate(testloader):
    #     image_x = torch.Tensor(batch["image"]).to(device)
    #     depth_y = torch.Tensor(batch["depth"]).to(device=device)
    #     normalized_depth_y = DepthNorm(depth_y)
    #     with torch.no_grad():
    #         preds = model(image_x)
    #         abs_rel, rmse, log10_rms, acc1, acc2, acc3 = evaluation(preds, normalized_depth_y)
    #         #print(f'abs_rel: {abs_rel:.4f}, rmse: {rmse:.4f}, log10_rms: {log10_rms:.4f}, acc1: {acc1:.4f}, acc2: {acc2:.4f}, acc3: {acc3:.4f}')

    #         total_abs_rel += abs_rel
    #         total_rmse += rmse
    #         total_log10_rms += log10_rms
    #         total_acc1 += acc1
    #         total_acc2 += acc2
    #         total_acc3 += acc3
    #         batch_count += 1
    # average_abs_rel = total_abs_rel / batch_count
    # average_rmse = total_rmse / batch_count
    # average_log10_rms = total_log10_rms / batch_count
    # average_acc1 = total_acc1 / batch_count
    # average_acc2 = total_acc2 / batch_count
    # average_acc3 = total_acc3 / batch_count
    
    # print(f'(AVG)abs_rel: {abs_rel:.4f}, rmse: {rmse:.4f}, log10_rms: {log10_rms:.4f}, acc1: {acc1:.4f}, acc2: {acc2:.4f}, acc3: {acc3:.4f}')

    ###########아래는 원본코드######
        # preds_origin = DepthNorm(preds)
        # preds_image = tensor_to_image(preds_origin[0])
        # preds_image.save('output/predImage.png')
        
        # preds_color = colorize(preds_origin[0])
        # tensor_to_image(preds_color).save('output/predColor.png')
    # for idx, img_name in enumerate(img_list):

    #     img = load_images([img_name])
    #     img = torch.Tensor(img).float().to(device)
    #     print("Processing {}, Tensor Shape: {}".format(img_name, img.shape))

    #     with torch.no_grad():
    #         preds = DepthNorm(model(img).squeeze(0))

    #     output = colorize(preds.data, cmap=args.cmap)
    #     output = output.transpose((1, 2, 0))
    #     cv2.imwrite(
    #         save_path + os.path.basename(img_name).split(".")[0] + "_result.png", output
    #     )

    #     print("Processing {} done.".format(img_name))


if __name__ == "__main__":
    print("Using torch version: ", torch.__version__)
    main()
