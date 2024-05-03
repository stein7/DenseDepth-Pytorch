cd /home/sslunder0/project/Depth-Estimation/DenseDepth-Pytorch/densedepth/
CUDA_VISIBLE_DEVICES=0 python3 train.py --batch 7 --save output/Real_Model169/ --data /home/sslunder0/project/Datasets/NYU-V2/nyu_data.zip 
#nyu_depth_v2_labeled.mat

CUDA_VISIBLE_DEVICES=1 python3 test.py --data /home/sslunder0/project/Datasets/NYU-V2/nyu_test.zip \
    --checkpoint /home/sslunder0/project/Depth-Estimation/DenseDepth-Pytorch/densedepth/output/#2_batch-7_Model161/ckpt_19_15.pth

    --checkpoint /home/sslunder0/project/Depth-Estimation/DenseDepth-Pytorch/densedepth/output/Model161/ckpt_14_17.pth

    --checkpoint /home/sslunder0/project/Depth-Estimation/DenseDepth-Pytorch/densedepth/output/Real_Model169/ckpt_13_18.pth

    --checkpoint /home/sslunder0/project/Depth-Estimation/DenseDepth-Pytorch/densedepth/output/Model169+Shuffle/ckpt_19_16.pth