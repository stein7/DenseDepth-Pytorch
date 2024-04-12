cd /home/sslunder0/project/NNPROJ/DenseDepth-Pytorch/densedepth/
CUDA_VISIBLE_DEVICES=2 python3 train.py --batch 7 --save output/ --data /home/sslunder0/project/NNPROJ/dataset/nyu_data.zip #nyu_depth_v2_labeled.mat
CUDA_VISIBLE_DEVICES=1 python3 test.py --checkpoint /home/sslunder0/project/NNPROJ/DenseDepth-Pytorch/densedepth/output/#2_batch-7/ckpt_19_15.pth