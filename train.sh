## train places with resnet
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29502 --num_processes 4  main_ddp_train.py \
--data data/places365_standard  --epochs 20 \
--milestones 10 15 -b 128 --test_bs 365 --lr 0.01 \
--model OSFA --base_name resnet50_places365 --clip_name ViT-B/32 \
--base_dim 2048 --clip_dim 512 --hidden_dim 512 --num_classes 365

## train places with pvt
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29502 --num_processes 4  main_ddp_train.py \
--data data/places365_standard  --epochs 20 \
--milestones 10 15 -b 128 --lr 0.01 \
--model OSFA --base_name pvt_b2 --clip_name ViT-B/32 \
--base_dim 2048 --clip_dim 512 --hidden_dim 512 --num_classes 365 --test_bs 365

## train mit
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29502 --num_processes 4  main_ddp_train.py \
--data data/MITIndoor67  --epochs 20 \
--milestones 10 15 -b 128 --lr 0.01 \
--model OSFA --base_name resnet50_places365 --clip_name ViT-B/32 \
--base_dim 2048 --clip_dim 512 --hidden_dim 512 --num_classes 67 --test_bs 67 \
--load_pth ./work_dirs/OSFA_resnet50_places365_ViTB32_places365_standard/OSFA_best.pth

## train sun397
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29502 --num_processes 4  main_ddp_train.py \
--data data/SUN397  --epochs 20 \
--milestones 10 15 -b 128 --lr 0.01 \
--model OSFA --base_name resnet50_places365 --clip_name ViT-B/32 \
--base_dim 2048 --clip_dim 512 --hidden_dim 512 --num_classes 397 --test_bs 397 \
--load_pth ./work_dirs/OSFA_resnet50_places365_ViTB32_places365_standard/OSFA_best.pth

