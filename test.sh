# test places365
CUDA_VISIBLE_DEVICES=0 python main_ddp_eval.py
--data data/places365_standard --model OSFA --base_name resnet50_places365 --clip_name ViT-B/32 \
--base_dim 2048 --clip_dim 512 --hidden_dim 512 --num_classes 67 --test_bs 67 \
--load_pth ./work_dirs/OSFA_resnet50_places365_ViTB32_places365_standard/OSFA_best.pth

# test mit
CUDA_VISIBLE_DEVICES=0 python main_ddp_eval.py\
--data data/MITIndoor67 --model OSFA --base_name resnet50_places365 --clip_name ViT-B/32 \
--base_dim 2048 --clip_dim 512 --hidden_dim 512 --num_classes 67 --test_bs 67 \
--load_pth ./work_dirs/OSFA_resnet50_places365_ViTB32_MITIndoor67/OSFA_best.pth

# test sun
CUDA_VISIBLE_DEVICES=0 python main_ddp_eval.py\
--data data/SUN397 --model OSFA --base_name resnet50_places365 --clip_name ViT-B/32 \
--base_dim 2048 --clip_dim 512 --hidden_dim 512 --num_classes 397 --test_bs 397 \
--load_pth ./work_dirs/OSFA_resnet50_places365_ViTB32_SUN397/OSFA_best.pth