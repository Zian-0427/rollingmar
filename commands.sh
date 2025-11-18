# Evaluate time
export eval_bsz=1; export num_iter=100; CUDA_VISIBLE_DEVICES=7 python main_mar.py --img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 400 --warmup_epochs 100 --batch_size 392 --blr 1.0e-4 --diffusion_batch_mul 4 --output_dir output_dir/rollingmar  --data_path ${IMAGENET_PATH} --online_eval  --num_images 10000  --num_iter ${num_iter} --evaluate --eval_bsz ${eval_bsz} --enable_timer > bs${eval_bsz}_iter${num_iter}.log


# Evaluate
export eval_bsz=1; export num_iter=100; CUDA_VISIBLE_DEVICES=7 python main_mar.py --img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 400 --warmup_epochs 100 --batch_size 392 --blr 1.0e-4 --diffusion_batch_mul 4 --output_dir output_dir/rollingmar  --data_path ${IMAGENET_PATH} --online_eval  --num_images 10000  --num_iter ${num_iter} --evaluate --eval_bsz ${eval_bsz} --enable_timer > bs${eval_bsz}_iter${num_iter}.log




# New train

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50012 \
main_mar.py \
--img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model mar_base --diffloss_d 6 --diffloss_w 1024 \
--epochs 800 --warmup_epochs 100 --batch_size 512 --blr 1.0e-4 --diffusion_batch_mul 4 \
--output_dir output_dir/rollingmar_imagenet50 \
--data_path /data/imagenet_50_private --online_eval  --num_images 2500  --num_iter 100 --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 100


# Cache

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_cache.py \
--img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 \
--batch_size 128 \
--data_path /data/imagenet_50_private --cached_path /data/imagenet_50_private/cached_64



# RollingMAR eval


export AR=8; export DPAR=1; export seed=1; CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50212 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50_muchsmalllr  --eval_freq 100 --evaluate    --denoise_t_per_step ${DPAR} --num_iter ${AR} > output_dir/eval/rollingmar/rollingmar_ar${AR}_dpar${DPAR}_seed${seed}.log  --seed ${seed}


export AR=8; export DPAR=10; export seed=1; CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50221 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50_muchsmalllr  --eval_freq 100 --evaluate    --denoise_t_per_step ${DPAR} --num_iter ${AR} > output_dir/eval/rollingmar/rollingmar_ar${AR}_dpar${DPAR}_seed${seed}.log   --seed ${seed}


export AR=8; export DPAR=100; export seed=1; CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50213 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50_muchsmalllr  --eval_freq 100 --evaluate    --denoise_t_per_step ${DPAR} --num_iter ${AR} > output_dir/eval/rollingmar/rollingmar_ar${AR}_dpar${DPAR}_seed${seed}.log  --seed ${seed}




export AR=8; export DPAR=1; export seed=2; CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50212 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50_muchsmalllr  --eval_freq 100 --evaluate    --denoise_t_per_step ${DPAR} --num_iter ${AR} > output_dir/eval/rollingmar/rollingmar_ar${AR}_dpar${DPAR}_seed${seed}.log  --seed ${seed}


export AR=8; export DPAR=10; export seed=2; CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50221 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50_muchsmalllr  --eval_freq 100 --evaluate    --denoise_t_per_step ${DPAR} --num_iter ${AR} > output_dir/eval/rollingmar/rollingmar_ar${AR}_dpar${DPAR}_seed${seed}.log   --seed ${seed}


export AR=8; export DPAR=100; export seed=2; CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50213 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50_muchsmalllr  --eval_freq 100 --evaluate    --denoise_t_per_step ${DPAR} --num_iter ${AR} > output_dir/eval/rollingmar/rollingmar_ar${AR}_dpar${DPAR}_seed${seed}.log  --seed ${seed}



export AR=8; export DPAR=1; export seed=3; CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50212 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50_muchsmalllr  --eval_freq 100 --evaluate    --denoise_t_per_step ${DPAR} --num_iter ${AR} > output_dir/eval/rollingmar/rollingmar_ar${AR}_dpar${DPAR}_seed${seed}.log  --seed ${seed}


export AR=8; export DPAR=10; export seed=3; CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50221 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50_muchsmalllr  --eval_freq 100 --evaluate    --denoise_t_per_step ${DPAR} --num_iter ${AR} > output_dir/eval/rollingmar/rollingmar_ar${AR}_dpar${DPAR}_seed${seed}.log   --seed ${seed}


export AR=8; export DPAR=100; export seed=3; CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50213 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50_muchsmalllr  --eval_freq 100 --evaluate    --denoise_t_per_step ${DPAR} --num_iter ${AR} > output_dir/eval/rollingmar/rollingmar_ar${AR}_dpar${DPAR}_seed${seed}.log  --seed ${seed}







export AR=1; export DPAR=1; export seed=3; CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50212 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50  --eval_freq 100 --evaluate --num_iter ${AR} > output_dir/eval/mar/mar_ar${AR}_seed${seed}.log  --seed ${seed}

export AR=4; export DPAR=1; export seed=3; CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50212 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50  --eval_freq 100 --evaluate --num_iter ${AR} > output_dir/eval/mar/mar_ar${AR}_seed${seed}.log  --seed ${seed}


export AR=16; export DPAR=1; export seed=3; CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50212 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50  --eval_freq 100 --evaluate --num_iter ${AR} > output_dir/eval/mar/mar_ar${AR}_seed${seed}.log  --seed ${seed}


export AR=1; export DPAR=1; export seed=1; CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50212 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50  --eval_freq 100 --evaluate --num_iter ${AR} > output_dir/eval/mar/mar_ar${AR}_seed${seed}.log  --seed ${seed}

export AR=4; export DPAR=1; export seed=1; CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50212 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50  --eval_freq 100 --evaluate --num_iter ${AR} > output_dir/eval/mar/mar_ar${AR}_seed${seed}.log  --seed ${seed}


export AR=16; export DPAR=1; export seed=1; CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50212 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50  --eval_freq 100 --evaluate --num_iter ${AR} > output_dir/eval/mar/mar_ar${AR}_seed${seed}.log  --seed ${seed}





export AR=8; export DPAR=1; export seed=1; CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50312 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50  --eval_freq 100 --evaluate --num_iter ${AR} > output_dir/eval/mar/mar_ar${AR}_seed${seed}.log  --seed ${seed}

export AR=8; export DPAR=1; export seed=2; CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50312 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50  --eval_freq 100 --evaluate --num_iter ${AR} > output_dir/eval/mar/mar_ar${AR}_seed${seed}.log  --seed ${seed}


export AR=8; export DPAR=1; export seed=3; CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1  --master_port=50312 main_mar.py --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_base --diffloss_d 6 --diffloss_w 1024 --epochs 8000 --warmup_epochs 100 --batch_size 512 --blr 0.2e-4 --diffusion_batch_mul 4 --output_dir output_dir/eval/rollingmar --data_path /data/imagenet_50_private --online_eval  --num_images 5000   --cal_on_the_fly --class_num 50 --use_cached --cached_path /data/imagenet_50_private/cached_64 --save_last_freq 200 --eval_bsz 1024 --resume output_dir/rollingmar_imagenet50  --eval_freq 100 --evaluate --num_iter ${AR} > output_dir/eval/mar/mar_ar${AR}_seed${seed}.log  --seed ${seed}




