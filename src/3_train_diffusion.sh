torchrun --nnode=1 --master_port=25678 train_diffusion.py --model DiT-XL/2 \
     --data-path /path/to/your/clustered_data --ckpt pretrained_models/DiT-XL-2-512x512.pt \
     --global-batch-size 4 --tag minimax --ckpt-every 1000 --log-every 500 --epochs 70 --image-size 512 \
     --condense --finetune-ipc -1 --results-dir ./path/to/results_dir --spec ost_cluster --nclass 7
