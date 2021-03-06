python3 ./train_ag.py --arch APDNet --batch_size 16 --gpu '0,1,2,3' --nepoch 1000 \
      --train_ps 256 \
      --train_gt_dir /home/amax/DN_Dataset/BSD_DIV2K/BSDS500_DIV2K/train/GT/  \
      --train_input_dir /home/amax/DN_Dataset/BSD_DIV2K/BSDS500_DIV2K/train/input/ \
      --val_gt_dir /home/amax/DN_Dataset/BSD68/val_gt/ \
      --val_input_dir /home/amax/DN_Dataset/BSD68/val_input/ \
      --embed_dim 64 --warmup --checkpoint 500 \
      --env APDNetdim64_G15_1 --noiseL 15 --lr_initial 0.0001
