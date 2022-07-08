python3 ./test_pad_ad.py --arch QF --batch_size 1 --gpu '0' \
    --input_dir /home/amax/Real/input/ \
    --gt_dir /home/amax/Real/gt/ \
    --save_in /home/amax/Real/output/ \
    --result_dir /home/amax/Real/6/ \
    --weights /home/amax/model_best_G15.pth \
    --embed_dim 96 --val_ps 64 --noiseL 15
