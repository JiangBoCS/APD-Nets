python3 ./test_pad_ad.py --arch QF --batch_size 1 --gpu '0' \
    --input_dir /home/amax/Jiangbo/results/Real/input/ \
    --gt_dir /home/amax/Jiangbo/results/Real/gt/ \
    --save_in /home/amax/Jiangbo/results/Real/output/ \
    --result_dir /home/amax/Jiangbo/results/Real/6/ \
    --weights /home/amax/Jiangbo/log/QFdim96_G15/models/model_best_G15.pth \
    --embed_dim 96 --val_ps 64 --noiseL 20






## CBSD68
##python3 ./test_pad.py --arch QF --batch_size 1 --gpu '0' \
##    --input_dir /home/amax/DN_Dataset/patch/BSD68_cut_4pathch/noisy15/ \
##    --gt_dir /home/amax/DN_Dataset/patch/BSD68_cut_4pathch/GT/ \
##    --result_dir /home/amax/Jiangbo/log/QFdim44_G15/results/CBSD68/ \
##    --weights /home/amax/Jiangbo/log/QFdim44_G15/models/model_best.pth --embed_dim 44 --val_ps 256
##
#### Set12    
##python3 ./test_pad.py --arch QF --batch_size 1 --gpu '0' \
##    --input_dir /home/amax/DN_Dataset/patch/Set12_cut_4patch/noise15/ \
##    --gt_dir /home/amax/DN_Dataset/patch/Set12_cut_4patch/GT/ \
##    --result_dir /home/amax/Jiangbo/log/QFdim44_G15/results/Set12/ \
##    --weights /home/amax/Jiangbo/log/QFdim44_G15/models/model_best.pth --embed_dim 44 --val_ps 256
###  
#### KOA24  
##python3 ./test_pad.py --arch QF --batch_size 1 --gpu '0' \
##    --input_dir /home/amax/DN_Dataset/patch/KODAK_cut_4patch/noise15/ \
##    --gt_dir /home/amax/DN_Dataset/patch/KODAK_cut_4patch/GT/ \
##    --result_dir /home/amax/Jiangbo/log/QFdim44_G15/results/KOA24/ \
##    --weights /home/amax/Jiangbo/log/QFdim44_G15/models/model_best.pth --embed_dim 44 --val_ps 256
##
#### McM18   
##python3 ./test_pad.py --arch QF --batch_size 1 --gpu '0' \
##    --input_dir /home/amax/DN_Dataset/patch/McM_cut_4pathch/noise15/ \
##    --gt_dir /home/amax/DN_Dataset/patch/McM_cut_4pathch/GT/ \
##    --result_dir /home/amax/Jiangbo/log/QFdim44_G15/results/McM18/ \
##    --weights /home/amax/Jiangbo/log/QFdim44_G15/models/model_best.pth --embed_dim 44 --val_ps 256
#    
## CBSD68
#python3 ./test_pad.py --arch QF --batch_size 1 --gpu '0' \
#    --input_dir /home/amax/DN_Dataset/patch/BSD68_cut_4pathch/noisy25/ \
#    --gt_dir /home/amax/DN_Dataset/patch/BSD68_cut_4pathch/GT/ \
#    --result_dir /home/amax/Jiangbo/log/QFdim44_G25/results/CBSD68/ \
#    --weights /home/amax/Jiangbo/log/QFdim44_G25/models/model_best.pth --embed_dim 44 --val_ps 256
#
### Set12    
#python3 ./test_pad.py --arch QF --batch_size 1 --gpu '0' \
#    --input_dir /home/amax/DN_Dataset/patch/Set12_cut_4patch/noise25/ \
#    --gt_dir /home/amax/DN_Dataset/patch/Set12_cut_4patch/GT/ \
#    --result_dir /home/amax/Jiangbo/log/QFdim44_G25/results/Set12/ \
#    --weights /home/amax/Jiangbo/log/QFdim44_G25/models/model_best.pth \
#    --embed_dim 44 --val_ps 256
##  
### KOA24  
#python3 ./test_pad.py --arch QF --batch_size 1 --gpu '0' \
#    --input_dir /home/amax/DN_Dataset/patch/KODAK_cut_4patch/noise25/ \
#    --gt_dir /home/amax/DN_Dataset/patch/KODAK_cut_4patch/GT/ \
#    --result_dir /home/amax/Jiangbo/log/QFdim44_G25/results/KOA24/ \
#    --weights /home/amax/Jiangbo/log/QFdim44_G25/models/model_best.pth --embed_dim 44 --val_ps 256
#
### McM18   
#python3 ./test_pad.py --arch QF --batch_size 1 --gpu '0' \
#    --input_dir /home/amax/DN_Dataset/patch/McM_cut_4pathch/noise25/ \
#    --gt_dir /home/amax/DN_Dataset/patch/McM_cut_4pathch/GT/ \
#    --result_dir /home/amax/Jiangbo/log/QFdim44_G25/results/McM18/ \
#    --weights /home/amax/Jiangbo/log/QFdim44_G25/models/model_best.pth --embed_dim 44 --val_ps 256

