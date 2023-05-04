#landmark
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py --cfg configs/new_adv.yml DIST_URL 'tcp://127.0.0.1:12356'
#reid
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train.py --cfg configs/new_model.yml \
#MODEL.ARCH 'resnet18_resnet18' \
#OLD_MODEL.MODEL_PATH outpu/resnet50/100data/ckpt/checkpoint_epoch120.pth.tar OLD_MODEL.ARCH 'resnet50' OLD_MODEL.NUM_CLASSES 751 \
#NEW_MODEL.ARCH 'resnet101' NEW_MODEL.NUM_CLASSES 751 \
#TAG '100data_comp' TRAIN.ROOT ./data/Market-1501 TRAIN.FILE_DIR imlist_label.txt TRAIN.DATASET_TYPE 'reid' \
#TRAIN.DATASET 'market1501' OUTPUT 'output_reid' MODEL.NUM_CLASSES 751 TRAIN.EPOCHS 120 TRAIN.SAVE_FREQ 10
