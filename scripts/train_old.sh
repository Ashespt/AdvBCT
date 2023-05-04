#landmark resnet50
CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore -u train.py \
--cfg configs/old_model.yml TAG 30class TRAIN.FILE_DIR ./data/GLDv2/gldv2_train_old_30percent_class.txt OUTPUT 'output' \
MODEL.NUM_CLASSES 24393 MODEL.ARCH 'resnet18' DIST_URL 'tcp://127.0.0.1:23456'

#reid
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore -u train.py \
#--cfg configs/old_model.yml TAG 100data TRAIN.ROOT ./data/Market-1501 TRAIN.FILE_DIR imlist_label.txt TRAIN.DATASET_TYPE 'reid' \
#TRAIN.DATASET 'market1501' OUTPUT 'output_reid' MODEL.NUM_CLASSES 751 TRAIN.EPOCHS 120 TRAIN.SAVE_FREQ 10