export PYTHONPATH=/your/code/path
export CUDA_VISIBLE_DEVICES=0,1,2,3
OLD_MODEL_PATH=./output/final_model/resnet18_30class_oldmodel.pth.tar
OLD_MODEL_ARCH='resnet18'

#self-test, landmark roxford5k rparis6k gldv2
python test.py --cfg configs/test_model.yml DISTRIBUTED True \
EVAL.BCT_TYPE 'base_model' \
MODEL.ARCH $OLD_MODEL_ARCH MODEL.PRETRAINED_PATH $OLD_MODEL_PATH \
MODEL.NUM_CLASSES 24393 \
EVAL.SAVE_DIR './logs/eval/tmp_eval' \
EVAL.DATASET_TYPE $1 EVAL.DATASET_NAME $2 EVAL.DATASET $2 \
EVAL.ROOT $3