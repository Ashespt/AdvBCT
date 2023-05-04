export PYTHONPATH=/your/code/path
export CUDA_VISIBLE_DEVICES=0,1,2,3
NEW_MODEL_PATH=./output/final_model/resnet18_100class_comp_advbct.pth.tar
OLD_MODEL_PATH=./output/final_model/resnet18_30class_oldmodel.pth.tar
NEW_MODEL_ARCH='resnet18'
OLD_MODEL_ARCH='resnet18'
#self-test, landmark roxford5k rparis6k gldv2
python test.py --cfg configs/test_model.yml DISTRIBUTED True \
EVAL.BCT_TYPE 'base_model' \
MODEL.ARCH $NEW_MODEL_ARCH MODEL.PRETRAINED_PATH $NEW_MODEL_PATH \
EVAL.SAVE_DIR './logs/eval/tmp_eval' \
EVAL.DATASET_TYPE $1 EVAL.DATASET_NAME $2 EVAL.DATASET $2 \
EVAL.ROOT $3

#cross-test
python test.py --cfg configs/test_model.yml DISTRIBUTED True \
EVAL.BCT_TYPE 'bct' \
OLD_MODEL.ARCH $OLD_MODEL_ARCH OLD_MODEL.PRETRAINED False OLD_MODEL.PRETRAINED_PATH $OLD_MODEL_PATH \
NEW_MODEL.ARCH $NEW_MODEL_ARCH NEW_MODEL.PRETRAINED False NEW_MODEL.PRETRAINED_PATH $NEW_MODEL_PATH \
EVAL.SAVE_DIR './logs/eval/tmp_eval'  \
EVAL.SAVE_DIR './logs/eval/tmp_eval' \
EVAL.DATASET_TYPE $1 EVAL.DATASET_NAME $2 EVAL.DATASET $2 \
EVAL.ROOT $3 EVAL.HOT_REFRESH False