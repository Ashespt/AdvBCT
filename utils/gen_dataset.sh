echo 'split dataset of gldv2 by data'
python -m utils.process_dataset --root_path 'data/GLDv2' --split_by_data --split_ratio 0.3
echo 'split dataset of gldv2 by class'
python -m utils.process_dataset --root_path 'data/GLDv2' --split_by_class --split_ratio 0.3
#echo 'split dataset of ms1m by data'
#python-m utils.process_dataset --root_path 'data/ms1m' --split_by_data --split_ratio 0.3 --dataset ms1m
#echo 'split dataset of ms1m by class'
#python -m utils.process_dataset --root_path 'data/ms1m' --split_by_class --split_ratio 0.3 --dataset ms1m
#echo 'split dataset of reid by data'
#python -m utils.process_dataset --root_path 'data/Market-1501' --split_by_data --split_ratio 0.3 --dataset market1501
#echo 'split dataset of reid by class'
#python -m utils.process_dataset --root_path 'data/Market-1501' --split_by_class --split_ratio 0.3 --dataset market1501
