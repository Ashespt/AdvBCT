## Datasets
### Landmark
#### ROxford and RParis
* download revisted Oxford & Paris (http://cmp.felk.cvut.cz/revisitop/) into ./data/

   （don’t need download R-1M distractor dataset. gnd_roxford5k.pkl and gnd_rparis6k.pkl  are required)
* ./data/ROxfordParis structure
```plaintext
.
└── ROxfordParis
    ├── gnd_roxford5k.pkl
    ├── gnd_rparis6k.pkl
    ├── roxford5k
    │   └── jpg
    └── rparis6k
        └── jpg
```

#### GLDv2
* download GLDv2 train, index and test (https://github.com/cvdfoundation/google-landmark)
* ./data/GLDv2 structure
```plaintext
.
└── GLDv2
    ├── test
       ├── 0
       ├── 1
       ...
    ├── index
       ├── 0
       ├── 1
       ...
    └── train
       ├── 0
       ├── 1
       ...
```

### Generate train labels
* download the total labels of GLDv2 and put it into ./data/GLDv2.
* run utils/gen_dataset.sh and you can get followings under ./data/GLDv2:
```plaintext
./
├── gldv2_gallery_list.txt
├── gldv2_private_query_gt.txt
├── gldv2_private_query_list.txt
├── gldv2_public_query_gt.txt
├── gldv2_public_query_list.txt
├── gldv2_train_new_100percent_class.txt
├── gldv2_train_old_30percent_class.txt
├── gldv2_train_old_30percent_data.txt
├── label_81313.txt
```