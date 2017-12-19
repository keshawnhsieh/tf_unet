#!/bin/bash

python gen_samples.py   \
    --num_train=2000    \
    --rs_file=../sample_library/luhe_train.tif  \
    --lb_file=../sample_library/luhe_buildings_classes_train.tif
python gen_samples.py   \
    --num_valid=500     \
    --rs_file=../sample_library/luhe_valid.tif  \
    --lb_file=../sample_library/luhe_buildings_classes_valid.tif
python gen_samples.py   \
    --num_test=500      \
    --rs_file=../sample_library/luhe_test.tif   \
    --lb_file=../sample_library/luhe_buildings_classes_test.tif