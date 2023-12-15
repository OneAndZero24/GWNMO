#!/usr/bin/env bash
curl -o train.txt https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/train.txt 
curl -o val.txt https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/val.txt
curl -o test.txt https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/test.txt

# OMNIGLOT_PATH
DATADIR="${OMNIGLOT_PATH}/images"
mkdir -p $DATADIR
curl -o images_background.zip https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip?raw=true -L
curl -o images_evaluation.zip https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip?raw=true -L
unzip images_background.zip -d $DATADIR
unzip images_evaluation.zip -d $DATADIR
mv $DATADIR/images_background/* $DATADIR/
mv $DATADIR/images_evaluation/* $DATADIR/
rmdir $DATADIR/images_background
rmdir $DATADIR/images_evaluation

python "${OMNIGLOT_PATH}/rot_omniglot.py"
python "${OMNIGLOT_PATH}/write_omniglot_filelist.py"
python "${OMNIGLOT_PATH}/write_cross_char_base_filelist.py"
