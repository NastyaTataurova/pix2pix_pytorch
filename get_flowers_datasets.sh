URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz https://www.robots.ox.ac.uk/~vgg/data/flowers/17/trimaps.tgz
TAR_FILE=./datasets/trimaps.tgz
TARGET_DIR=./datasets/flowers/trimaps
mkdir -p $TARGET_DIR
wget -N $URL -O $TAR_FILE
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
/content/trimaps/imlist.mat

jpg /content/jpg/files.txt
