URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz
TAR_FILE=./datasets/edges2shoes.tar.gz
TARGET_DIR=./datasets/edges2shoes/
mkdir -p $TARGET_DIR
wget -N $URL -O $TAR_FILE
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
rm ./datasets/edges2shoes/train/{2003..49001}_AB.jpg
