URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz
TAR_FILE=./datasets/edges2shoes.tar.gz
TARGET_DIR=./datasets/edges2shoes/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
