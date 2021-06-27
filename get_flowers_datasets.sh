URL=https://www.robots.ox.ac.uk/~vgg/data/flowers/17/trimaps.tgz
TAR_FILE=./datasets/flowers/trimaps.tgz
TARGET_DIR=./datasets/flowers/trimaps
mkdir -p $TARGET_DIR
wget -N $URL -O $TAR_FILE
tar -zxvf $TAR_FILE -C ./datasets/flowers/trimaps/
rm $TAR_FILE
rm ./datasets/flowers/trimaps/trimaps/imlist.mat


URL=https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz
TAR_FILE=./datasets/flowers/17flowers.tgz
TARGET_DIR=./datasets/flowers/jpg
mkdir -p $TARGET_DIR
wget -N $URL -O $TAR_FILE
tar -zxvf $TAR_FILE -C ./datasets/flowers/jpg/
rm $TAR_FILE
rm ./datasets/flowers/jpg/jpg/files.txt
