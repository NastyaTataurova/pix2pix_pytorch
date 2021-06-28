cd pix2pix_pytorch

URL=https://www.robots.ox.ac.uk/~vgg/data/flowers/17/trimaps.tgz
TAR_FILE=./datasets/flowers/train/trimaps.tgz
TARGET_DIR=./datasets/flowers/train/trimaps
TEST_DIR=./datasets/flowers/test/trimaps/trimaps
mkdir -p $TARGET_DIR
mkdir -p $TEST_DIR
wget -N $URL -O $TAR_FILE
tar -zxvf $TAR_FILE -C ./datasets/flowers/train/trimaps/
rm $TAR_FILE
rm ./datasets/flowers/train/trimaps/trimaps/imlist.mat
mv /content/pix2pix_pytorch/datasets/flowers/train/trimaps/trimaps/image_0{002,028,097,250,285}.png /content/pix2pix_pytorch/datasets/flowers/test/trimaps/trimaps/

URL=https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz
TAR_FILE=./datasets/flowers/train/17flowers.tgz
TARGET_DIR=./datasets/flowers/train/jpg
TEST_DIR=./datasets/flowers/test/jpg/jpg
mkdir -p $TARGET_DIR
mkdir -p $TEST_DIR
wget -N $URL -O $TAR_FILE
tar -zxvf $TAR_FILE -C ./datasets/flowers/train/jpg/
rm $TAR_FILE
rm ./datasets/flowers/train/jpg/jpg/files.txt
mv /content/pix2pix_pytorch/datasets/flowers/train/jpg/jpg/image_0{002,028,097,250,285}.jpg /content/pix2pix_pytorch/datasets/flowers/test/jpg/jpg/
