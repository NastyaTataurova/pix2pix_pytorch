URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/maps.zip
mkdir $datasets
ZIP_FILE=./maps.zip
TARGET_DIR=./maps/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
rm -r /content/pix2pix_pytorch/datasets/maps/train/resized




