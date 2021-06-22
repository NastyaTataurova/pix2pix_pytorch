FILE=$1

echo "Specified [$FILE]"
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
mkdir $datasets
ZIP_FILE=./$FILE.zip
TARGET_DIR=./$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
rm -r /content/pix2pix_pytorch/datasets/maps/train/resized




