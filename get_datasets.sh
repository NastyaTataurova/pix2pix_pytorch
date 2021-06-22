FILE=$1
if [[$FILE != "edges2shoes"]]; then
  URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
  TAR_FILE=./datasets/$FILE.tar.gz
  TARGET_DIR=./datasets/$FILE/
  wget -N $URL -O $TAR_FILE
  mkdir -p $TARGET_DIR
  tar -zxvf $TAR_FILE -C ./datasets/
  rm $TAR_FILE
fi

if [[$FILE != "maps"]]; then
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
fi



