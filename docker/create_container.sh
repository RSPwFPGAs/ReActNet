DOCKERIMAGE=ubuntu:18.04
SHAREDDIR=$PWD/shared
sudo docker rm reactnet_train
sudo docker run -it --privileged -v $SHAREDDIR:/shared -h reactnet_train --name reactnet_train $DOCKERIMAGE /bin/bash

