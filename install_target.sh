#!/bin/bash

set -ev

if [ ${USE_TARGET} = "true" ]; then
    echo "Installing Target Software"
    conda install -c conda-forge cfitsio
    conda install swig
    mkdir -p $HOME/Software
    cd $HOME/Software
    git clone https://github.com/watsonjj/target_software.git
    cd $TARGET_SOFTWARE
    export TRAVIS_PYTHON_VERSION=$PYTHON_VERSION  # install.sh needs $TRAVIS_PYTHON_VERSION
    ./install.sh
    cd $TRAVIS_BUILD_DIR
fi

exit 0;
