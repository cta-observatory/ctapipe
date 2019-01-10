#!/bin/bash

set -ev

# I assume if this exists miniconda is already installed
if [ ! -f $HOME/miniconda/bin/python ]; then
    if [ ${TRAVIS_OS_NAME} = "linux" ]; then
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
    fi
    if [ ${TRAVIS_OS_NAME} = "osx" ]; then
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
    fi
    bash miniconda.sh -b -f -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda  # get latest conda version
else
    echo 'miniconda is already installed.'
fi

exit 0;
