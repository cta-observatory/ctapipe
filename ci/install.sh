#!/bin/bash

echo "Using conda: $CONDA"
echo "Using python $TRAVIS_PYTHON_VERSION | $PYTHON_VERSION"


if [[ "$CONDA" == "true" ]]; then
	# Install miniconda following instructions at
	# http://conda.pydata.org/docs/travis.html

	if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
	  wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
	elif [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
	  wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
	fi

	bash miniconda.sh -b -p $HOME/miniconda
	source $HOME/miniconda/etc/profile.d/conda.sh

	hash -r
	conda config --set always_yes yes --set changeps1 no
	conda update -q conda  # get latest conda version
	# Useful for debugging any issues with conda
	conda info -a

	conda create --name cta-dev python=$PYTHON_VERSION
	travis_wait 20 conda env update -n cta-dev --file py${PYTHON_VERSION}_env.yaml
	conda activate cta-dev
else
	pip install -e .[all]
fi
