#!/bin/bash


if [[ "$INSTALL_METHOD" == "conda" ]]; then
  echo "Using conda located at "
  echo $CONDA
  which conda
  sudo chown -R $USER $CONDA # Give CONDA permission to its own files
  source $CONDA/etc/profile.d/conda.sh
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda  # get latest conda version
  # Useful for debugging any issues with conda
  conda info -a
  conda install -c conda-forge mamba

  sed -i -e "s/- python=.*/- python=$PYTHON_VERSION/g" environment.yml
  mamba env create -n ci --file environment.yml
  conda activate ci
  echo 'source $CONDA/etc/profile.d/conda.sh' >> ~/.bash_profile
  echo 'conda activate ci' >> ~/.bash_profile
else
  echo "Using conda located at "
  echo $CONDA
  which conda
  conda install python=$PYTHON_VERSION pip
  echo "Using pip"
  pip install -U pip setuptools wheel
fi
