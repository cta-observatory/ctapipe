
#!/bin/bash

set -ev

# I assume if this folder exists the environment is already installed
if [ ! -d $HOME/miniconda/envs/cta-dev ]; then
    conda create --name cta-dev python=$PYTHON_VERSION
else
    echo 'cta-dev env already created'
fi

exit 0;

