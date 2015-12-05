#!/usr/bin/env bash
# This script should be sourced before using this repo (for development).
# It creates the python virtualenv and using pip to populate it
# This only run to setup the development environment.
# Installation is handled by setup.py/disttools.

# Robust way of locating script folder
# from http://stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
SOURCE=${BASH_SOURCE:-$0}

DIR="$( dirname "$SOURCE" )"
while [ -h "$SOURCE" ]
do 
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
  DIR="$( cd -P "$( dirname "$SOURCE"  )" && pwd )"
done
WDIR="$( pwd )"
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

# create virtualenv
VENV=$DIR/venv
if [ -d VENV ]; then
   # Virtual Env exists
   echo vitual environment $VENV exist
else
    virtualenv $VENV --prompt "(learn_theano)"
fi

# add pylearn data path to activate
grep -q -F 'export PYLEARN2_DATA_PATH=$HOME/.pylearn_data' $VENV/bin/activate || echo 'export PYLEARN2_DATA_PATH=$HOME/.pylearn_data' >> $VENV/bin/activate

if [[ "$(uname -s)" == "Darwin" ]]; then
    grep -q -F 'export PYLEARN2_VIEWER_COMMAND="open -Wn"' $VENV/bin/activate || echo 'export PYLEARN2_VIEWER_COMMAND="open -Wn"' >> $VENV/bin/activate
fi

source $VENV/bin/activate


# install robustus into virtualenv
pip install -U git+http://github.com/braincorp/robustus.git

# create folder for packages compilation cache
mkdir -p ~/.robustus_rc/wheelhouse

# initialize robustus venv with cache path 
robustus --cache ~/.robustus_rc/wheelhouse env $VENV

# install this folder in developer mode
echo "Running robustus with options '$ROBUSTUS_OPTIONS'"
robustus install -e . $ROBUSTUS_OPTIONS

# install bleeding edge theano at Dec, 1, 2015
pip install git+git://github.com/Theano/Theano.git@30cc6380863b08a3a90ecbe083ddfb629a56161d
pip install git+git://github.com/fchollet/keras.git@5956dbe8fad1642f5c6529008fd9126d920b0e76

# install blocks and fuel
pip install picklable-itertools==0.1.1
pip install progressbar2==2.7.3
pip install pyyaml==3.11
pip install six==1.9.0
pip install toolz==0.7.2

pip install git+git://github.com/mila-udem/fuel.git
pip install git+git://github.com/mila-udem/blocks.git

# add path for fuel data
mkdir -p ~/.fuel_data
grep -q -F 'export FUEL_DATA_PATH=$HOME/.fuel_data' $VENV/bin/activate || echo 'export FUEL_DATA_PATH=$HOME/.fuel_data' >> $VENV/bin/activate
cd ~/.fuel_data
fuel-download mnist
fuel-convert mnist
cd -

if [[ "$(uname -s)" == "Darwin" ]]; then
    brew tap homebrew/science
    brew install opencv
    cp /usr/local/lib/python2.7/site-packages/cv* $VENV/lib/python2.7/site-packages/
else
    robustus install opencv==2.4.8
fi

# install pylearn2
cd $VENV/lib/python2.7/site-packages/
git clone git://github.com/lisa-lab/pylearn2.git
cd pylearn2
python setup.py develop
# download pylearn MNIST
python pylearn2/scripts/datasets/download_mnist.py
cd $DIR



