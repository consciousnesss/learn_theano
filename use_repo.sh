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

if [[ "$(uname -s)" == "Darwin" ]]; then
    brew tap homebrew/science
    brew install opencv
    cp /usr/local/lib/python2.7/site-packages/cv* $VENV/lib/python2.7/site-packages/
else
    robustus install opencv==2.4.8
fi
