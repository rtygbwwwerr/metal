#!/usr/bin/env bash

# Install miniconda to avoid compiling scipy
if test -e $HOME/miniconda/bin ; then
    echo "Package miniconda already installed."
else
    echo "Installing miniconda."
    rm -rf $HOME/miniconda
    mkdir -p $HOME/download
    if [[ -d $HOME/download/miniconda.sh ]] ; then rm -rf $HOME/download/miniconda.sh ; fi
    wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/download/miniconda.sh
    chmod +x $HOME/download/miniconda.sh
    $HOME/download/miniconda.sh -b -p $HOME/miniconda
fi
