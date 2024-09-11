#!/bin/bash
set -e
PWD=$(dirname "$0")
cd $PWD

if  [[ ! -d ~/.android ]]; then
    mkdir ~/.android/
fi
echo -e $1 > ~/.android/adbkey
