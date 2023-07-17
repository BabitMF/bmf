#!/bin/bash

#`set -o pipefail` should NOT be added here.
set -exu

if [ $# -eq 0 ]
then
    echo "bmf root directory is needed!"
    exit 1
fi

bmf_root=$1

function change_bin_deps() {
    otool -L $1 | awk '{if(NF>1){print($1)}}' | grep "libengine\|libbmf_module_sdk\|libhmp" | while read old_dep
    do
        new_dep="@executable_path/../lib/"$(basename ${old_dep})
        install_name_tool -change ${old_dep} ${new_dep} $1
    done
}

function change_lib_deps() {
    otool -L $1 | awk '{if(NF>1){print($1)}}' | grep "libengine\|libbmf_module_sdk\|libhmp" | while read old_dep
    do
        new_dep="@loader_path/"$(basename ${old_dep})
        install_name_tool -change ${old_dep} ${new_dep} $1
    done

    otool -L $1 | awk '{if(NF>1){print($1)}}' | grep "ffmpeg" | while read old_dep
    do
        new_dep="@rpath/"$(basename ${old_dep})
        install_name_tool -change ${old_dep} ${new_dep} $1
    done

    otool -L $1 | awk '{if(NF>1){print($1)}}' | grep "Python" | while read old_dep
    do
        new_dep="@executable_path/../../../../Python"
        install_name_tool -change ${old_dep} ${new_dep} $1
    done
}

cd ${bmf_root}/bmf
for bin in `find bin -maxdepth 1 -not -type d`
do
    change_bin_deps ${bin}
done
for lib in `find lib -maxdepth 1 -not -type d`
do
    change_lib_deps ${lib}
done
