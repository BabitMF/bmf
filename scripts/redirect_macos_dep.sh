#!/bin/bash

#`set -o pipefail` should NOT be added here.
set -exu

if [ $# -eq 0 ]
then
    echo "bmf root directory is needed!"
    exit 1
fi

bmf_root=$1

otool_bin=/Applications/Xcode_15.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/otool
install_name_tool_bin=/Applications/Xcode_15.2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/install_name_tool

function change_bin_deps() {
    $otool_bin -L $1 | awk '{if(NF>1){print($1)}}' | grep "libengine\|libbmf_module_sdk\|libhmp" | while read old_dep
    do
        new_dep="@executable_path/../lib/"$(basename ${old_dep})
        $install_name_tool_bin -change ${old_dep} ${new_dep} $1
        echo "Updating $old_dep -> $new_dep"
    done
}

function change_lib_deps() {
    $otool_bin -L $1 | awk '{if(NF>1){print($1)}}' | grep "libengine\|libbmf_module_sdk\|libhmp" | while read old_dep
    do
        new_dep="@loader_path/"$(basename ${old_dep})
        $install_name_tool_bin -change ${old_dep} ${new_dep} $1
        echo "Updating $old_dep -> $new_dep"
    done

    $otool_bin -L $1 | awk '{if(NF>1){print($1)}}' | grep "ffmpeg" | while read old_dep
    do
        new_dep="@rpath/"$(basename ${old_dep})
        $install_name_tool_bin -change ${old_dep} ${new_dep} $1
        echo "Updating $old_dep -> $new_dep"
    done

    $otool_bin -L $1 | awk '{if(NF>1){print($1)}}' | grep "Python" | while read old_dep
    do
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d '.' -f 1,2)
        #new_dep="@executable_path/../lib/libpython${PYTHON_VERSION}.dylib"
        new_dep="@rpath/Python"
        $install_name_tool_bin -change ${old_dep} ${new_dep} $1
        $install_name_tool_bin -add_rpath "@executable_path/../../../../" $1
        $install_name_tool_bin -add_rpath "@executable_path" $1
        echo "Updating $old_dep -> $new_dep"
    done
}

cd ${bmf_root}/bmf
for bin in `find bin -maxdepth 1 -not -type d`
do
    echo "Processing: $bin"
    change_bin_deps ${bin}
done
for lib in `find lib -maxdepth 1 -not -type d`
do
    echo "Processing: $lib"
    change_lib_deps ${lib}
done