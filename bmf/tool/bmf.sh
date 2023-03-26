#!/bin/bash

# script for running BMF by command line
#
# To run a bmf function:
#   ./bmf.sh --decoder ${decoder_param} --filter "${filter_param}" --module "${module_param}" --encoder ${encoder_param}
#
# For example:
#   ./bmf.sh --decoder -i input.mp4 -v [v1] --filter "[v1] -name scale -para 720:640 [v2]" --encoder -map [v2] -o out.mp4
#
# For module details : https://

# run by cmd
if [ $# ] >0; then
  python3 bmf_cmd.py $@
else
  echo "no arguments"
fi

