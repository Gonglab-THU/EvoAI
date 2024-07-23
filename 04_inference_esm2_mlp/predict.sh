#!/usr/bin/env bash

set -Eeuo pipefail

#######################################################################
# conda environment
#######################################################################

__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

conda activate evoai

#######################################################################
# run
#######################################################################

for name in $(ls ../mut_data); do
    echo ${name}
    python predict.py --ensemble_feature ../mut_data/${name}/ensemble.pt --saved_folder ../mut_data/${name}
done

rm -rf molecules
rm -f rotabase.txt
