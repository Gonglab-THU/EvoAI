#!/usr/bin/env bash

set -Eeuo pipefail

#######################################################################
# conda environment
#######################################################################

__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

conda activate geostab

#######################################################################
# run
#######################################################################

for name in $(ls ../mut_data); do
    bash ../predict.sh -o ../mut_data/${name}
done

rm -rf molecules
rm -f rotabase.txt
