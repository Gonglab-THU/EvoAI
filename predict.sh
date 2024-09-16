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
# software
#######################################################################

export GEOSTAB_DIR=$(dirname $(realpath $0))
software_foldx=${GEOSTAB_DIR}/software/foldx/foldx
wt_folder=${GEOSTAB_DIR}/wt_data

#######################################################################
# pasre parameters
#######################################################################

help() {
    echo -e "Usage:\n"
    echo -e "bash run.sh [-o FOLDER]\n"
    echo -e "Description:\n"
    echo -e " \e[1;31m-o\e[0m output folder (e.g. -o ./mut_data/A57S)"
    echo -e "\e[1;31mAll parameters must be set!\e[0m"
    exit 1
}

# check the number of parameters
if [ $# -ne 2 ]; then
    echo -e "\e[1;31mThe number of parameters is wrong!\e[0m"
    help
fi

# check the validity of parameters
while getopts 'o:' PARAMETER
do
    case ${PARAMETER} in
        o)
        mut_folder=$(realpath -e ${OPTARG});;
        ?)
        help;;
    esac
done

shift "$(($OPTIND - 1))"

#######################################################################
# run
#######################################################################

echo -e "Output mut folder: \e[1;31m${mut_folder}\e[0m"
mkdir -p ${mut_folder}/foldx_tmp

# generate mut features
if [ ! -s ${mut_folder}/fixed_embedding.pt ]; then
    echo -e "run \e[1;31mmut fixed embedding\e[0m"
    python ${GEOSTAB_DIR}/generate_features/fixed_embedding.py --fasta_file ${mut_folder}/result.fasta --saved_folder ${mut_folder} &
fi

if [ ! -s ${mut_folder}/esm2.pt ]; then
    echo -e "run \e[1;31mmut esm2 embedding\e[0m"
    python ${GEOSTAB_DIR}/generate_features/esm2_embedding.py --fasta_file ${mut_folder}/result.fasta --saved_folder ${mut_folder} &
fi

wait

if [ ! -s ${mut_folder}/relaxed_repair.pdb ]; then
    echo -e "run \e[1;31mmut FoldX\e[0m"
    ${software_foldx} --command=BuildModel --pdb=relaxed_repair.pdb --pdb-dir=${wt_folder} --mutant-file=${mut_folder}/individual_list.txt --numberOfRuns=3 --output-dir=${mut_folder}/foldx_tmp
    cp ${mut_folder}/foldx_tmp/relaxed_repair_1_2.pdb ${mut_folder}/relaxed_repair.pdb
fi

if [ ! -s ${mut_folder}/coordinate.pt ]; then
    echo -e "run \e[1;31mmut coordinate\e[0m"
    python ${GEOSTAB_DIR}/generate_features/coordinate.py --pdb_file ${mut_folder}/relaxed_repair.pdb --saved_folder ${mut_folder}
fi

if [ ! -s ${mut_folder}/pair.pt ]; then
    echo -e "run \e[1;31mmut pair\e[0m"
    python ${GEOSTAB_DIR}/generate_features/pair.py --coordinate_file ${mut_folder}/coordinate.pt --saved_folder ${mut_folder}
fi

if [ ! -s ${mut_folder}/ensemble.pt ]; then
    python ${GEOSTAB_DIR}/generate_features/ensemble_ddGdTm.py --af2_pickle_file ${wt_folder}/result.pkl --wt_folder ${wt_folder} --mut_folder ${mut_folder}
fi

# predict
python ${GEOSTAB_DIR}/02_final_model/predict.py --ensemble_feature ${mut_folder}/ensemble.pt --saved_folder ${mut_folder}
