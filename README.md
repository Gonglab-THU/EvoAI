<h1 align="center">EvoAI</h1>
<p align="center">EvoAI enables extreme compression and reconstruction of the protein sequence space</p>

## Prerequisites

* [AlphaFold](https://github.com/deepmind/alphafold)
* [FoldX](https://foldxsuite.crg.eu)
* [Anaconda](https://www.anaconda.com)

## Install software on Linux

1. download `EvoAI`

```bash
git clone https://github.com/Gonglab-THU/EvoAI.git
cd EvoAI
```

2. install `FoldX 5` software and put `FoldX` into `software` folders

3. install `Anaconda` software

4. install Python packages from `Anaconda`

```bash
conda create -n evoai python=3.10
conda activate evoai

conda install pytorch cpuonly -c pytorch
pip install biopython
pip install click
```

## Usage

```bash
bash predict.sh -o ./mut_data/D32E,S56R,M112L,I123K,R124W,T181S
```
