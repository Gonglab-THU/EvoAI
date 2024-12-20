<h1 align="center">EvoAI</h1>
<p align="center">EvoAI enables extreme compression and reconstruction of the protein sequence space</p>

## Prerequisites

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

5. We have calculated the features and prediction results of wild-type protein and some mutated proteins, which can be downloaded from the [Zenodo](https://doi.org/10.5281/zenodo.10686156) website.

* Place the downloaded and decompressed contents of the `wt_data.zip` in the `wt_data` folder.
* Place the downloaded and decompressed contents of the `mut_data.zip` in the `mut_data` folder.

## Usage

```bash
bash predict.sh -o ./mut_data/D32E,S56R,M112L,I123K,R124W,T181S
```

## Tips

This study involves model predictions carried out in two parts: one using **DMS information** and the other using **EvoScan anchors**. These correspond to the two boxplots on the left and right in **Figure 5C**, respectively. The DMS information part relies on the zero-shot predictions from the GeoFitness model.

As a result, there are two folders in the source code corresponding to the respective predictions.

The [folder](04_itertools_combinations_6mut_from_geofitness) contains the code and results related to predictions based on DMS information.
The [folder](05_itertools_combinations_6mut_from_82train) corresponds to the predictions using EvoScan anchors.

## Reference

[EvoAI enables extreme compression and reconstruction of the protein sequence space](https://doi.org/10.1038/s41592-024-02504-2)
