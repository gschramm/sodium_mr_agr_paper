# Source code for dual echo anatomically guided MR reconstruction and joint decay estimation

This repository containd the source code of our paper 
"Resolution enhancement, noise suppression, and joint T2* decay estimation in dual-echo sodium-23 MR imaging using anatomically guided reconstruction"
published in Magnetic Resonance in Medicine in Dec 2023 [link](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29936).
The accepted version of the paper is also freely available on arxiv [link](https://arxiv.org/abs/2311.03116).

Re-use of our code and results are very welcome, using a refernce to our paper

*Schramm G, Filipovic M, Qian Y, et al. Resolution enhancement, noise suppression, and joint T2\* decay estimation in dual-echo sodium-23 MR imaging using anatomically guided reconstruction. Magn Reson Med. 2023; 1-15. doi: 10.1002/mrm.29936*


```bibtex
@article{https://doi.org/10.1002/mrm.29936,
author = {Schramm, Georg and Filipovic, Marina and Qian, Yongxian and Alivar, Alaleh and Lui, Yvonne W. and Nuyts, Johan and Boada, Fernando},
title = {Resolution enhancement, noise suppression, and joint T2* decay estimation in dual-echo sodium-23 MR imaging using anatomically guided reconstruction},
journal = {Magnetic Resonance in Medicine},
keywords = {anatomical priors, brain, iterative reconstruction, quantification, sodium MR},
doi = {https://doi.org/10.1002/mrm.29936},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.29936},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/mrm.29936},
}
```


## Setup

(1) Clone this repository
```
git clone https://github.com/gschramm/sodium_mr_agr_paper.git
```

(2) Create a virtual conda environment containg all packages we need.
```
conda env create -f environment.yml
```

(3) Download brainweb subject 54 availalbe [here](https://brainweb.bic.mni.mcgill.ca/anatomic_normal_20.html)
    convert all images to .nii.gz and resample the T1w to the grid of the crisp_v image.
    Save the files to `data / brainweb54 / 'subject54_t1w_p4_resampled.nii.gz'`
    and `data / brainweb54 / 'subject54_crisp_v.nii.gz'`

## Running reconstructions

Use the script `00_reconstruction_brainweb.py` to run a single noise realization of the brainweb sodium MR
reconstructions. Use `01_analyze_brainweb.py` to analyze the results (of multiple realizations).

`02_reconstruction_real_data.py` can be used to reconstruct real TPI data. See the header of the file
for the required input files.
