# Source code for dual echo anatomically guided MR reconstruction and joint decay estimation

This repository will contain the source code for our paper on
dual echo anatomically guided MR reconstruction and joint decay estimation upon formal
acceptance of our manuscript submitted to MRM


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
