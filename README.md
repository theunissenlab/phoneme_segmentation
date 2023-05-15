## phoneme_segmentation

## Summary

This repository contains code used to perform the analysis for the paper:

> **[1]** Gong, X., Huth, A. G., Deniz, F., Johnson, K., Gallant, J. L., & Theunissen, F. E..
> Phonemic segmentation of narrative speech in human cerebral cortex.
> Nature Communications, (2023). https://doi.org/

Dataset used in the paper can be found [here](https://gin.g-node.org/gallantlab/story_listening)

If you publish any work using the code in this repository, please cite the original publication [1], and cite this repository in the following recomended format:

**[1]** Gong, X., Huth, A. G., Gallant, J. L., & Theunissen, F. E.. Phoneme segmentation analysis. DOI: https://zenodo.org/badge/latestdoi/547471345

## How to get started
```bash
# clone the repository
git clone https://github.com/theunissenlab/phoneme_segmentation.git
cd phoneme_segmentation
python setup.py install

```

## Content
This repository contains a general pipeline for analysis of the publication [1]:
1. [Feature generation](phoneme_segmentation/features/io.py)
2. [Voxelwise encoding modeling and variance partitioning](phoneme_segmentation/modeling/modeling_wrapper.py)
3. [Statistical analysis](phoneme_segmentation/analysis/)
4. [Simulation](phoneme_segmentation/simulation/simulation_wrapper.py)
5. [Visualization](phoneme_segmentation/viz/) 
