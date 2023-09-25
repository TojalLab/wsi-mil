
## wsi-mil

A framework to train weakly supervised deep learning models on whole-slide images.

## Setup

Install dependecies of the project with conda:

`$ mamba env create --file conda.env.yaml`

## Configuration

The configuration file instructs the software about how it is suppose to use the resources, such as metadata file, file locations, artifact directory, target variable. See the example configuration available in `configs/example_config.yaml`

### Create a metadata table file that points to your data

The metadata file is used by the software to keep track of the slides and its classification targets. Of note, you can set several targets for the slide by adding new target columns. Slides can be skipped in a given target column by setting it to `NaN`. See the example metadata file available in `inputs/example_metadata.tsv`. 

### Pre-trained helper models

The weights of the feature extractor model can be downloaded here: [RetCCL](https://github.com/Xiyue-Wang/RetCCL),
download the weights and add the `RetCCL_resnet50_weights.pth` file to the `pretrained_models` directory.

The weights of the tile filtering models can be downloaded [here](https://github.com/TojalLab/wsi-mil/releases/tag/pretrained_models).

More information regarding these networks can be found [here](/filtering%20networks%20-%20instructions.md).

### Set a target label

The `common.target_label` setting, on the config file selects which column of the metadata file will be used as target variable.

### Adjust other parameters

The config file have many other paramters that can be changed, check the [configs/example\_config.yaml](configs/example_config.yaml) file.

## Run the steps

The pipeline is split into the following steps:

- create\_tiles: create a tile map of all slides and filter tiles of interest according to the pretrained filter NN.
- extract\_features: run the feature extraction network on all tiles of interest
- create\_splits: create the train/test splits for all folds
- train\_model: train the model on all folds
- create\_att\_heatmaps: generate attention heatmaps from the trained NN results

Run each step in order with the `run_steps.sh` script.

## License

wsi-mil code is released under the GPLv3 License.

## Acknowledgements and References

Models implemented and used here are a re-implementation and/or inspired on existing works:

- ADMIL: [code](https://github.com/AMLab-Amsterdam/AttentionDeepMIL), [paper](https://arxiv.org/abs/1802.04712)

- CLAM: [code](https://github.com/mahmoodlab/CLAM), [paper](https://www.nature.com/articles/s41551-020-00682-w)

- RetCCL: [code](https://github.com/Xiyue-Wang/RetCCL), [paper](https://doi.org/10.1016/j.media.2022.102645)

