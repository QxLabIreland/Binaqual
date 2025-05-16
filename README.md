# Binaqual
Python implementation of BINAQUAL localization similarity metric for binaural audio.
## Setup
Install the required packages by running:

`pip install -r requirements.txt`

## Usage
The program can be used by running:

`python -m binaqual --ref /path/to/dir/reference_signal --deg /path/to/dir/test_sginal`


## Validation

To validate the BINAQUAL metric, first download the SynBAD dataset from the following link and extract it in the main directory:
https://zenodo.org/records/15431990

Then, run the model_validation.py script under the validation directory to apply Binaqual on the SynBAD dataset, as used in the paper's experiments. The results can then be plotted using the plots.py script.


## Citation
If you use this code, please cite the associated paper:


## Licence
This project is licensed under the Apache 2.0 License.
