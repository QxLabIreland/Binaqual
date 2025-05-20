# Binaqual
Python implementation of BINAQUAL localization similarity metric for binaural audio.
## Setup
Install the required packages by running:

`pip install -r requirements.txt`

## Usage
The program can be used by running:

`python -m binaqual --ref /path/to/dir/reference_signal --deg /path/to/dir/test_sginal`


## Model validation

To validate the BINAQUAL metric, first download the SynBAD dataset from the following link and extract it in the main directory:
https://zenodo.org/records/15431990

Then, run the model_validation.py script under the validation directory to apply Binaqual on the SynBAD dataset, as used in the paper's experiments. The results can then be plotted using the plots.py script.


## Citation
If you use this code, please cite the associated [paper](http://arxiv.org/abs/2505.11915):
```bibtex
@misc{panah2025binaqualfullreferenceobjectivelocalization,
      title={BINAQUAL: A Full-Reference Objective Localization Similarity Metric for Binaural Audio}, 
      author={Davoud Shariat Panah and Dan Barry and Alessandro Ragano and Jan Skoglund and Andrew Hines},
      year={2025},
      eprint={2505.11915},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2505.11915}, 
}
```

If you use the SynBAD dataset, cite the above paper and the Binamix library [paper](https://arxiv.org/abs/2505.01369):

```bibtex
@misc{barry2025binamixpythonlibrary,
      title={Binamix - A Python Library for Generating Binaural Audio Datasets}, 
      author={Dan Barry and Davoud Shariat Panah and Alessandro Ragano and Jan Skoglund and Andrew Hines},
      year={2025},
      eprint={2505.01369},
      journal={arXiv preprint arXiv:2505.01369},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2505.01369}, 
}
```


## Licence
This project is licensed under the Apache 2.0 License.
