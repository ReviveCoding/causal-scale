# Third-Party Notices

## CAUSAL-SCALE source code

Original CAUSAL-SCALE source code is available under the repository's [MIT License](LICENSE). That license does not grant rights to third-party datasets, software, model artifacts, trademarks, or other materials.

## Criteo Uplift dataset

The Criteo Uplift Prediction Dataset is not distributed in this repository. Users must obtain it from Criteo and comply with the provider's terms, including the applicable Creative Commons Attribution-NonCommercial-ShareAlike terms identified in [the dataset configuration](configs/datasets.yaml). CAUSAL-SCALE does not claim ownership of the dataset. Dataset hashes and aggregate experimental results are included only to identify inputs and support reproducibility.

## Other benchmark datasets

External benchmark datasets, including Twins and any optionally used IHDP or Hillstrom data, are not relicensed by CAUSAL-SCALE. They remain subject to their respective providers' licenses, attribution requirements, and usage restrictions. Raw copies are not part of the public repository.

## Python and systems dependencies

Third-party Python packages, Java/Spark components, CUDA tooling, and their transitive dependencies retain their own licenses. Package names and versions appear in `pyproject.toml` and `requirements.lock`; inclusion as a dependency does not incorporate those projects into the MIT-licensed CAUSAL-SCALE source.
