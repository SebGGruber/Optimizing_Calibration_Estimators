# Optimizing Estimators of Squared Calibration Errors in Classification

This is the code that accompanies the paper ["Optimizing Estimators of Squared Calibration Errors in Classification"](https://openreview.net/forum?id=BPDVZajOW5), published at TMLR.

We propose a novel risk-based approach to compare and optimize estimators of squared calibration errors in the classification setup.
This allows to make a definitive, data-driven choice on the calibration estimator for a given classifier.

## To Reproduce our Results

All evaluation outputs are presented in Jupyter Notebooks for an easier and more interactive exploration of our experiments.
The conda environment is located in `environment.yml` and should be installed first.

### Simulations

All simulation results can be found in `estimator_simulation.ipynb`.
The simulations are light-weight and can be easily run on a low-budget laptop.

### Real-world Experiments

All real-world experiments evaluate the logit predictions of common classifiers and image classification datasets.
To download the zipped logits, run `gdown https://drive.google.com/uc?id=1aVgR5X4lbgBaOnppikx6eyYhG0nuv3XG`.

To extract them into the folder `logits` run `unzip logits-20250107T131832Z-001.zip`.

All evaluations involving convolutional neural networks are located in `cnn_calib_evaluations.ipynb`.
All evaluations involving VisionTransformers are located in `vit_calib_evaluations.ipynb`.
The figures are printed as cell outputs and are also stored in the folder `plots`.

These experiments are more costly to run due to the n^2 and n^3 complexity of some estimators.
Consequently, some of the cells may take too long to run.
In such cases, one may run `run_experiments.py` to run the experiments in a shell (note the flag `notion` in the python script, which is used to switch between top-label confidence calibration and canonical calibration).
The results are stored in the folder `results` and still have to be evaluated in the jupyter notebook after the script has finished.

## Reference
If you found this work or code useful, please cite:

```
@article{
gruber2025optimizing,
title={Optimizing Estimators of Squared Calibration Errors in Classification},
author={Gruber, Sebastian and Bach, Francis},
journal={Transactions on Machine Learning Research},
year={2025},
url={https://openreview.net/forum?id=BPDVZajOW5},
}
```

## License

Everything is licensed under the [MIT License](https://opensource.org/licenses/MIT).
