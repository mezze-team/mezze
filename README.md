# mezze

A toolbox for simulating open quantum system dynamics

## Getting Started

Install using `pip install` from the cloned repo.

Look at the notebooks in the `examples` folder, and also `tests` for more granular usage.

### Citing mezze

If you use the SchWARMA tools in your research, please cite:

Schultz, K., Quiroz, G., Titum, P., & Clader, B. D. (2020). SchWARMA: A model-based approach for time-correlated noise in quantum circuits. arXiv preprint [arXiv:2010.04580](https://arxiv.org/abs/2010.04580).

## Tensorflow Quantum Branch

To use the SchWARMA tools implemented using [TensorFlow Quantum](https://www.tensorflow.org/quantum) and [Cirq](https://github.com/quantumlib/Cirq):

* Install TensorFlow Quantum using the instructions [here](https://www.tensorflow.org/quantum/install)
* Fetch remote branches - `git fetch`
* Checkout the `tfq` branch of mezze - `git checkout tfq`
* Install mezze - `pip install -e .` (or similar)

This will resolve issues with dependencies that occur by using `pip` only
