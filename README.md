# mezze

A toolbox for simulating open quantum system dynamics

## Getting Started

There are some issues with dependencies that may occur when using `pip` only so instead follow these steps:

* Install TensorFlow Quantum using the instructions [here](https://www.tensorflow.org/quantum/install)
  * If installation fails on `sympy`, clone and install via git or with `conda install sympy` and repeat above instructions
* Install mezze - `pip install -e .` (or similar)

Install using `pip install` from the cloned repo. 

Look at the notebooks in the `examples` folder, and also `tests` for more granular usage.

### Windows Install Issues

At the time of this writing, TensorFlow Quantum is not maintained for Windows and so the above instructions will fail.  Mezze can be installed using only Cirq but the TensorFlow-based simulators and inference routines will not be loaded. Cirq-based simulators will continue to work.

Windows install directions:
* `pip install cirq`
  * If installation fails on `sympy`, clone and install via git or with `conda install sympy` and repeat above instructions
* `pip install --no-deps -e .`


### Citing mezze

If you use the SchWARMA tools in your research, please cite:

Schultz, K., Quiroz, G., Titum, P., & Clader, B. D. (2020). SchWARMA: A model-based approach for time-correlated noise in quantum circuits. Physical Review Research 3, 033229. Available as an arXiv preprint [arXiv:2010.04580](https://arxiv.org/abs/2010.04580).

If you use SchWARMA for noise injection, please cite:

Murphy, A., Epstein, J., Quiroz, G., Schultz, K., Tewala, L., McElroy, K., ... & Sweeney, T. M. (2021). Universal Dephasing Noise Injection via Schrodinger Wave Autoregressive Moving Average Models. arXiv preprint [arXiv:2102.03370](https://arxiv.org/abs/2102.03370).

If you experiment with the heavy-tailed (Levy-alpha stable) variants of SchWARMA, please cite:

Clader, B. D., Trout, C. J., Barnes, J. P., Schultz, K., Quiroz, G., & Titum, P. (2021). Impact of correlations and heavy-tails on quantum error correction. Physical Review A 103, 052428. Available as an arXiv preprint [arXiv:2101.11631](https://arxiv.org/abs/2101.11631).

## Tensorflow Quantum Branch

Note that most of the TensorFlow Quantum-based tools are now in the main branch, and tfq will only be for developing capabilities

To use the SchWARMA tools implemented using [TensorFlow Quantum](https://www.tensorflow.org/quantum) and [Cirq](https://github.com/quantumlib/Cirq):

* Install TensorFlow Quantum using the instructions [here](https://www.tensorflow.org/quantum/install)
* Fetch remote branches - `git fetch`
* Checkout the `tfq` branch of mezze - `git checkout tfq`
* Install mezze - `pip install -e .` (or similar)

This will resolve issues with dependencies that occur by using `pip` only
