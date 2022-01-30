# mezze

A toolbox for simulating open quantum system dynamics

## Getting Started


For users who just want to use the master equation solver, channel classes, and/or `cirq`-based SchWARMA simulation cloning the repo and running `pip install . `  from the project directory should work fine. The install can be verified by installing `nose` and running `nosetests` from the project directory. Some tests may fail with extremely small numerical differences based on random inputs.

The basic install does not include the faster Tensorflow-based simulation routines as well as the inference routines.  Note that some of the example notebooks use this capability. To use these capabilities, see the next session.

### Advanced Install -- Tensorflow-based simulation and inference

There are some issues with dependencies that may occur when using `pip` only so instead follow these steps:

* Install TensorFlow Quantum using the instructions [here](https://www.tensorflow.org/quantum/install)
  * *Note* sometimes there are issues with these directions, you may need to try other versions of tensorflow and tensorflow-quantum (in particular `tfq-nightly` will often work)
  * If installation fails on `sympy`, clone and install via git or with `conda install sympy` and repeat above instructions
* Install mezze - `pip install .` (or similar)

Even if there are `pip` issues with dependencies, as long as the tests run you should be fine.

## Usage

Look at the notebooks in the `examples` folder, and also `tests` for more granular usage. The example notebooks have the following structure:

* `examples` - notebooks in this directory cover the older `mezze` functionality for manipulating quanutum states and channels, as well as stochastic master equation simulation
* `examples/SchWARMAPaperExamples` - notebooks in this directory generate the figures for Ref. [1], and are instructive for the finer details of SchWARMA, but most of that functionality has incorporated in a faster and more flexible codebase in `mezze.mtfq`
* `examples/TFQ` - notebooks that cover the new functionality uses `cirq` objects and "SchWARMAFies" them to induce spatiotemporally correlated noise, these can then be simulated by simulators from `cirq`, `tensorflow-quantum`, and `qsim` (assuming the latter two are installed). This directory also contains an example of how to do noise injection in the IBMQE as in [2]. The `mtfq` codebase also supports heavy-tailed distributions as in [3].
* `examples/ZNE` - notebooks from the paper [4], that show how temporally correlated dephasing noise impacts circuit folding techniques for zero noise extrapolation. Also has some examples of the recent filter function interface from the appendix of [4]. 



### Citing mezze

If you use the SchWARMA tools in your research, please cite:

[1] Schultz, K., Quiroz, G., Titum, P., & Clader, B. D. (2020). SchWARMA: A model-based approach for time-correlated noise in quantum circuits. Physical Review Research 3, 033229. Available as an arXiv preprint [arXiv:2010.04580](https://arxiv.org/abs/2010.04580).

If you use SchWARMA for noise injection, please cite:

[2] Murphy, A., Epstein, J., Quiroz, G., Schultz, K., Tewala, L., McElroy, K., ... & Sweeney, T. M. (2021). Universal Dephasing Noise Injection via Schrodinger Wave Autoregressive Moving Average Models. arXiv preprint [arXiv:2102.03370](https://arxiv.org/abs/2102.03370).

If you experiment with the heavy-tailed (Levy-alpha stable) variants of SchWARMA, please cite:

[3] Clader, B. D., Trout, C. J., Barnes, J. P., Schultz, K., Quiroz, G., & Titum, P. (2021). Impact of correlations and heavy-tails on quantum error correction. Physical Review A 103, 052428. Available as an arXiv preprint [arXiv:2101.11631](https://arxiv.org/abs/2101.11631).

The zero noise extrapolation examples are from 

[4] Schultz, K., LaRose, R., Mari, A.,Quiroz, G., Shammah, N., Clader, B. D., and Zeng, W. J. (2022) Reducing the impact of time-correlated noise on zero-noise extrapolation.

## Tensorflow Quantum Branch

Note that most of the TensorFlow Quantum-based tools are now in the main branch, and tfq will only be for developing capabilities

To use the SchWARMA tools implemented using [TensorFlow Quantum](https://www.tensorflow.org/quantum) and [Cirq](https://github.com/quantumlib/Cirq):

* Install TensorFlow Quantum using the instructions [here](https://www.tensorflow.org/quantum/install)
* Fetch remote branches - `git fetch`
* Checkout the `tfq` branch of mezze - `git checkout tfq`
* Install mezze - `pip install -e .` (or similar)

This will resolve issues with dependencies that occur by using `pip` only
