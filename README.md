# ParTIpy: A Scalable Framework for Archetypal Analysis and Pareto Task Inference

This repository accompanies the preprint **“ParTIpy: A Scalable Framework for Archetypal Analysis and Pareto Task Inference”**.

It contains the code to reproduce benchmarks of initialization and optimization strategies, as well as experiments evaluating coreset sizes and performance gains in Archetypal Analysis.

For additional examples and applications of Archetypal Analysis and Pareto Task Inference on single-cell and spatial omics data, please refer to the [documentation](https://partipy.readthedocs.io/en/latest/).

---

## Installation

We recommend installing dependencies with **mamba** or **mamba**.  

```bash
mamba env create -n partipy -f env.yaml
```

This will install all required packages, including [ParTIpy](https://github.com/saezlab/ParTIpy)

## Download Data

Benchmark and example datasets are downloaded automatically when running the benchmark scripts (via functions defined in `utils/data_utils.py`).

## Benchmarks

### Initialization & Optimization Algorithms

```bash
mamba activate partipy

python -m code.benchmark_algorithms.ms_bench
python -m code.benchmark_algorithms.ms_xenium_bench
python -m code.benchmark_algorithms.lupus_bench
python -m code.benchmark_algorithms.bench_meta
```

### Coresets

```bash
mamba activate partipy

python -m code.benchmark_coresets.ms_coreset
python -m code.benchmark_coresets.ms_xenium_coreset
python -m code.benchmark_coresets.lupus_coreset
python -m code.benchmark_coresets.coreset_meta
```

### Memory

```bash
mamba activate partipy

python -m code.benchmark_memory.k562_memory_bench.py
```

## Examples

### Simulated Data

```bash
mamba activate partipy

python -m code.benchmark_algorithms.simulated_data
```

### Hepatocyte Example

```bash
mamba activate partipy

python -m code.examples.hepatocyte_example
```

### Additional Scripts

```bash
mamba activate partipy

python -m code.examples.overview_figure
python -m code.examples.delta_visualization
```

## Citing ParTIpy

If you use ParTIpy in your work, please cite the following preprint:

```
@article{schaefer2025partipy,
  title   = {ParTIpy: A Scalable Framework for Archetypal Analysis and Pareto Task Inference},
  author  = {Schäfer, Philipp S. L. and Zimmermann, Leoni and Burmedi, Paul L. and Walfisch, Avia and Goldenberg, Noa and Yonassi, Shira and Shaer Tamar, Einat and Adler, Miri and Tanevski, Jovan and Ramirez Flores, Ricardo O. and Saez-Rodriguez, Julio},
  year    = {2025},
  journal = {bioRxiv},
  doi     = {10.1101/xxxxxx},
}
```