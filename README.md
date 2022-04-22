# genray-cql3d-ml
Machine learning models, code, sample data for approximating the computations of the Genray/CQL3D codes.

## Description

Three machine learning techniques (multilayer perceptron, random forest, and Gaussian process) provide fast surrogate models for lower hybrid current drive (LHCD) simulations.  A single GENRAY/CQL3D simulation without radial diffusion of fast electrons requires several minutes of wall-clock time to complete, which is acceptable for many purposes, but too slow for integrated modeling and real-time control applications.  The machine learning models use a database of 16,000+ GENRAY/CQL3D simulations for training, validation, and testing.  Latin hypercube sampling methods ensure that the database covers the range of 9 input parameters ($n_{e0}$, $T_{e0}$, $I_p$, $B_t$, $R_0$, $n_{||}$, $Z_{eff}$, $V_{loop}$, $P_{LHCD}$) with sufficient density in all regions of parameter space.  The surrogate models reduce the inference time from minutes to ~ms with high accuracy across the input parameter space.

## Getting Started

### Dependencies

* GPflow, Jupyter notebook, NumPy, ONNX, ONNX Runtime, PyTorch.
* Mac OSX, linux and Windows.

### Installing

* install dependencies

### Executing program

* check onnx API
```
onnx_model = onnx.load("MLP_trained_power.onnx")
onnx.checker.check_model(onnx_model)
```
* run model using onnxruntime
```
ort_sess = onnxruntime.InferenceSession("MLP_trained_power.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: x} 
ort_outs = ort_session.run(None, ort_inputs)
```

## Authors

Contributors names and contact info

## Version History

* 1.0
    * Initial Release

## License

This project is licensed under the BSD License - see the LICENSE.md file for details.

## Data availability

* [data source](https://doi.org/10.7910/DVN/5YY6PE)

