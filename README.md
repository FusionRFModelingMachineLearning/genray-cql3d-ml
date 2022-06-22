# genray-cql3d-ml
Machine learning models, code, sample data for approximating the computations of the Genray/CQL3D codes.

## Description

This repository provides access to code, data, trained machine learning models, and instructions for reproducing work of a recent publication, Towards Fast and Accurate Predictions of RF Power Deposition and Current Profile, by G. Wallace et. al, 2022.

Three machine learning techniques (multilayer perceptron, random forest, and Gaussian process) provide fast surrogate models for lower hybrid current drive (LHCD) simulations.  A single GENRAY/CQL3D simulation without radial diffusion of fast electrons requires several minutes of wall-clock time to complete, which is acceptable for many purposes, but too slow for integrated modeling and real-time control applications.  The machine learning models use a database of 16,000+ GENRAY/CQL3D simulations for training, validation, and testing.  Latin hypercube sampling methods ensure that the database covers the range of 9 input parameters ($n_{e0}$, $T_{e0}$, $I_p$, $B_t$, $R_0$, $n_{||}$, $Z_{eff}$, $V_{loop}$, $P_{LHCD}$) with sufficient density in all regions of parameter space.  The surrogate models reduce the inference time from minutes to ~ms with high accuracy across the input parameter space.

## Getting Started

### Manifest

This repository contains:
* Models: trained ML models for Multi-Layer Perceptron (MLP),  Random Forest regression (RFR), and Gaussian Process regression (GPR). In some cases (GPR, MLP), the trained models are "small" and are located in this github repo. In other cases (RFR), the trained model is too large for the github repo, and is accessible in the subsection below: [External Data URL](#external-data-url).
* Code: Jupyter notebooks containing code that will load a trained ML model, ingest data from the simulation database, and perform an inference workload
* Data: the simulation database is accessible at a URL shown below in the section of this document entitled [External Data URL](#external-data-url).
 


### Dependencies

* ONNX, ONNX Runtime, Tensorflow.
* Environment: Python or Jupyter notebook.
* Mac OSX, linux and Windows.

### Installing

```
pip install onnx
pip install onnxruntime
```
Note that onnx 1.10.2 and onnxruntime 1.10.0 have been tested to work on Python 3.7. Latest released version of the tools are backwards compatible.

### Executing program

* check onnx API
```
onnx_model = onnx.load("MLP_trained_power.onnx")
onnx.checker.check_model(onnx_model)
```
* run model using onnxruntime
```
ort_session = onnxruntime.InferenceSession("MLP_trained_power.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: 'input'} 
ort_outputs = ort_session.run(None, ort_inputs)
```

## Authors

* Zhe Bai, zhebai@lbl.gov
* E. Wes Bethel, ewesbethel@gmail.com

## Version History

* 1.0
    * Initial Release

## About
## License

This project is licensed under the BSD License - see the [LICENSE file](License.txt) for details.

## External Data URL 

The simulation database and trained RFR models are accessible [at this location at the Plasma Science and Fusion Center 
Dataverse](https://doi.org/10.7910/DVN/5YY6PE).

*** Copyright Notice ***

Fusion RF Modeling Machine Learning (FusionML_RF) Copyright (c) 2022, 
The Regents of the University of California, through Lawrence Berkeley 
National Laboratory and Princeton Plasma Physics Laboratory (subject to 
receipt of any required approvals from the U.S. Dept. of Energy), and 
Massachusetts Institute of Technology.  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.


