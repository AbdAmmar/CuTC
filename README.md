# CUDA Library for Transcorrelated Integrals 

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Overview

This CUDA library is designed for the efficient computation of Transcorrelated integrals and dressing elements for the application of normal-ordering using orthogonal or bi-orthogonal molecular orbitals.


## Features

### Transcorrelated Integrals

The library computes and provides:
- **2-Electron Transcorrelated Integrals**: Excluding the Coulomb term.
- **Key Tersors for 3-Electron Integrals**: Specifically, the tensors of the form:

$$
X_{\mu \nu}(\mathbf{r_1}) = \int d \mathbf{r_2} \chi_\mu(\mathbf{r_2}) \chi_\nu(\mathbf{r_2}) \nabla_1 u(\mathbf{r_1}, \mathbf{r_2})
$$
  
  where $\nabla_1 u(\mathbf{r_1}, \mathbf{r_2})$ represents the gradient of the function $u$ used to define the Jastrow factor.

#### Input

- **Grid Sets**: Two sets of grids $(\mathbf{r_1}, w_1)$ and $(\mathbf{r_2}, w_2)$, where $\mathbf{r_1}$ and $\mathbf{r_2}$ are spatial coordinates, and $w_1$ and $w_2$ are corresponding weights.
- **Atomic Orbitals**: A set of atomic orbitals $\{\chi_\mu\}$ evaluated on the provided grids.
- **Nuclear Positions**: The positions of nuclei in the system.
- **Jastrow Parameters**: Parameters associated with the Jastrow factor.


### Dressing Elements for Normal-Ordering

0-electron, 1-electron, and 2-electron dressing integrals are computed to reduce the 3-electron Transcorrelated Hamiltonian to an effective 2-electron Hamiltonian.

#### Input

- **Number of Electrons**: The number of $\uparrow$-electrons and $\downarrow$-electrons.
- **Left and Right Orbitals on a Grid**: Two sets of orbitals computed on a grid with the corresponding weights.
- **3-Electron Tensor**: The tensor $X_{\mu \nu}(\mathbf{r_1})$ transformed into molecular orbitals


## Installation

```bash
git clone https://github.com/AbdAmmar/CuTC.git
cd CuTC
source config/env.rc
make
```

## Usage

The primary routine of the Transcorrelated Integrals CUDA Library is implemented in C and can be easily integrated into projects using other programming languages. 
To use the library, include the compiled shared library `libcutcint.so` in your project.

### Integration

1. **Linking the Library**:
   - Ensure that the shared library `CuTC/build/libcutcint.so` is accessible to your project. You may need to specify the path to this library when linking your application.

2. **Calling from C/C++**:
   - You can directly call the functions defined in the library from your C or C++ code. Include the relevant headers and link against the shared library during the build process.

### Fortran Integration

A demonstration of how to use the library with Fortran is provided. To see an example of calling the CUDA library from Fortran, refer to the following files:
- **Fortran Source**: `CuTC/src/cutc_int_f.f90`
- **Fortran Module**: `CuTC/src/cutc_module.f90`

These files contain example code and a module that interfaces with the C routine, showing how to invoke the library functions from Fortran.


## Acknowledgments

CuTC is supported by the [PTEROSOR](https://lcpq.github.io/PTEROSOR/) project that has received funding from the 
European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (Grant 
agreement No. 863481).

<img src="https://lcpq.github.io/PTEROSOR/img/ERC.png" width="200" />

