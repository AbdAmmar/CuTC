# Transcorrelated Integrals CUDA Library

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Overview

This CUDA library is designed to efficiently calculate Transcorrelated integrals, leveraging the power of GPU computing.

### Input

The library requires the following inputs:
- **Grid Sets**: Two sets of grids $(\mathbf{r_1}, w_1)$ and $(\mathbf{r_2}, w_2)$, where $\mathbf{r_1}$ and $\mathbf{r_2}$ are spatial coordinates, and $w_1$ and $w_2$ are corresponding weights.
- **Atomic Orbitals**: A set of atomic orbitals $\{\chi_\mu\}$ evaluated on the provided grids.
- **Nuclear Positions**: The positions of nuclei in the system.
- **Jastrow Parameters**: Parameters associated with the Jastrow factor.

### Output

The library computes and provides:
- **2-Electron Transcorrelated (TC) Integrals**: Excluding the Coulomb term.
- **Key Tersors for 3-Electron Integrals**: Specifically, the tensors of the form:

$$
\int d \mathbf{r_2} \chi_\mu(\mathbf{r_2}) \chi_\nu(\mathbf{r_2}) \nabla u(\mathbf{r_1}, \mathbf{r_2})
$$
  
  where $\nabla u(\mathbf{r_1}, \mathbf{r_2})$ represents the gradient of the function $u$ used to define the Jastrow factor.


## Installation

```bash
git clone https://github.com/AbdAmmar/CuTC.git
cd CuTC
source config/env.rc
make
```

## Usage

The primary routine of the Transcorrelated Integrals CUDA Library is implemented in C and can be easily integrated into projects using other programming languages. 
To use the library, include the compiled shared library `libtc_int_cu.so` in your project.

### Integration

1. **Linking the Library**:
   - Ensure that the shared library `CuTC/build/libtc_int_cu.so` is accessible to your project. You may need to specify the path to this library when linking your application.

2. **Calling from C/C++**:
   - You can directly call the functions defined in the library from your C or C++ code. Include the relevant headers and link against the shared library during the build process.

### Fortran Integration

A demonstration of how to use the library with Fortran is provided. To see an example of calling the CUDA library from Fortran, refer to the following files:
- **Fortran Source**: `CuTC/src/tc_int_f.f90`
- **Fortran Module**: `CuTC/src/cutc_module.f90`

These files contain example code and a module that interfaces with the C routine, showing how to invoke the library functions from Fortran.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


