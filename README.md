# RAndom Circuit Block Encoded Matrix (RACBEM)  

This is a python module implementing RACBEM using IBM's Qiskit. It can used in various quantum linear algebra problems.



## Dependencies and Installing Instruction

+ Pyhton3.7, IBM Qiskit, QuTiP, numpy, scipy

+ Installing Instruction:

  1. Install Anaconda for python-package management, avaliable at https://www.anaconda.com/products/individual

     Go to the terminal after installing Anaconda.

  2. If you have already had a Python3.7 environment, just skip this step.

     `conda create --name=$ENV_NAME python=3.7 # have the environment use Python 3.7`

     `source activate $ENV_NAME`

  3. Make sure the modules for scientific computing, **numpy, scipy**, are installed.

     `conda install numpy scipy`

     You are also suggested to install **matplotlib** for visualization and **Cython** due to the requirement of QuTiP.

     `conda install matplotlib cython`

  4. Install the package **pickle** to load/save data locally.

     `conda install pickle`

  5. Install **QuTiP**. If the installation is terminated due to missing dependent packages, please check the document provided by QuTiP, http://qutip.org/docs/latest/installation.html

     `conda install qutip`

     If conda doesn't work, try pip for instead.

     `pip install qutip`

     You can also install directly from source if both conda and pip don't work but it's rare. See **Installing from Source** in the document for details. http://qutip.org/docs/latest/installation.html

  6. Install **IBM Qiskit**. See the document for details. https://qiskit.org/documentation/install.html

     `pip install qiskit`

  7. In your **python** environment, test whether you successfully install **IBM Qiskit**.

     `python`

     `>>> import qiskit`

     If the module is loaded, the installation is all set.

  8. Run our demo code.

     `python main_test.py`

You are suggested to get and save locally an IBM account which is used frequently when using IBM Qiskit. See **Access IBM Quantum Systems** in the document for details. https://qiskit.org/documentation/install.html



## Introduction

This is an implementation of the a RAndom Circuit Block Encoded Matrix (RACBEM) and its Hermitian conjugate. It is then used to build a quantum singular value circuit using the method of quantum singular value transformation (QSVT).

Take a RAndom Circuit Block Encoded Matrix (RACBEM), this function uses a quantum signal processing circuit to evaluate the matrix inverse, using the method of quantum singular value transformation (QSVT). This implements a (non-Hermitian) block-encoding of a Hermitian matrix manually.



## References

+ [Yulong Dong, Lin Lin. Random circuit block-encoded matrix and a proposal of quantum LINPACK benchmark. arXiv: 2006.04010](http://arxiv.org/abs/2006.04010)
+ [Y. Dong, X. Meng, K. B. Whaley, and L. Lin. Efficient Phase Factor Evaluation in Quantum Signal Processing. arXiv: 2002.11649](https://arxiv.org/abs/2002.11649)
+ [A. Gilyén, Y. Su, G. H. Low, and N. Wiebe. Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics. In Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing, pages 193–204, 2019](https://dl.acm.org/doi/10.1145/3313276.3316366)



## Citing our work

If you find our work useful or you use our work in your own project, please consider to cite our work.



## The authors

We hope that the package is useful for your application. If you have any bug reports or comments, please feel free to email one of the software authors:

* Yulong Dong, dongyl@berkeley.edu

* Lin Lin, linlin@math.berkeley.edu

  

## Notice

It's possible to get the following **warning** message when importing **racbem**. It's due to a problem of QuTiP (github.com/qutip/qutip/issues/1205). Please just **ignore** it because it doesn't affect the computation.

It may happen in MacOS, Windows or Linux.

```
/Users/$USER/.pyxbld/temp.macosx-10.9-x86_64-3.8/pyrex/qutip/cy/openmp/parfuncs.cpp:614:10: fatal error: 'src/zspmv_openmp.hpp' file not found
#include "src/zspmv_openmp.hpp"
         ^~~~~~~~~~~~~~~~~~~~~~
1 error generated.
```



## Running the demo

`python main_test.py`

The output should be like the follows.

```
retrieve architecture from IBM Q and save locally at: ibmq_burlington_backend_config.pkl

kappa=5, sigma=1.00, polynomial approximation error=1.902e-02

Generic RACBEM
singular value (A) = 
 [0.99  0.979 0.952 0.79  0.694 0.306 0.248 0.204]
Job Status: job has successfully run
succ prob (exact)     =  0.22205160210017913
succ prob (noiseless) =  0.23183536309935282
succ prob (measure)   =  0.2242431640625

Hermitian RACBEM
singular value (A) = 
 [0.963 0.91  0.908 0.802 0.397 0.292 0.289 0.236]
condition number (A)  = 4.074
||A - A^\dagger||_2   = 2.675e-15
```



## Generating phase factors using QSPPACK

To implement QSVT, a set of phase factors which characterizes the target function is needed. You can generate phase factors by using **QSPPACK**.

### Installing Instruction:

1. Download the latest source code in our Github repository. https://github.com/qsppack/QSPPACK

2. Go to the directory where you save QSPPACK and run the following command in **Matlab** terminal.

   `>> startup`

### Running the demo:

We provide a demo code to generate phase factors which solve QLSP whose condition number is upper bounded. You can follow the steps below after you set up QSPPACK.

1. Open `Remez.ipynb` in **jupyter notebook**. Make sure you installed **Julia** language. See https://julialang.org for details.

   Run `jupyter notebook` or `LANG=zn jupyter notebook` in terminal under the directory of RACBEM to open the notebook.

2. Run the code in the notebook. A data file `coef_5_6.mat` will apear in your directory. A polynomial approximation is saved in that file.

3. In **Matlab** terminal under the directory of RACBEM, run `>> GeneratePhi(5,6)`. You will get 2 figures, the following messages, and a txt file `phi_inv_5.txt` in which the informations about phase factors are saved.

   ```
   approx error (inf) of coef = 0.0223098
   extra scaling factor = 1.17326
   total scaling factor = 5.86631
   L-BFGS solver started 
   iter          obj  stepsize des_ratio
      1  +3.3827e-03 +1.00e+00 +4.95e-01
      2  +1.0600e-03 +1.00e+00 +7.08e-01
      3  +8.8408e-05 +1.00e+00 +6.04e-01
      4  +3.9504e-06 +1.00e+00 +6.05e-01
      5  +6.4384e-08 +1.00e+00 +5.41e-01
      6  +1.0029e-09 +1.00e+00 +5.38e-01
      7  +2.7395e-11 +1.00e+00 +5.60e-01
      8  +5.8496e-14 +1.00e+00 +5.14e-01
      9  +7.4040e-17 +1.00e+00 +5.15e-01
     10  +8.5125e-21 +1.00e+00 +5.04e-01
   iter          obj  stepsize des_ratio
     11  +5.2673e-24 +1.00e+00 +5.06e-01
     12  +7.8591e-27 +1.00e+00 +5.16e-01
   Stop criteria satisfied.
   - Info: 		QSP phase factors --- solved by L-BFGS
   - Parity: 		even
   - Degree: 		6
   - Iteration times: 	12
   - CPU time: 	0.0 s
   approx error (inf) of polynomial = 1.384e-13
   approx error (inf) of g circ h = 1.902e-02
   ```

   

