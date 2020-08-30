"""Solving QLSP and testing Hermitian RACBEM.

This is a demo code of RACBEM.

This includes a utility function for generating the matrix A (to be
included in the racbem.py) and the utility function for generating
random circuit (to be included in the random_circuit.py).

This tests the success probability of solving QLSP and tests that the 
condition number of the Hermitian RACBEM is upper bounded by the given value.

References:
    Yulong Dong, Lin Lin. Random circuit block-encoded matrix and a proposal 
        of quantum LINPACK benchmark.
Authors:
    Yulong Dong     dongyl (at) berkeley (dot) edu
    Lin Lin         linlin (at) math (dot) berkeley (dot) edu
Version: 1.0
Last revision: 06/2020
"""

import numpy as np
import scipy.linalg as la

from qiskit import execute
from qiskit import Aer
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer.noise import NoiseModel

from racbem import *

import os
import pickle

def GetBackend(backend_name=None):
    if backend_name == None:
        backend = Aer.get_backend('qasm_simulator')
    else:
        provider = IBMQ.load_account()
        backend = provider.get_backend(backend_name)
    return backend

if __name__ == '__main__':
    backend_name = 'ibmq_burlington'
    #backend_name = None
    kappa = 5                   # condition number
    n_sys_qubit = 3             # the number of system qubits
    n_be_qubit = 1              # the number of block-encoding qubit
    n_sig_qubit = 1             # the number of signal qubit
    n_tot_qubit = n_sig_qubit+n_be_qubit+n_sys_qubit
    n_depth = 15                # the depth of random circuit
    prob_one_q_op = 0.5         # the probability of selecting a one-qubit
                                # operation when two_q_op is allowed
    basis_gates = ['u1','u2','cx']
    digit_shots = 13
    n_shots = 2**digit_shots    # the number of shots used in measurements
    sigma = 1.0                 # parameter used to rescale noise model
    # state |0^n>
    b = np.zeros((2**n_sys_qubit,))
    b[0] = 1.0
    load_architecture = True    # True:     load architure locally
                                # False:    need to save an IBM account beforehand

    # instances of RACBEM
    be = BlockEncoding(n_be_qubit, n_sys_qubit)
    qsp = QSPCircuit(n_sig_qubit, n_be_qubit, n_sys_qubit)

    # retrieve backends and architectures
    backend = GetBackend()
    if load_architecture:
        if os.path.exists(backend_name+'_backend_config.pkl'):
            noise_backend = pickle.load(open(backend_name+'_backend_config.pkl','rb'))
            noise_model = NoiseModel.from_dict(noise_backend['noise_dict'])
            coupling_map = noise_backend['coupling_map']
            tot_q_device = noise_backend['tot_q_device']
            print("load architecture locally at: %s_backend_config.pkl\n"%(backend_name))
        else:
            raise Exception("no locally saved architecture: %s_backend_config.pkl"%(backend_name), load_architecture)
    else:
        noise_backend = GetBackend(backend_name=backend_name)
        coupling_map = noise_backend.configuration().coupling_map
        noise_model = NoiseModel.from_backend(noise_backend)
        tot_q_device = noise_backend.configuration().n_qubits
        pickle.dump({'noise_dict': noise_model.to_dict(), 'coupling_map': coupling_map, 'tot_q_device': tot_q_device, 
                    'basis_gates': noise_backend.configuration().basis_gates}, open(backend_name+'_backend_config.pkl','wb'))
        print("retrieve architecture from IBM Q and save locally at: %s_backend_config.pkl\n"%(backend_name))
    assert tot_q_device >= n_tot_qubit
    new_noise_model = scale_noise_model(noise_model, sigma)

    # exclude qubit 0 as signal qubit, shift the remaining labels by -1
    be_map = [[q[0]-1,q[1]-1] for q in coupling_map if (0 not in q) and 
            (q[0] < n_tot_qubit) and (q[1] < n_tot_qubit)]
    be.build_random_circuit(n_depth, basis_gates=basis_gates, 
            prob_one_q_op=prob_one_q_op, coupling_map=be_map)
    be.build_dag()

    # load phase factors
    data = np.loadtxt("phi_inv_%d.txt"%(kappa))
    phi_seq = data[:-2]
    scale_fac = data[-2]
    app_err = data[-1]

    # retrieve block-encoded matrix
    UA = retrieve_unitary_matrix(be.qc)
    A = UA[0:2**n_sys_qubit, 0:2**n_sys_qubit]
    (svd_U, svd_S, svd_VH) = la.svd(A)
    print("kappa=%d, sigma=%.2f, polynomial approximation error=%.3e"%(kappa, sigma, app_err))
    print("")
    print("Generic RACBEM")
    print("singular value (A) = \n", np.around(svd_S, decimals=3))

    # succ prob via measurement
    qsp.build_circuit(be.qc, be.qc_dag, phi_seq, realpart=True, measure=True)
    compiled_circ = qsp.qcircuit
    job = execute(compiled_circ, backend=backend,
            noise_model=new_noise_model, shots=n_shots)
    job_monitor(job)
    result = job.result()
    counts = result.get_counts(compiled_circ)
    # both the signal and the ancilla qubit for block-encoding needs to
    # be 0
    prob_meas = np.float(counts['00']) / n_shots
    # succ prob via noiseless simulator
    qsp.build_circuit(be.qc, be.qc_dag, phi_seq, realpart=True, measure=False)
    state = retrieve_state(qsp.qcircuit)
    x = state[0:2**n_sys_qubit]
    prob_qsp = la.norm(x)**2
    # exact succ prob
    svd_S_herm = (1-1.0/kappa)*svd_S**2+1.0/kappa
    A_herm_inv = svd_VH.transpose().conjugate() @ np.diag(1/svd_S_herm) @ svd_VH
    x_exact = A_herm_inv @ b / scale_fac
    prob_exact = la.norm(x_exact)**2
    print("succ prob (exact)     = ", prob_exact)
    print("succ prob (noiseless) = ", prob_qsp)
    print("succ prob (measure)   = ", prob_meas)
    print("")

    # instance of Hermitian Block-Encoding
    n_be_qubit = n_be_qubit + 1     # add one extra qubit as sig_qubit in quadratic QSVT
    be = Hermitian_BlockEncoding(n_be_qubit, n_sys_qubit)
    be.set_cndnum(kappa)
    be.build_random_circuit(n_depth, basis_gates=basis_gates, 
            prob_one_q_op=prob_one_q_op, coupling_map=be_map)
    be.build_dag()
    UA = retrieve_unitary_matrix(be.qc)
    A = UA[0:2**n_sys_qubit, 0:2**n_sys_qubit]
    UA = retrieve_unitary_matrix(be.qc_dag)
    A_dag = UA[0:2**n_sys_qubit, 0:2**n_sys_qubit]
    (svd_U, svd_S, svd_VH) = la.svd(A)
    print("Hermitian RACBEM")
    print("singular value (A) = \n", np.around(svd_S, decimals=3))
    print("condition number (A)  = %.3f"%(svd_S.max()/svd_S.min()))
    print("||A - A^\dagger||_2   = %.3e"%(la.norm(A - A_dag)))
