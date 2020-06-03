"""Module for RACBEM implemented using IBM's Qiskit.

This is an implementation of the RAndom Circuit Block Encoded Matrix
(RACBEM) and its Hermitian conjugate. It is then used to build a quantum
singular value circuit using the method of quantum singular value
transformation (QSVT).

Take a RAndom Circuit Block Encoded Matrix (RACBEM), this function uses
a quantum signal processing circuit to evaluate the matrix inverse,
using the method of quantum singular value transformation (QSVT). This
implements a (non-Hermitian) block-encoding of a Hermitian matrix
manually.

Be aware that Qiskit uses the column-major ordering of tensors (1st
qubit goes first), instead of the standard row-major ordering of tensors
(last qubit goes first). There is a global phase factor that could be missing.

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
from numpy import pi

from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister

from random_circuit import random_circuit
from qiskit import Aer
from qiskit import execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.circuit.exceptions import CircuitError
import scipy.linalg as la

from qutip import Qobj

def matprint(mat, fmt="g"):
    """Pretty print a numpy matrix."""

    col_maxes = [
        max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T
    ]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


def retrieve_unitary_matrix(qcircuit):
    """Retrieve the matrix block-encoded by qcircuit.
    
    Args: 
        qcircuit: qiskit circuit, n_tot_qubit 

    Returns:
        UA: n_tot_qubit matrix.
    """
    backend = Aer.get_backend('unitary_simulator')
    n_tot_qubit = qcircuit.n_qubits
    job = execute(qcircuit, backend)
    result = job.result()
    UA = result.get_unitary(qcircuit)
    # qiskit's column-major oder
    UA_Qobj = Qobj(UA, dims=[[2]*n_tot_qubit, [2]*n_tot_qubit])
    # python's row-major order
    UA_Qobj_rowmajor = UA_Qobj.permute(np.arange(0,n_tot_qubit)[::-1])
    UA = np.array(UA_Qobj_rowmajor)
    
    return UA


def retrieve_state(qcircuit):
    """Retrieve the state by qcircuit."""
    backend = Aer.get_backend('statevector_simulator')
    n_tot_qubit = qcircuit.n_qubits
    job = execute(qcircuit, backend)
    result = job.result()
    state = result.get_statevector(qcircuit)
    # qiskit's column-major oder
    state_Qobj = Qobj(state, dims=[[2]*n_tot_qubit, [1]])
    # python's row-major order
    state_Qobj_rowmajor = state_Qobj.permute(np.arange(0,n_tot_qubit)[::-1])
    state_rowmajor = np.array(state_Qobj_rowmajor)[:,0]
    
    return state_rowmajor


def scale_noise_model(noise_model, sigma):
    """Retrieve the noise model from qiskit backend and scale it.

    After retrieving the noise model from a backend, create a new
    noise model by scaling the probability.

    We assume that there are only two types of noises:
        
        'roerror': readout error.
            probabilities is of type np.array(np.array()). Each inner
            array is a probability distribution. We assume the one
            closer to 1.0 to be the 'correct' option. Example of
            scaling of the noise:

                prob = [0.90, 0.06, 0.04]

            then prob[0] is identified to be the correct one. then after
            scaling

                prob = [1-(1-0.90)*sigma, 0.06*sigma, 0.04*sigma]

        'qerror': standard quantum error. There are two possibilities:
            'kraus': a Kraus operator. It is already a quantum channel,
                and therefore 'probabilities' is always [1.0]. This type
                of error mode is discarded.

            not 'kraus': scale it the same way as the 'roerror'.
            
    Args:
        noise_model: noise_model obtained from a quantum backend, or a
            custom noise_model
        sigma : scaling factor for the noise. 0 <= sigma <= 1
            When sigma = 1.0, the result should be the same as the noise
                model retrieved from the backend without 'kraus'.
            When sigma = 0.0, the result should be noiseless

    Returns:
        noise_model_scaled: scaled noise model. 
    """
    # dump out the noise model
    noise_dict = noise_model.to_dict()

    new_noise_dict = {
            'errors': [],
            'x90_gates': [],
            }

    for ierr in range(0,len(noise_dict['errors'])):
        err = noise_dict['errors'][ierr]
        if err['type'] == 'roerror':
            assert err['operations'] == ['measure']
            probs = err['probabilities']
            for iprob in range(0, len(probs)):
                # Note that this directly modifies the value of probs
                prob = probs[iprob]
                imax = np.argmax(prob)
                for idx in range(0,len(prob)):
                    if idx != imax:
                        prob[idx] *= sigma
                prob[imax] = 1.0-(1.0-prob[imax])*sigma
            new_noise_dict['errors'].append(err)

        elif err['type'] == 'qerror': 
            prob = err['probabilities']
            if prob == [1.0]:
                # This is a Kraus opeartor.
                # https://qiskit.org/documentation/stubs/qiskit.quantum_info.Kraus.html
                assert True

            else:
                # other errors, the same treatment as roerror
                assert len(prob) > 1
                imax = np.argmax(prob)
                for idx in range(0,len(prob)):
                    if idx != imax:
                        prob[idx] *= sigma
                prob[imax] = 1.0-(1.0-prob[imax])*sigma
                new_noise_dict['errors'].append(err)

    new_noise_model = NoiseModel.from_dict(new_noise_dict)

    return new_noise_model

class BlockEncoding(object):
    """Implementation of a RACBEM via Qiskit."""

    def __init__(self, n_be_qubit, n_sys_qubit):
        self.n_be_qubit = n_be_qubit
        self.n_sys_qubit = n_sys_qubit
        self.n_tot_qubit = n_be_qubit + n_sys_qubit
        self.qregs = QuantumRegister(self.n_tot_qubit, 'q')

    def build_block_encoding(self):
        """This gives a demo of the block-encoding U_A."""
        if hasattr(self, 'qc'):
            self.qc.data.clear()
        qr = self.qregs
        self.qc = QuantumCircuit(self.qregs, name='UA  ')

        self.qc.h(qr[0])
        self.qc.x(qr[0])
        self.qc.cx(qr[0], qr[1])
        self.qc.x(qr[0])
        self.qc.cz(qr[0], qr[2])
        self.qc.ry(pi * 2. / 3., qr[0])

    def build_random_circuit(self, n_depth, coupling_map=None, basis_gates=None, prob_one_q_op=0.5):
        """Build a random circuit as the block-encoding U_A.
        Args:
            n_depth (int):                      the number of the layers of operations
            coupling_map(CouplingMap or list):  coupling map specifies allowed CNOTs
                default:                        fully connected graph coupling n_qubits
            basis_gates (list):                 the list of labels of basis gates used in construction
                default:                        available gates in gate_set
            prob_one_q_op (float):              the probability of selecting a one-qubit operation when two_q_op is allowed
                default:                        equal probability 0.5
        """
        if hasattr(self, 'qc'):
            self.qc.data.clear()
        self.qc = random_circuit(self.n_tot_qubit, n_depth,
                coupling_map=coupling_map, basis_gates=basis_gates, prob_one_q_op=prob_one_q_op)
        self.qc.name = 'UA  '

    def build_dag(self):
        """Build the circuit for U_A^{\dagger}."""
        if hasattr(self, 'qc_dag'):
            self.qc_dag.data.clear()
        self.qc_dag = self.qc.inverse()

    def build_measure_circuit(self):
        """Build a circuit that can be used to measure the success
        probability of block encoding"""
        qr = self.qregs
        cr = ClassicalRegister(self.n_be_qubit, 'c')
        qcircuit = QuantumCircuit(qr, cr)
        qcircuit.append(self.qc.to_instruction(), qr)

        # measurements
        qcircuit.measure(qr[0], cr[0])

        return qcircuit

class QSPCircuit(object):
    """Build a QSP circuit.

    Attributes:
        qcircuit: QSP circuit.
        qregs: quantum registers
        cregs: classical registers
    """
    def __init__(self, n_sig_qubit, n_be_qubit, n_sys_qubit):
        self.n_sig_qubit = n_sig_qubit
        self.n_be_qubit = n_be_qubit
        self.n_sys_qubit = n_sys_qubit
        self.n_tot_qubit = n_sig_qubit + n_be_qubit + n_sys_qubit
        self.qregs = QuantumRegister(self.n_tot_qubit, 'q')
        self.cregs = ClassicalRegister(self.n_sig_qubit + self.n_be_qubit, 'c')

        self.qcircuit = QuantumCircuit(self.qregs, self.cregs, name='QSP')

        # so far the code only works with one OR two block encoding qubit due
        # to the implementation of the multi-qubit Toffoli gate.
        assert n_be_qubit == 1 or n_be_qubit == 2
        # so far the code only works if the number of signal qubit is 1
        # (i.e. no LCU yet)
        assert n_sig_qubit == 1

    def build_control_rotation(self, phi):
        """Build the controlled rotation gate."""

        qr = QuantumRegister(self.n_sig_qubit + self.n_be_qubit)
        qc_crot = QuantumCircuit(qr, name='CR\n%.1f'%(phi))

        # Add the X gate to perform control-0.
        if self.n_be_qubit == 1:
            qc_crot.x(qr[1])
            qc_crot.cx(qr[1], qr[0])
            qc_crot.rz(phi * 2., qr[0])
            qc_crot.cx(qr[1], qr[0])
            qc_crot.x(qr[1])
        elif self.n_be_qubit == 2:
            qc_crot.x(qr[1])
            qc_crot.x(qr[2])
            qc_crot.ccx(qr[1], qr[2], qr[0])
            qc_crot.rz(phi * 2., qr[0])
            qc_crot.ccx(qr[1], qr[2], qr[0])
            qc_crot.x(qr[1])
            qc_crot.x(qr[2])          

        return qc_crot
    
    def build_circuit(self, be_qc, be_qc_dag, phi_seq, realpart=True, measure=False, init_prepare=None):
        """Build a QSP circuit.

        Build a circuit for quantum signal processing, given a
        block-encoding matrix. 

        Args:
            be_qc: quantum circuit for block encoding. Can be a compiled
                version
            be_qc_dag: quantum circuit for the inverse of block
                encoding. Can be a compiled version. 
            phi_seq: a sequence of phase factors defined in the QSP
                circuit.
            realpart: if True, returns the real part of the polynomial
                encoded by QSP. This is implemented without the need of
                an extra ancilla qubit.
            measure: if True, adds measurements on the sig_ and be_qubit
            init_prepare: if not None (default), qcircuit will start from
                a prepare circuit on system qubits. Multiple formats are 
                supported
                #. QuantumCircuit
                #. str: string of 0 and 1 specifying state wrt Z basis

        Returns:
            None. This function provides a circuit in self.qcircuit .
        """
        if hasattr(self, 'qcircuit'):
            self.qcircuit.data.clear()
        qr = self.qregs

        if init_prepare is not None:
            if isinstance(init_prepare, QuantumCircuit):
                assert init_prepare.n_qubits + 2 == self.n_tot_qubit
                self.qcircuit.append(init_prepare.to_instruction(), qr[2:])
            elif isinstance(init_prepare, str):
                assert len(init_prepare) + 2 == self.n_tot_qubit
                for bitpos in range(len(init_prepare)):
                    if init_prepare[bitpos] == '1':
                        self.qcircuit.x(qr[bitpos+2])
            else:
                raise CircuitError("only support QuantumCircuit or str object as init_prepare")
            self.qcircuit.barrier()

        dag = False
        if realpart:
            # Add Hadamard gate as prepare oracle
            self.qcircuit.h(qr[0])
        self.qcircuit.barrier()

        # The for loop starts from the last phase factor, starting from
        # UA instead of UA_dag
        qc_crot = self.build_control_rotation(phi_seq[-1])
        self.qcircuit.append(qc_crot.to_instruction(), qr[0:self.n_sig_qubit+self.n_be_qubit])
        self.qcircuit.barrier()
        for phi in reversed(phi_seq[:-1]):
            if not dag:
                self.qcircuit.append(be_qc.to_instruction(), qr[1:])
            else:
                self.qcircuit.append(be_qc_dag.to_instruction(), qr[1:])
            self.qcircuit.barrier()

            if realpart:
                # Add a Z gate before the control. 
                self.qcircuit.z(qr[0])

            qc_crot = self.build_control_rotation(phi)
            self.qcircuit.append(qc_crot.to_instruction(), qr[0:self.n_sig_qubit+self.n_be_qubit])
            self.qcircuit.barrier()

            dag = not dag

        if realpart:
            # Add Hadamard gate as prepare oracle
            self.qcircuit.h(qr[0])
        
        # neglecting the global phase for now

        # measurements. For statevector_simulation and
        # unitary_simulation, measure should be set to False
        if measure:
            cr = self.cregs
            self.qcircuit.measure(qr[0:2], cr[0:2])


class Hermitian_BlockEncoding(BlockEncoding):
    """Implementation of a Hermitian-RACBEM via Qiskit.

    Attributes:
        n_be_qubit, n_sys_qubit -- partition of qubits
        cnd_num -- condition number
        ab -- a,b in h(x) = a*x^2+b
        shift -- constant when shifting spectrum A + c I
        phi_seq -- phi sequence used in quadratic QSVT
    """
    def __init__(self, n_be_qubit, n_sys_qubit):
        super(Hermitian_BlockEncoding, self).__init__(n_be_qubit, n_sys_qubit)
        self.qsp = QSPCircuit(1, n_be_qubit, n_sys_qubit)

    def set_ab(self, a, b):
        """Construct phi_seq according to h(x) = a*x^2+b"""
        self.ab = (a, b)
        phi0 = 0.25 * (np.arccos(a+b) + np.arccos(b))
        phi1 = 0.5 * (np.arccos(a+b) - np.arccos(b))
        self.phi_seq = np.array([phi0,phi1,phi0])

    def shift_spectrum(self, c):
        """ Shift the spectrum of the Hermitian Block-Encoding
        Let tilde A = h(A), where h(x) = ax^2+b and A is the block-encoding
        this function further shifts tilde A by a constant c
        set (a,b) to output a (1+|c|,2,0)-block-encoding of the
        Hermitian matrix tilde A + c I
        """
        assert hasattr(self, 'ab')
        a = self.ab[0] / (1.0+np.abs(c))
        b = (self.ab[1] + c) / (1.0+np.abs(c))
        self.set_ab(a, b)
        self.shift = c

    def set_cndnum(self, cndnum):
        """Upper bound the condition number of the Hermitian Block-Encoding"""
        self.cndnum = cndnum
        self.set_ab(1-1.0/cndnum, 1.0/cndnum)

    def build_random_circuit(self, n_depth, coupling_map=None, basis_gates=None, prob_one_q_op=0.5):
        """Build a random circuit as the Hermitian block-encoding as U_A.
        
        Args: refer to those in Block-Encoding::build_random_circuit()
        """
        assert hasattr(self, 'phi_seq')
        if hasattr(self, 'qc'):
            self.qc.data.clear()
        qc = random_circuit(self.n_tot_qubit, n_depth,
                coupling_map=coupling_map, basis_gates=basis_gates, prob_one_q_op=prob_one_q_op)
        qc.name = 'UA_0  '
        self.qsp.build_circuit(qc, qc.inverse(), self.phi_seq,
                realpart=True, measure=False)
        self.qc = self.qsp.qcircuit
        self.qc.name = 'UA  '
