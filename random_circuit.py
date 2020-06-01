# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Generating random circuits according to the basis_gates and coupling_map.
This code is modified from random_circuit function provided in Qiskit.
This function generates random quantum circuit (RQC) faithfully according to 
the basis_gates and coupling_map given in the arguments.
The gates used in the construction are randomly selected from the intersection 
between given basis_gates and available gates (gate_set) in qiskit.extensions.
Only one/two-qubit gates are allowed for the consideration of near-term 
implementation.
Coupling_map specifies allowed (src, targ) pairs to perform CNOT.
In each layer, each qubit in thee register is subjected to a gate.

References:
    Yulong Dong, Lin Lin. Random circuit block-encoded matrix and a proposal 
        of quantum LINPACK benchmark.
Authors:
    Yulong Dong     dongyl (at) berkeley (dot) edu
    Lin Lin         linlin (at) math (dot) berkeley (dot) edu
Version: 1.0
Last revision: 03/2020
"""

import numpy as np

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Reset
from qiskit.extensions import (IdGate, U1Gate, U2Gate, U3Gate, XGate,
                               YGate, ZGate, HGate, SGate, SdgGate, TGate,
                               TdgGate, RXGate, RYGate, RZGate, CnotGate,
                               CyGate, CzGate, CHGate, CrzGate, Cu1Gate,
                               Cu3Gate, SwapGate, RZZGate,
                               ToffoliGate, FredkinGate)
from qiskit.circuit.exceptions import CircuitError
from qiskit.transpiler.coupling import CouplingMap

label2gate = {'id' : IdGate, 'u1' : U1Gate, 'u2' : U2Gate, 'u3' : U3Gate, 
        'x' : XGate, 'y' : YGate, 'z' : ZGate, 'h': HGate, 's' : SGate, 
        'sdg' : SdgGate, 't' : TGate, 'tdg' : TdgGate, 'cx' : CnotGate,
        'cz' : CzGate, 'cu1' : Cu1Gate, 'cu3' : Cu3Gate, 'swap' : SwapGate,
        'ccx' : ToffoliGate, 'cswap' : FredkinGate,
        'rx' : RXGate, 'ry' : RYGate, 'rz' : RZGate, 'cy' : CyGate, 'ch' : CHGate,
        'crz' : CrzGate, 'rzz' : RZZGate
}

# the classification of available gates provided in qiskit.extensions
gate_set = set(['id', 'u1', 'u2', 'u3', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg',
                'rx', 'ry', 'rz', 'cx'])

one_q_ops_label = set(['id', 'u1', 'u2', 'u3', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz'])
two_q_ops_label = set(['cx'])   # only includes CNOT
one_param_label = set(['u1', 'rx', 'ry', 'rz'])
two_param_label = set(['u2'])
three_param_label = set(['u3'])



def random_circuit(n_qubits, depth, coupling_map=None, basis_gates=None, prob_one_q_op=0.5, reset=False, seed=None):
    """Generate RQC faithfully according to the given coupling_map and basis_gates.

    Args:
        n_qubits (int):                     the number of qubits, must > largest qubit label in coupling_map
        depth (int):                        the number of the layers of operations
        coupling_map (CouplingMap or list): coupling map specifies allowed CNOTs
            default:                        fully connected graph coupling n_qubits
        basis_gates (list):                 the list of labels of basis gates used in construction
            default:                        all available gates in gate_set
        prob_one_q_op (float):              the probability of selecting a one-qubit operation when two_q_op is allowed
            default:                        equal probability 0.5
        reset (bool):                       if True, insert middle resets
        seed (int):                         sets random seed (optional)

    Returns:
        QuantumCircuit: constructed circuit

    Raises:
        CircuitError: when invalid options given
    """
    max_operands = 2
    assert max_operands == 2

    if isinstance(coupling_map,list):
        coupling_map = CouplingMap(coupling_map)
    if coupling_map != None and n_qubits < max(coupling_map.physical_qubits)+1:
        raise CircuitError("n_qubits is not enough to accomodate CouplingMap")

    if basis_gates == None:
        basis_gates = gate_set

    one_q_ops = [label2gate[name] for name in one_q_ops_label & set(basis_gates)]
    two_q_ops = [label2gate[name] for name in two_q_ops_label & set(basis_gates)]
    one_param = [label2gate[name] for name in one_param_label & set(basis_gates)]
    two_param = [label2gate[name] for name in two_param_label & set(basis_gates)]
    three_param = [label2gate[name] for name in three_param_label & set(basis_gates)]

    if len(one_q_ops) == 0:
        raise CircuitError("no available one-qubit gate")
    if len(two_q_ops) == 0:
        raise CircuitError("CNOT is not available")

    qreg = QuantumRegister(n_qubits, 'q')
    qc = QuantumCircuit(n_qubits)
    # default coupling_map is fully connected
    if coupling_map == None:
        coupling_map = CouplingMap.from_full(n_qubits)

    if reset:
        one_q_ops += [Reset]

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.RandomState(seed)

    for _ in range(depth):
        remaining_qubits = coupling_map.physical_qubits
        remaining_edges = coupling_map.get_edges()
        if remaining_edges:
            allow_two_q_op = True
        while remaining_qubits:
            if allow_two_q_op:
                max_possible_operands = min(len(remaining_qubits), max_operands)
            else:
                max_possible_operands = 1
            if max_possible_operands == 1:
                possible_operands_set = [1]
                num_operands = 1
            else:
                possible_operands_set = [1,2]
                num_operands = (not (rng.uniform() < prob_one_q_op)) + 1
            if num_operands == 1:
                operation = rng.choice(one_q_ops)
                operands = rng.choice(remaining_qubits)
                register_operands = [qreg[int(operands)]]
                operands = [operands]
            elif num_operands == 2:
                operation = rng.choice(two_q_ops)
                operands = remaining_edges[rng.choice(range(len(remaining_edges)))]
                register_operands = [qreg[i] for i in operands]
            remaining_qubits = [q for q in remaining_qubits if q not in operands]
            if remaining_edges:
                remaining_edges = [pair for pair in remaining_edges if pair[0] not in operands and pair[1] not in operands]
            if allow_two_q_op and not remaining_edges:
                allow_two_q_op = False
            if operation in one_param:
                num_angles = 1
            elif operation in two_param:
                num_angles = 2
            elif operation in three_param:
                num_angles = 3
            else:
                num_angles = 0
            angles = [rng.uniform(0, 2*np.pi) for x in range(num_angles)]
            
            op = operation(*angles)

            qc.append(op, register_operands)

    return qc