"""
Grover's Search Benchmark Program - CUDA-Q
"""

import time

import cudaq
import numpy as np

np.random.seed(0)


############### Circuit Definition

def GroversSearch(num_qubits, marked_item, n_iterations):

    marked_item_bits = [int(i) for i in format(marked_item, f"0{num_qubits}b")[::-1]]

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(num_qubits)
        # start with Hadamard on all qubits
        h(q)

        # loop over the estimated number of iterations
        for _ in range(n_iterations):

            # add the oracle
            grover_oracle(q, marked_item_bits)

            # add the diffusion operator
            diffusion_operator(q)

        # measure all qubits
        mz(q)


    kernel.compile()
    # return a handle on the circuit
    return kernel

############## Grover Oracle

@cudaq.kernel
def grover_oracle(qubits: cudaq.qview, marked_item: list[int]):
    for i, bit in enumerate(marked_item):
        if not bit:
            x(qubits[i])
    cz(qubits[0:-1], qubits[-1])
    for i, bit in enumerate(marked_item):
        if not bit:
            x(qubits[i])

############## Grover Diffusion Operator

@cudaq.kernel
def diffusion_operator(qubits: cudaq.qview):
    h(qubits)
    x(qubits)
    cz(qubits[0:-1], qubits[-1])
    x(qubits)
    h(qubits)

