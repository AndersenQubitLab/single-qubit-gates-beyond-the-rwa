"""
Generating the symbolic analytical expressions for the integrals that 
are required to compute the ideal pulse parameters is a very time-consuming process.
Therefore, we generate them here and dump them using `pickle`. We actually use
`cloudpickle` because the lambdified functions returned by Sympy can't be pickled.

In theory, anyone but me has to run this. It might be possible that pickled objects
are not compatible between different python versions. In that case, you either 
have to run this yourself or install python 3.12.0. 
"""

from sqgbrwa.pulse_parameters import *
import cloudpickle

def generate_sympy_integrals_1():
    sympy_integrals = analytical_comm_sympy_functions(K=30)

    with open("sympy-integrals/sympy-integrals-1.pickle", "wb") as f:
        cloudpickle.dump(sympy_integrals, f)


def generate_sympy_integrals_2():
    sympy_integrals = analytical_detuning_sympy_functions(K=30)

    with open("sympy-integrals/sympy-integrals-2.pickle", "wb") as f:
        cloudpickle.dump(sympy_integrals, f)


def generate_sympy_integrals_3():
    sympy_integrals = analytical_detuning_comm_sympy_functions(K=15, processes=32)

    with open("sympy-integrals/sympy-integrals-3.pickle", "wb") as f:
        cloudpickle.dump(sympy_integrals, f)


if __name__=="__main__":
    generate_sympy_integrals_1()
    generate_sympy_integrals_2()
    generate_sympy_integrals_3()
