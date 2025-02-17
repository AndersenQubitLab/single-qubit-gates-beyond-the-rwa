import multiprocessing
from itertools import product, repeat
import numpy as np
import scqubits as scq
from sqgbrwa.time_evolutions import *
from sqgbrwa.experiment_simulations import *
from sqgbrwa.utils import *


def get_fluxonium_params():
    """
    Returns paramaters of fluxonium qubit
    """
    # Params FX5
    EJ = 4.92281
    EL = 0.50083
    EC = 0.88049

    # Define fluxonium
    fluxonium = scq.Fluxonium(EJ=EJ, 
                              EC=EC, 
                              EL=EL, 
                              flux=0.5, 
                              cutoff=30, 
                              truncated_dim=30)

    # Get energies
    E = fluxonium.eigenvals()
    w01 = (0.09897419)*1e9*2*np.pi
    w02 = (E[2]-E[0])*1e9*2*np.pi
    w03 = (E[3]-E[0])*1e9*2*np.pi

    # Get matrix elements
    eta12 = np.abs(fluxonium.matrixelement_table("n_operator")[1,2])/np.abs(fluxonium.matrixelement_table("n_operator")[0,1])
    eta03 = np.abs(fluxonium.matrixelement_table("n_operator")[0,3])/np.abs(fluxonium.matrixelement_table("n_operator")[0,1])
    eta23 = np.abs(fluxonium.matrixelement_table("n_operator")[2,3])/np.abs(fluxonium.matrixelement_table("n_operator")[0,1])

    mat_elems = np.array([[0,1,0,eta03],
                          [1,0,eta12,0],
                          [0,eta12,0,eta23],
                          [eta03,0,eta23,0]])
    
    return w01, w02, w03, mat_elems


def generate_time_evolution_vs_ppp_detuning():
    """
    Generates the time evolution operators vs the ppp, 
    detuning and carrier phase
    """
    # Load extended data
    extended_data = load_dataset("20241201-152543", folder_name="extended-data", json_fname="extended_data")

    # Get parameters of FX5
    w01, w02, w03, mat_elems = get_fluxonium_params()

    tgs = extended_data["drive_strength_calibration_1"]["x0"]
    V0s_pi2 = extended_data["drive_strength_calibration_1"]["y0"]
    V0s_pi = extended_data["drive_strength_calibration_1"]["y1"]
    wds = np.linspace(-5e6,5e6,41)*2*np.pi + w01

    # Option to add time-dependent drive frequency
    # omega_delta = -1*(
    #     mat_elems[0,3]**2/(w03-3*w01) -
    #     mat_elems[1,2]**2/(w02-2*w01)
    # )
    omega_delta = 0

    num_phases = 41
    num_ppps = 41

    # Define global phase
    global_phase = 0
    kwargs_iter = []

    ## pi gates
    iter_obj = list(product(zip(tgs,V0s_pi),wds))
    iter_obj = [(x[0][0],x[0][1],x[1]) for x in iter_obj]

    # Arguments that are the same for every experiment
    base_arguments = dict(
        hamiltonian=H_full_4_levels,
        num_phases=num_phases,
        num_ppps=num_ppps,
        num_levels=4,
        w01=w01,
        alpha2=w02-2*w01,
        alpha3=w03-3*w01,
        mat_elems=mat_elems,
        omega_delta=omega_delta,
        envelope_phase=global_phase
    )
    # Varying arguments
    kwargs_iter += [
        {**base_arguments,
        'tg': tg,
        'V0': V0,
        'wd': wd} for tg,V0,wd in iter_obj
    ]

    ## pi/2 gates
    iter_obj = list(product(zip(tgs,V0s_pi2),wds))
    iter_obj = [(x[0][0],x[0][1],x[1]) for x in iter_obj]

    # Arguments that are the same for every experiment
    base_arguments = dict(
        hamiltonian=H_full_4_levels,
        num_phases=num_phases,
        num_ppps=num_ppps,
        num_levels=4,
        w01=w01,
        alpha2=w02-2*w01,
        alpha3=w03-3*w01,
        mat_elems=mat_elems,
        omega_delta=omega_delta,
        envelope_phase=global_phase
    )
    # Varying arguments
    kwargs_iter += [
        {**base_arguments,
        'tg': tg,
        'V0': V0,
        'wd': wd} for tg,V0,wd in iter_obj
    ]

    ## mpi gates
    iter_obj = list(product(zip(tgs,V0s_pi),wds))
    iter_obj = [(x[0][0],x[0][1],x[1]) for x in iter_obj]

    # Arguments that are the same for every experiment
    base_arguments = dict(
        hamiltonian=H_full_4_levels,
        num_phases=num_phases,
        num_ppps=num_ppps,
        num_levels=4,
        w01=w01,
        alpha2=w02-2*w01,
        alpha3=w03-3*w01,
        mat_elems=mat_elems,
        omega_delta=omega_delta,
        envelope_phase=global_phase + np.pi
    )
    # Varying arguments
    kwargs_iter += [
        {**base_arguments,
        'tg': tg,
        'V0': V0,
        'wd': wd} for tg,V0,wd in iter_obj
    ]

    ## mpi2 gates
    iter_obj = list(product(zip(tgs,V0s_pi2),wds))
    iter_obj = [(x[0][0],x[0][1],x[1]) for x in iter_obj]

    base_arguments = dict(
        hamiltonian=H_full_4_levels,
        num_phases=num_phases,
        num_ppps=num_ppps,
        num_levels=4,
        w01=w01,
        alpha2=w02-2*w01,
        alpha3=w03-3*w01,
        mat_elems=mat_elems,
        omega_delta=omega_delta,
        envelope_phase=global_phase+np.pi
    )
    # Varying arguments
    kwargs_iter += [
        {**base_arguments,
        'tg': tg,
        'V0': V0,
        'wd': wd} for tg,V0,wd in iter_obj
    ]

    ## Initialize Multiprocessing pool and batch jobs
    args_iter = repeat([])
    pool = multiprocessing.Pool(processes=41)
    results = starmap_with_kwargs(pool, simulate_evolution_vs_carrier_phase, args_iter, kwargs_iter)
    pool.close()
    pool.join()

    ## Parse and save results
    n = len(tgs)*len(wds)
    results1 = results[0:n]
    results2 = results[n:2*n]
    results3 = results[2*n:3*n]
    results4 = results[3*n:]

    results_parsed_1 = dict()
    results_parsed_2 = dict()
    results_parsed_3 = dict()
    results_parsed_4 = dict()

    # Setpoints
    results_parsed_1["x0"] = tgs
    results_parsed_1["x1"] = V0s_pi
    results_parsed_1["x2"] = wds
    results_parsed_2["x0"] = tgs
    results_parsed_2["x1"] = V0s_pi2
    results_parsed_2["x2"] = wds
    results_parsed_3["x0"] = tgs
    results_parsed_3["x1"] = V0s_pi
    results_parsed_3["x2"] = wds
    results_parsed_4["x0"] = tgs
    results_parsed_4["x1"] = V0s_pi2
    results_parsed_4["x2"] = wds

    # Simulation results
    for i in range(len(tgs)*len(wds)):
        results_parsed_1[f"y{i}"] = results1[i].reshape((num_phases,num_ppps*4*4))
        results_parsed_2[f"y{i}"] = results2[i].reshape((num_phases,num_ppps*4*4))
        results_parsed_3[f"y{i}"] = results3[i].reshape((num_phases,num_ppps*4*4))
        results_parsed_4[f"y{i}"] = results4[i].reshape((num_phases,num_ppps*4*4))

    g = "x" if global_phase==0 else "y"
    extended_data[f"time_evolutions_1_{g}pi"] = results_parsed_1
    extended_data[f"time_evolutions_1_{g}pi2"] = results_parsed_2
    extended_data[f"time_evolutions_1_{g}mpi"] = results_parsed_3
    extended_data[f"time_evolutions_1_{g}mpi2"] = results_parsed_4

    save_figuredata(extended_data, folder_name="extended-data", json_fname="extended_data")


if __name__=="__main__":
    generate_time_evolution_vs_ppp_detuning()
