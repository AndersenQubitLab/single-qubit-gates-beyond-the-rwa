import os
import multiprocessing
import json
from glob import glob
from datetime import datetime as dt
from itertools import repeat
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


NumberTypes = (int, float, complex, np.float64, np.complex64)
sigmax_np = np.array([[0,1],
                      [1,0]], dtype=complex)
sigmay_np = np.array([[0,-1j],
                      [1j,0]], dtype=complex)
sigmaz_np = np.array([[1,0],
                      [0,-1]], dtype=complex)
# Define color scheme
colors = {
    "red": "#ff4500",
    "green": "#00cc99",
    "yellow": "#FED100",
    "black": "#333333",
    "purple": "#836fff",
    "blue": "#127CB3"
}


class Hamiltonian:
    """
    Defines a Hamiltonian for one qubit
    """

    def __init__(self, 
                 RI: float = 0, 
                 Rx: float = 0, 
                 Ry: float = 0, 
                 Rz: float = 0) -> None:
        self.RI = RI
        self.Rx = Rx
        self.Ry = Ry
        self.Rz = Rz


    def __mul__(self, other) -> 'Hamiltonian':
        if isinstance(other, NumberTypes):
            return Hamiltonian(RI=other*self.RI,
                               Rx=other*self.Rx,
                               Ry=other*self.Ry,
                               Rz=other*self.Rz)
        elif isinstance(other, Hamiltonian):
            RI = self.Rx*other.Rx + self.Ry*other.Ry + self.Rz*other.Rz + self.RI*other.RI
            Rx = 1j*(self.Ry*other.Rz - other.Ry*self.Rz) + self.Rx*other.RI + other.Rx*self.RI
            Ry = 1j*(self.Rz*other.Rx - other.Rz*self.Rx) + self.Ry*other.RI + other.Ry*self.RI
            Rz = 1j*(self.Rx*other.Ry - other.Rx*self.Ry) + self.Rz*other.RI + other.Rz*self.RI
            return Hamiltonian(RI=RI, Rx=Rx, Ry=Ry, Rz=Rz)
        else:
            return NotImplemented


    def __rmul__(self, other) -> 'Hamiltonian':
        if isinstance(other, NumberTypes):
            return self.__mul__(other)
        else:
            return NotImplemented
     

    def __add__(self, other) -> 'Hamiltonian':
        if not isinstance(other, Hamiltonian):
            return NotImplemented
        
        return Hamiltonian(RI=self.RI + other.RI,
                           Rx=self.Rx + other.Rx,
                           Ry=self.Ry + other.Ry,
                           Rz=self.Rz + other.Rz)
    

    def __sub__(self, other) -> 'Hamiltonian':
        if not isinstance(other, Hamiltonian):
            return NotImplemented
        
        return Hamiltonian(RI=self.RI - other.RI,
                           Rx=self.Rx - other.Rx,
                           Ry=self.Ry - other.Ry,
                           Rz=self.Rz - other.Rz)
    

    def __str__(self) -> str:
        return f"Rx={self.Rx}, Ry={self.Ry}, Rz={self.Rz}, RI={self.RI}"
    
    
    def H(self) -> np.ndarray:
        """
        Returns Hamiltonian in Qobj form
        """
        return self.RI * np.eye(2) + self.Rx * sigmax_np + self.Ry * sigmay_np + self.Rz * sigmaz_np


    def unitary(self) -> np.ndarray:
        """
        Returns unitary evolution corresponding to exp(-1j*H)
        """
        z = self.z()
        if z==0:
            U = np.eye(2)
        else:
            U = np.cos(z)*np.eye(2) - 1j/z * np.sin(z) * (self.Rx*sigmax_np + self.Ry*sigmay_np + self.Rz*sigmaz_np)

        return U
    

    def z(self) -> float:
        """Returns total rotation on Bloch sphere"""
        return np.sqrt(self.Rx**2 + self.Ry**2 + self.Rz**2)
    

def rx(theta: float):
    """Unitary time evolution for an X_\theta gate"""
    return np.cos(theta/2)*np.eye(2) - 1j*np.sin(theta/2)*sigmax_np


def ry(theta: float):
    """Unitary time evolution for a Y_\theta gate"""
    return np.cos(theta/2)*np.eye(2) - 1j*np.sin(theta/2)*sigmay_np


def apply_args_and_kwargs(fn, args, kwargs):
    """
    Helper function for multiprocessing
    """
    return fn(*args, **kwargs)


def starmap_with_kwargs(pool: multiprocessing.Pool, 
                        fn, 
                        args_iter, 
                        kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def minimize_global(objective,
                    x0,
                    bounds,
                    *args,
                    **kwargs):
    """
    Attempts to find a global minimum by trying different x0's
    """
    if isinstance(x0[0], list):
        x0s = x0
    else:
        x0s = [x0]

    _res = None
    _fun = np.inf
    for _x0 in x0s:
        res = minimize(objective,
                       x0=_x0,
                       bounds=bounds,
                       *args,
                       **kwargs)
        
        if res["fun"]<_fun:
            _fun = res["fun"]
            _res = res

    return _res


def save_figuredata(figure_data: dict,
                    folder_name: str,
                    json_fname: str = "figure_data"):
    """
    Saves figure data in an appropriate format.
    We'll just dump it into a JSON, but because numpy arrays
    aren't really 'JSON-dumpable' they are exported to a 
    csv file first, and the corresponding entry in 'figuredata'
    is modified with a link to that file.
    """
    # Gen output dir
    _time = dt.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"{folder_name}/{_time}/"
    os.mkdir(output_dir)

    # Copy figure data into new dict
    json_data = figure_data.copy()

    # Loop over figures
    for fig_number, fig_data in figure_data.items():
        for k,v in fig_data.items():
            if isinstance(v, np.ndarray):
                fname = f"{output_dir}fig-{fig_number}-dataset-{k}.csv"
                np.savetxt(fname, v)
                json_data[fig_number][k] = fname

    # Dump data into file
    with open(f"{output_dir}{json_fname}.json", 'w') as f:
        json.dump(json_data, f)


def load_dataset(dataset: str,
                 folder_name: str,
                 json_fname: str = "figure_data"):
    """
    Loads data from dataset
    """
    input_dir = f"{folder_name}/{dataset}/"

    # Read figure data json file
    with open(f"{input_dir}{json_fname}.json", 'r') as f:
        json_data = json.load(f)

    # Do this little hacky thing for now because I saved the data
    # with incorrect keys 
    json_data = {k.replace("-","_"):v for k,v in json_data.items()}

    # Loop over CSV files and load data
    datafiles = glob(f"{input_dir}/*.csv")
    for datafile in datafiles:
        f = datafile.split("\\")[-1]
        f = f.split(".")[0]
        fig_number = f.split("-")[1]
        var_number = f.split("-")[3]

        try:
            # Load as real
            json_data[fig_number][var_number] = np.loadtxt(datafile)
        except: 
            # Load as complex
            json_data[fig_number][var_number] = np.loadtxt(datafile, dtype=np.complex_)

    # Transform keys back to ints
    try:
        json_data = {int(k):v for k,v in json_data.items()}
    except:
        pass

    return json_data


def fig_prepare(xlabel, ylabel, xscale="linear", yscale="linear"):
    """Prepare a matplotlib figure

    Parameters
    ----------
    xlabel : str 
        Label for x-axis
    ylabel : str 
        Label for y-axis
    xscale : str 
        Valid xscale label for matplotlib x-axis
    yscale : str
        Valid xscale label for matplotlib y-axis

    Returns
    ----------
    Matplotlib figure and axis
    """
    fig,ax = plt.subplots()
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(color="#A2A2A2", linestyle="--", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig,ax


def get_data_folders(base_dir: str, 
                     start: int, 
                     stop: int):
    """
    Finds a range of data folders based on their timestamp.

    Parameters
    ----------
    base_dir : str
        Directory to search in
    start : int
        Start time stamp
    stop : int
        Stop time stamp
    """
    all_data_folders = next(os.walk(base_dir))[1]
    folder_numbers = np.array([int(x.split("-")[1]) for x in all_data_folders])
    idx = np.argwhere((start<=folder_numbers)&(folder_numbers<=stop))
    data_folders = np.squeeze(np.array(all_data_folders)[idx], axis=1)
    data_folders = [base_dir+x for x in data_folders]

    return data_folders
