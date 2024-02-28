import netCDF4 as nc
from numpy.typing import NDArray
import pymbar
from pymbar import timeseries, MBAR
import logging
import numpy as np

logger = logging.getLogger("mace_fep")


class RepexAnalyser:
    def __init__(self, netcdf_file: str):
        self.storage = self.open_netcdf_file(netcdf_file)


    def open_netcdf_file(self, netcdf_file: str):
        return nc.Dataset(netcdf_file, "r")

    

    def compute_free_energy(self, energy_key: str = "u_kln", verbose: bool = False):
        # decorrelate the free energy data 
        u_kln = self.storage.variables[energy_key][:]
        u_kln = u_kln.transpose(1,2,0)[:, :,::2]
        n, k, l = u_kln.shape
        
        final_values = []
        final_nk = []
        for i in range(n):
            A_t = u_kln[i,i,:]
            t0, g, Neff_max = timeseries.detectEquilibration(A_t)
            print(f"Sampler state: {i}, t0: {t0}, g: {g}, Neff_max: {Neff_max}")
            A_t_equil = A_t[t0:]
            indices = timeseries.subsampleCorrelatedData(A_t_equil, g=g)
            # evaluate the i'th sampled state at all thermodynamic states
            print(indices)
            print(indices + t0)
            A_ln = u_kln[i,:,indices + t0]
            final_nk.append(len(indices))
            final_values.append(A_ln)

        subsampled_uln = np.concatenate(final_values, axis=0).transpose(1,0)

        mbar = MBAR(u_kn=subsampled_uln, N_k=final_nk,  verbose=verbose)
        G_ij = mbar.getFreeEnergyDifferences()

        # print(G_ij)
        # in some arbitrary units
        results = {"value": G_ij[0][0,-1],
        "std_error": G_ij[1][0,-1]}
        return results
