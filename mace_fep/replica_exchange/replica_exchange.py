import logging
import os
import time
import timeit
from typing import List, Optional, Callable
from ase import Atoms
import torch

import numpy as np
import netCDF4
from pydantic import BaseModel
from ase.io import read
from pymbar import MBAR, timeseries
from pymbar.utils import ParameterError

from ase.md.langevin import Langevin
from ase import units
from ase.optimize import LBFGS
from ase.constraints import FixAtoms
import mpiplus
import yaml

from mace_fep.utils import with_timer

logger = logging.getLogger("mace_fep")


class System:
    def __init__(
        self,
        atoms: Atoms,
        lmbda: float,
        output_dir: str,
        idx: int,
        timestep: float = 1.0,
        temperature: float = 300.0,
        friction: float = 0.01,
    ) -> None:
        self.lmbda = lmbda
        # this will imutably tag the simulation with the lambda value that is it started with, regardless of where the lambda value ends up due to replica exchange
        self.idx = idx

        # TODO: is this a reference to some heap allocated atoms, such that this gets updated after we call .minimise() and .propagate()?
        self.atoms = atoms

        self.integrator = Langevin(
            self.atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature,
            friction=friction,
            # for some reason adding the following lines causes the step count and lambda schedule to stop working
            # logfile=os.path.join(output_dir, f"replica_{self.idx}.log"),
            # trajectory=os.path.join(output_dir, f"replica_{self.idx}.traj"),
        )

        def write_frame():
            self.integrator.atoms.write(
                os.path.join(output_dir, f"output_replica_{self.idx}.xyz"),
                append=True,
                parallel=False,
            )

        self.integrator.attach(write_frame, interval=1000)

    def propagate(self, steps: int) -> None:
        self.integrator.run(steps)

    def minimise(self, tol=0.1):
        minimiser = LBFGS(self.atoms)
        minimiser.run(fmax=tol)

class NonEqiulibriumSwitching:
    mace_model: str
    ligA_idx: List[int]
    iters: int
    atoms: Atoms
    init_lambda: float
    system: System

    def __init__(
        self,
        mace_model: str,
        ligA_idx: List[int],
        xyz_file: str,
        init_lambda: float,
        output_dir: str,
        steps_per_iter: int,
        fep_calc: Callable,
        equilibrate: bool,
        dtype: str,
    ):
        self.atoms = read(xyz_file)
        self.mace_model = mace_model
        self.ligA_idx = ligA_idx
        self.steps_per_iter = steps_per_iter

        self.init_lambda = init_lambda
        self.steps_per_iter = steps_per_iter
        self.output_dir = output_dir
        self.equilibrate = equilibrate
        self.dtype = dtype
        self._initialize_system(init_lambda, fep_calc)

    def _initialize_system(self, init_lambda, fep_calc: Callable):
        # we just have the one Atoms object, and we propagate the lambda value with the trajectory
        if self.equilibrate:
            delta_lambda = 0.0
        else:
            delta_lambda = 1 / self.steps_per_iter if init_lambda == 0.0 else -1 / self.steps_per_iter

        logger.info(f"delta_lambda: {delta_lambda}")
        self.atoms.set_calculator(
            fep_calc(
                model_path= self.mace_model,
                lmbda=init_lambda,
                device="cuda",
                delta_lambda = delta_lambda,
                stateA_idx=self.ligA_idx,
                default_dtype=self.dtype
            )
        )

        self.systems = [System(atoms=self.atoms, lmbda=init_lambda, idx=0, output_dir=self.output_dir)]

    def run(self):
        logger.debug(f"propagating replica for {self.steps_per_iter} steos")
        self.systems[0].propagate(self.steps_per_iter)

    def minimise(self):
        logger.debug("Minimising system")
        self.systems[0].minimise()




class ReplicaExchange:
    mace_model: str
    ligA_idx: List[int]
    ligB_idx: Optional[List[int]]
    iters: int
    steps_per_iter: int
    _current_iter: int
    atoms: Atoms
    lmbdas: np.ndarray
    systems: List[System]
    energies_last_iteration: np.ndarray
    reporter: netCDF4.Dataset
    restart: bool
    no_mixing: bool = False

    def __init__(
        self,
        mace_model: str,
        output_dir: str,
        iters: int,
        steps_per_iter: int,
        xyz_file: str,
        replicas: int,
        ligA_idx: List[int],
        ligB_idx: List[int],
        constrain_atoms_idx: Optional[List[int]],
        # ligA_const: int,
        # ligB_const: int,
        restart: bool,
        fep_calc: Callable,
        dtype: torch.dtype,
        no_mixing: bool
    ) -> None:
        self.mace_model = mace_model
        self.ligA_idx = ligA_idx
        self.ligB_idx = ligB_idx
        self.constrain_atoms_idx = constrain_atoms_idx
        self.iters = iters
        self.replicas = replicas
        self.output_dir = output_dir
        self.steps_per_iter = steps_per_iter
        self._current_iter = 0
        self._iter_time = 0.0
        self.fep_calc = fep_calc
        self.dtype = dtype
        # these will get updated at each iteration
        self.atoms = [read(xyz_file) for _ in range(replicas)]
        self.no_mixing = no_mixing
        # find the oxygen atom in ligand B

        if constrain_atoms_idx is not None: 
            c = FixAtoms(indices=constrain_atoms_idx)
            for atoms in self.atoms:
                atoms.set_constraint(c)

        self.lmbdas = np.linspace(0, 1, replicas)
        logger.info("Initialising systems")
        self.systems = self._initialise_systems(
            self.lmbdas, self.atoms, self.output_dir, self.fep_calc
        )
        if restart and os.path.exists(os.path.join(output_dir, "repex.nc")):
            try:
                (energies_last_iteration, n_steps, coords, velocities, lambda_vals) =self._restart_from_latest()
                self.energies_last_iteration = energies_last_iteration
                self._current_iter = n_steps
                for idx, replica in enumerate(self.systems):
                    replica.atoms.set_positions(coords[idx])
                    replica.atoms.set_velocities(velocities[idx])
                    logger.debug(f"Setting replica {idx} lambda to {lambda_vals[idx]}")
                    replica.atoms.calc.set_lambda(lambda_vals[idx])
            except OSError as e:
                logger.error(f"Could not restart from latest: {e}")
                self._initialise_storage_file(output_dir)

        # initialise the netcdf storage file
        else:
            self._initialise_storage_file(output_dir)

    def _initialise_systems(
        self, lambdas: np.ndarray, atoms: List[Atoms], output_dir: str, fep_calc: Callable
    ) -> List[System]:
        systems = []
        # attach a calculator to each atoms object, with a different lambda
        for idx, (lmbda, atoms) in enumerate(zip(lambdas, self.atoms)):
            atoms.set_calculator(
                fep_calc(
                    model_path=self.mace_model,
                    lmbda=lmbda,
                    stateA_idx=self.ligA_idx,
                    stateB_idx=self.ligB_idx,
                    device="cuda",
                    default_dtype=self.dtype
                )
            )
            # TODO: parametrise integrator kwargs
            systems.append(
                System(atoms=atoms, lmbda=lmbda, idx=idx, output_dir=output_dir)
            )
        return systems

    @mpiplus.on_single_node(0, broadcast_result=True, sync_nodes=True)
    def _restart_from_latest(self):
        # take the positions from the netcdf file, TODO: reload the lambda positions once the replica mixing works
        # open the netcdf file for reading
        self.reporter = netCDF4.Dataset(os.path.join(self.output_dir, "repex.nc"), "r+")
        # get the positions from the last iteration
        n_steps = self.reporter.dimensions["iteration"].size
        logger.info(f"Found last iteration: {n_steps}")
        # this is only propagated to the root node, we need to broadcast this to all nodes
		
        self._current_iter = n_steps
        coords = self.reporter.variables["positions"][-1]
        velocities = self.reporter.variables["velocities"][-1]
        lambda_vals = self.reporter.variables["lambda_vals"][-1]

        # set the energies last iteration
        energies_last_iteration = self.reporter.variables["u_kln"][-1]

        return ( energies_last_iteration, n_steps, coords, velocities, lambda_vals)

    def reformat_energies_for_mbar(self, u_kln: np.ndarray, n_k: Optional[np.ndarray] = None):
        """
        Convert [replica, state, iteration] data into [state, total_iteration] data

        This method assumes that the first dimension are all samplers,
        the second dimension are all the thermodynamic states energies were evaluated at
        and an equal number of samples were drawn from each k'th sampler, UNLESS n_k is specified.

        Parameters
        ----------
        u_kln : np.ndarray of shape (K,L,N')
            K = number of replica samplers
            L = number of thermodynamic states,
            N' = number of iterations from state k
        n_k : np.ndarray of shape K or None
            Number of samples each _SAMPLER_ (k) has drawn
            This allows you to have trailing entries on a given kth row in the n'th (n prime) index
            which do not contribute to the conversion.

            If this is None, assumes ALL samplers have the same number of samples
            such that N_k = N' for all k

            **WARNING**: N_k is number of samples the SAMPLER drew in total,
            NOT how many samples were drawn from each thermodynamic state L.
            This method knows nothing of how many samples were drawn from each state.

        Returns
        -------
        u_ln : np.ndarray of shape (L, N)
            Reduced, non-sparse data format
            L = number of thermodynamic states
            N = \sum_k N_k. note this is not N'
        """
        k, l, n = u_kln.shape
        if n_k is None:
            n_k = np.ones(k, dtype=np.int32) * n
        u_ln = np.zeros([l, n_k.sum()])
        n_counter = 0
        for k_index in range(k):
            u_ln[:, n_counter : n_counter + n_k[k_index]] = u_kln[
                k_index, :, : n_k[k_index]
            ]
            n_counter += n_k[k_index]
        return u_ln

    @mpiplus.on_single_node(0, broadcast_result=False)
    def online_analysis(self):
        # use MBAR to attempt to compute free energies at this iteration, write to stdout
        # write the mbar anaysis info the a yaml file, same as openff
        # this function is inspired by the equivalent function from the openmmtools MultiStateSampler
        try:  # Trap errors for MBAR being under sampled and the W_nk matrix not being normalized correctly
            # get the latest u_kln from the netcdf file [n_replicas, n_states]
            u_kln = self.reporter.variables["u_kln"][:].transpose(1,2,0)
            k, l, n = u_kln.shape


            logger.debug(u_kln)

            u_kn = self.reformat_energies_for_mbar(u_kln)


            mbar = MBAR(
                u_kn=u_kn,
                N_k=[n for _ in range(k)],
                verbose=False,
            )
            free_energy, err_free_energy = mbar.getFreeEnergyDifferences()
            logger.debug(f"Free energy: {free_energy} KT at iteration {self._current_iter}")
            # in some arbitrary units
            free_energy = free_energy[0, -1]
            err_free_energy = err_free_energy[0, -1]

            # TODO: get the timeseries, ideally we want to find the max equilibration time out of all sampled states, this just takes from the zero state and assumes that is representative
            (
                t0,
                statistical_inefficiency,
                n_uncorr_samples,
            ) = timeseries.detectEquilibration(u_kln[0, 0, :])
            data_dict = {
                "iteration": self._current_iter,
                "percent_complete": self._current_iter * 100 / self.iters,
                "mbar_analysis": {
                    "free_energy_in_eV": float(free_energy),
                    "standard_error_in_eV": float(err_free_energy),
                    "number_of_uncorrelated_samples": float(n_uncorr_samples),
                    # "n_equilibrium_iterations": int(n_equilibration_iterations),
                    "statistical_inefficiency": float(statistical_inefficiency),
                },
            }

            # append this to yaml file in the output
            with open(os.path.join(self.output_dir, "analysis.yaml"), "a") as f:
                yaml.dump(data_dict, f)

            # get the statistical inefficiency
        except (ParameterError, IndexError) as e:
            logger.warning(f"MBAR analysis failed: {e}")

    def run(self) -> None:
        logger.debug("Current iter is {}".format(self._current_iter))
        if self._current_iter == 0:
            # get the initial energies
            logger.info("Computing inintial energies")
            energies = self.compute_energies()
            self.energies_last_iteration = energies
            self.report_iteration()
        # main loop of replica exchange
        while self._current_iter < self.iters:
            logger.info(f"Running iteration {self._current_iter} of replica exchange")

            t1 = time.time()
            logger.debug("Mixing replicas")
            self.mix_replicas(no_mixing=self.no_mixing)
            t2 = time.time()

            t1 = time.time()
            logger.debug("Propagating replicas")
            self.propagate_replicas()
            t2 = time.time()
            logger.info(
                f"Propagated {self.replicas} replicas for {self.steps_per_iter} steps in {t2-t1:.4f} seconds"
            )
            logger.debug("Computing energies")
            self.energies_last_iteration = self.compute_energies()

            self._iter_time = time.time() - t1
    
            logger.debug("Reporting iteration")
            self.report_iteration()

            logger.debug("Performing online analysis")
            self.online_analysis()

            # check for nan values in coords
            # print(self._current_iter)
            # print(self.reporter.variables["positions"].shape)
            # coords = self.reporter.variables["positions"][self._current_iter]
            self._current_iter += 1
            logger.debug(f"Current iter now {self._current_iter}")
            # if np.isnan(coords).any():
            #     logger.error("NaN values in coordinates, exiting")
            #     break

        # finally close the netcdf file
        self.reporter.close()

    def propagate_replicas(self) -> None:
        # run the mace FEP calculator for each replica, each on a separate MPI rank
        positions = mpiplus.distribute(
            self._propagate_replica, range(len(self.lmbdas)), send_results_to="all", sync_nodes=True
        )


        # set the new positions
        for idx, replica in enumerate(self.systems):
            replica.atoms.set_positions(positions[idx])


    def _propagate_replica(self, idx) -> np.ndarray:
        logger.debug("propagating replica")
        system = self.systems[idx]
        logger.debug(f"Propagating replica with lambda = {system.lmbda:.2f}")
        # print first row of positions before and after
        # this is all stateful, the atoms object has stuff updated in place.
        system.propagate(self.steps_per_iter)
        return system.atoms.get_positions()

    @mpiplus.on_single_node(0, broadcast_result=True)
    def compute_energies(self) -> np.ndarray:
        # extract energies from each replica - get this from the atoms object
        all_energies = np.zeros([len(self.systems), len(self.lmbdas)])
        logger.info("Computing energies for all replicas")

        for idx, replica in enumerate(self.systems):
            # this is a little complex, I think the easiest, to avoid getting and setting lots of np arrays, is to have the calculator do an energy evaluation on those position over the range of lambda values, that should be its own method since it should be super easy to do
            for jdx, lmbda in enumerate(self.lmbdas):
                replica.atoms.calc.set_lambda(lmbda)
                all_energies[idx, jdx] = replica.atoms.get_potential_energy()
                logger.debug(
                    f"Energy of replica {idx:.2f} with lambda {lmbda:.2f} at iteration {self._current_iter} is {all_energies[idx, jdx]}"
                )
            replica.atoms.calc.reset_lambda()
        return all_energies

    @mpiplus.on_single_node(0, broadcast_result=False, sync_nodes=False)
    @mpiplus.delayed_termination
    def report_iteration(self) -> None:
        # write the arrays to the prepared netcdf file
        logger.debug("Writing iteration {} to file".format(self._current_iter))
        self.reporter.variables["u_kln"][
            self._current_iter
        ] = self.energies_last_iteration
        self.reporter.variables["iter_time"][self._current_iter] = self._iter_time

        # strip the coordinates out of the atoms object
        # coords of shape [system_idx, atoms, 3]
        coords = np.zeros([len(self.systems), len(self.atoms[0]), 3])
        velocities = np.zeros_like(coords)
        lambda_vals = np.zeros_like(self.lmbdas)
        for idx, replica in enumerate(self.systems):
            coords[idx] = replica.atoms.get_positions()
            velocities[idx] = replica.atoms.get_velocities()
            lambda_vals[idx] = replica.lmbda
        self.reporter.variables["positions"][self._current_iter] = coords


        self.reporter.variables["velocities"][self._current_iter] = velocities
        self.reporter.variables["lambda_vals"][self._current_iter] = lambda_vals

        self.reporter.sync()

    # do the IO from the root node, otherwise we're going to do a lot of overwriting
    @mpiplus.on_single_node(0, broadcast_result=False, sync_nodes=False)
    def _initialise_storage_file(self, output_dir: str) -> netCDF4.Dataset:
        # create a netcdf file to store the arrays

        # create the netcdf file
        logger.info(f"Initialising netcdf file in {output_dir}")
        self.reporter = netCDF4.Dataset(os.path.join(output_dir, "repex.nc"), "w")

        # create the dimensions
        self.reporter.createDimension("iteration", None)
        self.reporter.createDimension("lambda", len(self.lmbdas))
        self.reporter.createDimension("replica", len(self.systems))
        self.reporter.createDimension("iter_time", None)
        self.reporter.createDimension("n_atoms", len(self.atoms[0]))
        self.reporter.createDimension("n_dims", 3)

        self.reporter.createVariable("u_kln", "f8", ("iteration", "replica", "lambda"))
        self.reporter.createVariable("iter_time", "f8", ("iteration",))
        self.reporter.createVariable(
            "positions", "f8", ("iteration", "replica", "n_atoms", "n_dims")
        )
        self.reporter.createVariable("velocities", "f8", ("iteration", "replica", "n_atoms", "n_dims"))
        self.reporter.createVariable("lambda_vals", "f8", ("iteration", "replica"))

    @mpiplus.on_single_node(0, broadcast_result=True)
    @with_timer("Mixing replicas")
    def mix_replicas(self, no_mixing: bool = False) -> None:
        # pass the arrays to the rust module to perform the mixing
        # this is the complex bit: we attempt something like n^3 swaps for n replicas,
        iter_swap_info = np.zeros([self.replicas, self.replicas])
        if no_mixing:
            logger.info("No replica mixing performed")
            return

        n_swap_attempts = self.replicas**3
        n_successful_swaps = 0

        for _ in range(n_swap_attempts):
            # select two replicas as random to attempt to swap
            replica_i = np.random.randint(0, self.replicas)
            replica_j = np.random.randint(0, self.replicas)

            # attempt the swap
            result = self._attempt_swap(replica_i, replica_j)
            if result:
                n_successful_swaps += 1
                # record the new lambda value of the replica
                iter_swap_info[replica_i, replica_j] = self.systems[replica_i].lmbda

        logger.info(
            f" Attempted {n_swap_attempts} swaps, {n_successful_swaps} were successful"
        )

    def _attempt_swap(self, replica_i: int, replica_j: int) -> bool:
        # get the thermodunamic state of the replica
        # get the energyes if ij, ji, ii and jj

        # extract the energies that have just been saved
        u_ij = self.energies_last_iteration[replica_i, replica_j]
        u_ji = self.energies_last_iteration[replica_j, replica_i]
        u_ii = self.energies_last_iteration[replica_i, replica_i]
        u_jj = self.energies_last_iteration[replica_j, replica_j]

        print("u_ij",  u_ij)
        print("u_ji", u_ji)
        print("u_ii", u_ii)
        print("u_jj", u_jj)

        # logP = 0 if the replicas if we end up with the same i and j
        log_p = -(u_ij + u_ji) + u_ii + u_jj
        # logger.debug(f"logP: {log_p}")
        #
        if log_p > 0 or np.random.random() < np.exp(log_p):
            # swap the replicas
            lambda_i = self.systems[replica_i].atoms.calc.get_lambda()
            lambda_j = self.systems[replica_j].atoms.calc.get_lambda()
            self.systems[replica_i].atoms.calc.set_lambda(lambda_j)
            self.systems[replica_j].atoms.calc.set_lambda(lambda_i)
            return True
        else:
            return False
        
    def _minimise_system(self, system_idx):
        system = self.systems[system_idx]
        logger.info(f"Minimising replica with lambda = {system.lmbda:.2f}")
        system.minimise()
        # return positions
        return system.atoms.get_positions()

    def minimise(self,):
        min_positions = mpiplus.distribute(self._minimise_system, range(len(self.lmbdas)), send_results_to="all")
        # set the new positions
        for system in self.systems:
            system.atoms.set_positions(min_positions[system.idx])

        # write out the minimised positions
        for system in self.systems:
            system.atoms.write(
                os.path.join(self.output_dir, f"minimised_replica_{system.idx}.xyz"),
                append=True,
                parallel=False,
            )
