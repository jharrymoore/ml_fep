import logging
import os
import time
import timeit
from typing import List
from ase import Atoms

import numpy as np
import netCDF4
from pydantic import BaseModel
from ase.io import read
from sympy import true
from mace_fep.replica_exchange.fep_calculator import (
    AbsoluteMACEFEPCalculator,
    FullCalcAbsoluteMACEFEPCalculator,
    FullCalcMACEFEPCalculator,
    MACEFEPCalculator,
)
# from mace_fep.utils import timeit
from ase.md.langevin import Langevin
from ase import units
from ase.optimize import LBFGS
from ase.constraints import FixAtoms
import mpiplus

from mace_fep.utils import with_timer


# TODO: I absolutely do not like that the atoms object is being quietly updated by ase under the hood, it is impossible to track what is happening here
# This is why we invented functional programming...
logger = logging.getLogger("mace_fep")
# set logging level


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
        )

        def write_frame():
            self.integrator.atoms.write(
                os.path.join(output_dir, f"output_{self.lmbda}.xyz"),
                append=True,
                parallel=False,
            )

        self.integrator.attach(write_frame, interval=100)

    def propagate(self, steps: int) -> None:
        self.integrator.run(steps)

    def minimise(self, tol=0.2):
        minimiser = LBFGS(self.atoms)
        minimiser.run(fmax=tol)


class ReplicaExchange:
    mace_model: str
    ligA_idx: List[int]
    ligB_idx: List[int]
    iters: int
    steps_per_iter: int
    _current_iter: int
    atoms: Atoms
    lmbdas: np.ndarray
    systems: List[System]
    energies_last_iteration: np.ndarray
    reporter: netCDF4.Dataset
    restart: bool

    def __init__(
        self,
        mace_model: str,
        output_dir: str,
        iters: int,
        steps_per_iter: int,
        xyz_file: str,
        replicas: int,
        ligA_idx: List[int],
        ligB_idx: List,
        restart: bool,
        minimise: bool = False,
    ) -> None:
        self.mace_model = mace_model
        self.ligA_idx = ligA_idx
        self.ligB_idx = ligB_idx
        self.iters = iters
        self.replicas = replicas
        self.output_dir = output_dir
        self.steps_per_iter = steps_per_iter
        self._current_iter = 0
        self._iter_time = 0.0
        # these will get updated at each iteration
        self.atoms = [read(xyz_file) for _ in range(replicas)]
        ligA_oxygen_idx = [i for i in ligA_idx if self.atoms[0][i].symbol == "O"][0]
        # find the oxygen atom in ligand B
        ligB_oxygen_idx = [i for i in ligB_idx if self.atoms[0][i].symbol == "N"][0]

        c = FixAtoms(indices=[ligA_oxygen_idx, ligB_oxygen_idx])
        for atoms in self.atoms:
            atoms.set_constraint(c)



        self.lmbdas = np.linspace(0, 1, replicas)
        logger.info("Initialising systems")
        self.systems = self._initialise_systems(
            self.lmbdas, self.atoms, self.output_dir
        )
        if restart:
            try:
                self._restart_from_latest()
            except OSError as e:
                logger.error(f"Could not restart from latest: {e}")
                self._initialise_storage_file(output_dir)

        # initialise the netcdf storage file
        else:
            self._initialise_storage_file(output_dir)

    def _initialise_systems(
        self, lambdas: np.ndarray, atoms: List[Atoms], output_dir: str
    ) -> List[System]:
        systems = []
        # attach a calculator to each atoms object, with a different lambda
        for idx, (lmbda, atoms) in enumerate(zip(lambdas, self.atoms)):
            atoms.set_calculator(
                FullCalcMACEFEPCalculator(
                    model_path=self.mace_model,
                    lmbda=lmbda,
                    stateA_idx=self.ligA_idx,
                    stateB_idx=self.ligB_idx,
                    device="cuda",
                )
            )
            # TODO: parametrise integrator kwargs
            systems.append(
                System(atoms=atoms, lmbda=lmbda, idx=idx, output_dir=output_dir)
            )
        return systems

    @mpiplus.on_single_node(0, broadcast_result=False, sync_nodes=False)
    def _restart_from_latest(self):
        # take the positions from the netcdf file, TODO: reload the lambda positions once the replica mixing works
        # open the netcdf file for reading

        self.reporter = netCDF4.Dataset(os.path.join(self.output_dir, "repex.nc"), "r+")
        # get the positions from the last iteration
        n_steps = self.reporter.dimensions["iteration"].size
        logger.info(f"Restarting from iteration {n_steps}")
        coords = self.reporter.variables["positions"][-1]
        # set the positions of the atoms objects
        for idx, replica in enumerate(self.systems):
            replica.atoms.set_positions(coords[idx])

        # close the netcdf file
        # self.reporter.close()

    def run(self) -> None:
        if self._current_iter == 0:
            # get the initial energies
            logger.info("Computing inintial energies")
            self.compute_energies()
            self.report_iteration()
        # main loop of replica exchange
        while self._current_iter < self.iters:
            logger.info(f"Running iteration {self._current_iter} of replica exchange")

            t1 = time.time()
            self.mix_replicas()
            t2 = time.time()

            t1 = time.time()
            self.propagate_replicas()
            t2 = time.time()
            logger.info(f"Propagated {self.replicas} replicas for {self.steps_per_iter} steps in {t2-t1:.4f} seconds")
            self.compute_energies()
            self._iter_time = time.time() - t1

            self.report_iteration()

            # TODO: the reporter update is not working with MPI at the moment

            # check for nan values in coords
            # print(self._current_iter)
            # print(self.reporter.variables["positions"].shape)
            # coords = self.reporter.variables["positions"][self._current_iter]
            self._current_iter += 1
            # if np.isnan(coords).any():
            #     logger.error("NaN values in coordinates, exiting")
            #     break

        # finally close the netcdf file
        self.reporter.close()

    def propagate_replicas(self) -> None:
        # run the mace FEP calculator for each replica, each on a separate MPI rank
        positions = mpiplus.distribute(self._propagate_replica, range(len(self.lmbdas)), send_results_to='all', sync_nodes=True)

        logger.debug(positions)

        # set the new positions
        for idx, replica in enumerate(self.systems):
            replica.atoms.set_positions(positions[idx])

        for idx in range(len(self.lmbdas)):
            logger.debug(f"system {idx} positions after propagation: {self.systems[idx].atoms.get_positions()[0]}")

    def _propagate_replica(self, idx) -> np.array:
        system = self.systems[idx]
        # print first row of positions before and after
        # logger.debug(f"system {idx} positions before propagation: {system.atoms.get_positions()[0]}")
        logger.debug(f"Propagating replica with lambda = {system.lmbda:.2f}")
        # this is all stateful, the atoms object has stuff updated in place.
        system.propagate(self.steps_per_iter)
        # how does the memory model
        logger.debug(f"system {idx} positions after propagation: {system.atoms.get_positions()[0]}")

        # return the positions
        return system.atoms.get_positions()

    @mpiplus.on_single_node(0, broadcast_result=True)
    def compute_energies(self) -> None:
        # extract energies from each replica - get this from the atoms object
        all_energies = np.zeros([len(self.systems), len(self.lmbdas)])
        logger.info("Computing energies for all replicas")

        for idx, replica in enumerate(self.systems):
            # this is a little complex, I think the easiest, to avoid getting and setting lots of np arrays, is to have the calculator do an energy evaluation on those position over the range of lambda values, that should be its own method since it should be super easy to do
            for jdx, lmbda in enumerate(self.lmbdas):
                replica.atoms.calc.set_lambda(lmbda)
                all_energies[idx, jdx] = replica.atoms.get_potential_energy()
                logger.debug(f"Energy of replica {idx:.2f} with lambda {lmbda:.2f} at iteration {self._current_iter} is {all_energies[idx, jdx]}")
            replica.atoms.calc.reset_lambda()
        logger.debug(all_energies)
        self.energies_last_iteration = all_energies

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
        for idx, replica in enumerate(self.systems):
            coords[idx] = replica.atoms.get_positions()
        self.reporter.variables["positions"][self._current_iter] = coords


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

    @mpiplus.on_single_node(0, broadcast_result=True)
    @with_timer("Mixing replicas")
    def mix_replicas(self) -> None:
        # pass the arrays to the rust module to perform the mixing
        # this is the complex bit: we attempt something like n^3 swaps for n replicas, 
        # TODO: this should be a compiled rust module - I don't want to introduce a numba dependency

        n_swap_attempts = self.replicas ** 3
        n_successful_swaps = 0


        for _ in range(n_swap_attempts):
            replica_i = np.random.randint(0, self.replicas)
            replica_j = np.random.randint(0, self.replicas)

            # attempt the swap
            result = self._attempt_swap(replica_i, replica_j)
            if result:
                n_successful_swaps += 1


        # get the mpi rank of the current node
        # rank = mpiplus.get_rank()

        logger.info(f" Attempted {n_swap_attempts} swaps, {n_successful_swaps} were successful")


    def _attempt_swap(self, replica_i: int, replica_j: int) -> bool:
        # get the thermodunamic state of the replica
        # get the energyes if ij, ji, ii and jj

        # extract the energies that have just been saved
        u_ij = self.energies_last_iteration[replica_i, replica_j]
        u_ji = self.energies_last_iteration[replica_j, replica_i]
        u_ii = self.energies_last_iteration[replica_i, replica_i]
        u_jj = self.energies_last_iteration[replica_j, replica_j]

        # logger.debug("u_ij:  {u_ij)
        # logger.debug("u_ji", u_ji)
        # logger.debug("u_ii", u_ii)
        # logger.debug("u_jj", u_jj)


        # logP = 0 if the replicas if we end up with the same i and j
        log_p = -(u_ij+ u_ji) + u_ii + u_jj
        # logger.debug(f"logP: {log_p}")
        # 
        if log_p > 0 or np.random.random() < np.exp(log_p):
            # swap the replicas
            lambda_i = self.systems[replica_i].lmbda
            lambda_j = self.systems[replica_j].lmbda
            self.systems[replica_i].lmbda = lambda_j
            self.systems[replica_j].lmbda = lambda_i
            return true
        else:
            return False


    def minimise(self):
        for system in self.systems:
            logger.info(f"Minimising replica with lambda = {system.lmbda}")
            system.minimise()
