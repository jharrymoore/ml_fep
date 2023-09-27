import logging
import time
from typing import List
from ase import Atoms

import numpy as np
import netCDF4
from pydantic import BaseModel
from ase.io import read
from mace_fep.replica_exchange.fep_calculator import MACEFEPCalculator
from ase.md.langevin import Langevin
from ase import units
from ase.optimize import LBFGS
import mpiplus


# TODO: I absolutely do not like that the atoms object is being quietly updated by ase under the hood, it is impossible to track what is happening here
# This is why we invented functional programming...
logger = logging.getLogger("mace_fep")

class System:
    def __init__(
        self,
        atoms: Atoms,
        lmbda: float,
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
            self.atoms, timestep=timestep * units.fs, temperature_K=temperature, friction=friction
        )
        def write_frame():
            # logger.info("Writing frame, lambda = {} to file {}".format(lmbda, output_file))

            self.integrator.atoms.write(f"output_{self.lmbda}.xyz", append=True, parallel=False)

        self.integrator.attach(write_frame, interval=10)

    def propagate(self, steps: int) -> None:
        self.integrator.run(steps)

    def minimise(self, tol=1e-2):
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



    def __init__(
        self,
        mace_model: str,
        storage_file: str,
        iters: int,
        steps_per_iter: int,
        xyz_file: str,
        replicas: int,
        ligA_idx: List[int],
        ligB_idx: List,
        minimise: bool = False,
    ) -> None:
        # initi
        self.mace_model = mace_model
        self.ligA_idx = ligA_idx
        self.ligB_idx = ligB_idx
        self.iters = iters
        self.steps_per_iter = steps_per_iter
        self._current_iter = 0
        self._iter_time = 0.0
        # these will get updated at each iteration
        self.atoms = [read(xyz_file) for _ in range(replicas)]
        self.lmbdas = np.linspace(0, 1, replicas)
        logger.info("Initialising systems")
        self.systems = self._initialise_systems(self.lmbdas, self.atoms)
        if minimise:
            logger.info("Minimising replicas")
            self.minimise_replicas()

        # initialise the netcdf storage file
        self._initialise_storage_file(storage_file)

    def _initialise_systems(
        self, lambdas: np.ndarray, atoms: List[Atoms]
    ) -> List[System]:
        systems = []
        # attach a calculator to each atoms object, with a different lambda
        for idx, (lmbda, atoms) in enumerate(zip(lambdas, self.atoms)):
            atoms.set_calculator(
                MACEFEPCalculator(
                    model_path=self.mace_model,
                    lmbda=lmbda,
                    stateA_idx=self.ligA_idx,
                    stateB_idx=self.ligB_idx,
                    device="cuda",
                )
            )
            # now create the system for each atoms object
            # TODO: parametrise integrator kwargs
            systems.append(System(atoms=atoms, lmbda=lmbda, idx=idx))

        return systems

    def run(self) -> None:

        if self._current_iter == 0:
            # get the initial energies
            self.compute_energies()
            self.report_iteration()

        # main loop of replica exchange
        while self._current_iter < self.iters:
            t1 = time.time()
            self._current_iter += 1
            logger.info(
                f"Running iteration {self._current_iter} of replica exchange"
            )
            # note that at the moment this does nothing
            self.mix_replicas()

            self.propagate_replicas()

            self.compute_energies()
            self._iter_time = time.time() - t1

            self.report_iteration()


        # finally close the netcdf file
        self.reporter.close()

    def propagate_replicas(self) -> None:
        # run the mace FEP calculator for each replica, each on a separate MPI rank

        # in serial for now, althouth this function should distribute the replica computations over MPI ranks
        # for system in self.systems:
        #     logger.debug("Propagating replica with lambda = {}".format(system.lmbda))
        #     # this is all stateful, the atoms object has stuff updated in place.
        #     system.propagate(self.steps_per_iter)
        mpiplus.distribute(self._propagate_replica, range(len(self.lmbdas)) )
                           
    def _propagate_replica(self, idx) -> None:
        system = self.systems[idx]
        logger.debug("Propagating replica with lambda = {}".format(system.lmbda))
        # this is all stateful, the atoms object has stuff updated in place.
        system.propagate(self.steps_per_iter)


    def compute_energies(self) -> None:
        # extract energies from each replica - get this from the atoms object
        all_energies = np.zeros([len(self.systems), len(self.lmbdas)])
        logger.info("Computing energies for all replicas")

        for idx, replica in enumerate(self.systems):
            # this is a little complex, I think the easiest, to avoid getting and setting lots of np arrays, is to have the calculator do an energy evaluation on those position over the range of lambda values, that should be its own method since it should be super easy to do
            for jdx, lmbda in enumerate(self.lmbdas):
                replica.atoms.calc.set_lambda(lmbda)
                all_energies[idx, jdx] = replica.atoms.get_potential_energy()

        self.energies_last_iteration = all_energies

    def report_iteration(self) -> None:
        # write the arrays to the prepared netcdf file
        logger.debug("Writing iteration {} to file".format(self._current_iter))
        self.reporter.variables["u_kln"][self._current_iter] = self.energies_last_iteration
        self.reporter.variables["iter_time"][self._current_iter] = self._iter_time
    

        # flush to disk
        self.reporter.sync()


    def _initialise_storage_file(self, storage_file: str) -> netCDF4.Dataset:
        # create a netcdf file to store the arrays

        # create the netcdf file
        self.reporter = netCDF4.Dataset(storage_file, "w")

        # create the dimensions
        self.reporter.createDimension("iteration", None)
        self.reporter.createDimension("lambda", len(self.lmbdas))
        self.reporter.createDimension("replica", len(self.systems))
        self.reporter.createDimension("iter_time", None)

        self.reporter.createVariable("u_kln", "f8", ("iteration", "replica", "lambda"))
        self.reporter.createVariable("iter_time", "f8", ("iteration",))


    def mix_replicas(self) -> None:
        # pass the arrays to the rust module to perform the mixing
        pass

    def minimise_replicas(self):
        for system in self.systems:
            logger.info(f"Minimising replica with lambda = {system.lmbda}")
            system.minimise()
