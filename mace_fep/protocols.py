import logging
import os
import time
from typing import Iterable, List, Optional, Callable
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import LBFGS
from ase.md.langevin import Langevin
import torch
from datetime import date
import datetime
import os
from ase import Atoms, units

import numpy as np
import netCDF4
from ase.io import read
from ase import Atoms
from pymbar import MBAR, timeseries
from pymbar.utils import ParameterError

from ase.constraints import FixAtoms
import mpiplus
import yaml
from mace_fep.replica import Replica

from mace_fep.utils import with_timer

logger = logging.getLogger("mace_fep")


# ioan's settings for organic systems - from ethanol water simulations
tstep   = 1.0*units.fs
ttime   = 50*units.fs
B_water = 2.0*units.GPa #vs 100*units.GPa recommended default
ptime   = 2500*units.fs

MD_dt = 100
TH_dt = 10
pres  = 1.013
densfact = (units.m/1.0e2)**3/units.mol



class NonEquilibriumSwitching:
    def __init__(
        self,
        atoms: Atoms,
        output_dir: str,
        total_steps: int,
        idx: int = 0,
        timestep: float = 1.0,
        temperature: float = 300.0,
        friction: float = 0.01,
        lbfgs_fmax: float = 0.2,
        integrator: str = "Langevin",
        report_interval: int = 100,
        constrain_atoms_idx: list = [],
        start_step: int = 0,
    ) -> None:
        self.total_steps = total_steps
        self.idx = idx
        self.restart = True if start_step > 0 else False
        self.header = '   Time(fs)     Latency(ns/day)    Temperature(K)         Lambda   Density(g/cm$^3$)        Energy(eV)        MSD(A$^2$)       COMSD(A$^2$)   Volume(cm$^3$)   Time remaining\n'

        self.atoms = atoms
        if constrain_atoms_idx is not None: 
            logger.info(f"Constraining atoms {constrain_atoms_idx}")
            c = FixAtoms(indices=constrain_atoms_idx)
            self.atoms.set_constraint(c)
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature)
        self.output_dir = output_dir
        start_config_com = atoms.get_center_of_mass().copy()
        start_config_positions = atoms.positions.copy()
        self.lbfgs_fmax = lbfgs_fmax
        self.report_interval = report_interval
        self.checkpoint_time = time.time()
        
        if integrator == "Langevin":
            logging.info("Setting up Langevin dynamics")
            self.integrator = Langevin(
                self.atoms,
                timestep=timestep * units.fs,
                temperature_K=temperature,
                friction=friction,
            )
        else:
            raise ValueError(f"Unknown integrator {integrator}")

        self.integrator.nsteps = start_step
        logging.info(f"Starting from step {start_step}")

        def write_frame():
            self.integrator.atoms.write(
                os.path.join(output_dir, f"output_replica_{self.idx}.xyz"),
                append=True,
                parallel=False,
            )

        def update_lambda():
            next(self.integrator.atoms.calc.lambda_schedule)

        def recompute_nl():
            self.integrator.atoms.calc.update_nl()

        def print_traj():
            current_time = time.time()
            time_elapsed = current_time - self.checkpoint_time
            # steps per second
            steps_per_day = (1/time_elapsed) * 86400
            ns_per_day = steps_per_day * timestep * 1e-6
            time_remaining_seconds = (self.total_steps - self.integrator.nsteps) / (steps_per_day / 86400)
            # format to days:hours:minutes:seconds
            time_remaining = str(datetime.timedelta(seconds=time_remaining_seconds))

            a = self.integrator.atoms
            calc_time = self.integrator.get_time()/units.fs
            calc_temp = a.get_temperature()
            calc_dens = np.sum(a.get_masses())/a.get_volume()*densfact
            # calc_pres = -np.trace(a.get_stress(include_ideal_gas=True, voigt=False))/3/units.bar if self.integrator.__class__.__name__ == 'NPT' else np.zeros(1)
            calc_epot = a.get_potential_energy()
            dhdl = a.get_potential_energy(force_consistent=True)
            calc_msd  = (((a.positions-a.get_center_of_mass())-(start_config_positions-start_config_com))**2).mean(0).sum(0)
            calc_drft = ((a.get_center_of_mass()-start_config_com)**2).sum(0)
            # calc_tens = -a.get_stress(include_ideal_gas=True, voigt=True)/units.bar if self.integrator.__class__.__name__ == 'NPT' else np.zeros(6)
            calc_volume = a.get_volume() * densfact
            current_lambda = float(self.integrator.atoms.calc.lambda_schedule.output_lambda)
            a.info['step'] = self.integrator.nsteps
            a.info['lambda'] = current_lambda
            a.info['time_fs'] = self.integrator.get_time()/units.fs
            a.info['time_ps'] = self.integrator.get_time()/units.fs/1000
            with open(os.path.join(output_dir, "thermo_traj.dat"), 'a') as thermo_traj:
                thermo_traj.write(('%12.1f'+' %17.6f'*8+'    %s'+'\n') % (calc_time, ns_per_day, calc_temp,current_lambda, calc_dens, calc_epot, calc_msd, calc_drft, calc_volume, time_remaining ))
                thermo_traj.flush()
            print(('%12.1f'+' %17.6f'*8+ '    %s') % (calc_time, ns_per_day, calc_temp, current_lambda, calc_dens, calc_epot, calc_msd, calc_drft, calc_volume, time_remaining), flush=True)
            with open(os.path.join(output_dir, "dhdl.xvg"), 'a') as xvg:
                time_ps = calc_time 
                xvg.write(f"{time_ps:.4f} {dhdl:.6f}\n")
            self.checkpoint_time = time.time()

        self.integrator.attach(write_frame, interval=self.report_interval)
        self.integrator.attach(print_traj, interval=1)
        self.integrator.attach(recompute_nl, interval=1)
        if self.integrator.atoms.calc.lambda_schedule.delta != 0:
            self.integrator.attach(update_lambda, interval=1)

    def propagate(self) -> None:
        if not self.restart:
            with open(os.path.join(self.output_dir, "thermo_traj.dat"), 'a') as thermo_traj:
                thermo_traj.write('# ASE Dynamics. Date: '+date.today().strftime("%d %b %Y")+'\n')
                thermo_traj.write(self.header)
                print('# ASE Dynamics. Date: '+date.today().strftime("%d %b %Y"))
            with open(os.path.join(self.output_dir, "dhdl.xvg"), 'a') as dhdl:
                dhdl.write('# Time (ps) dH/dL\n')
            
        print(self.header)
        self.integrator.run(self.total_steps)

    def minimise(self):
        minimiser = LBFGS(self.atoms)
        minimiser.run(fmax=self.lbfgs_fmax)


class ReplicaExchange:

    def __init__(
        self,
        output_dir: str,
        iters: int,
        steps_per_iter: int,
        replicas: List[Replica],
        restart: bool,
        dtype: torch.dtype,
        no_mixing: bool,
        constrain_atoms_idx: Optional[List[int]] = None,
        temperature: float=298.0
    ) -> None:
        self.iters=  iters
        self.replicas = replicas
        self.output_dir = output_dir
        self.steps_per_iter = steps_per_iter
        self._current_iter = 0
        self._iter_time = 0.0
        self.dtype = dtype
        self.no_mixing = no_mixing

        if constrain_atoms_idx is not None: 
            c = FixAtoms(indices=constrain_atoms_idx)
            for r in self.replicas:
                r.atoms.set_constraint(c)

        for r in self.replicas:
            MaxwellBoltzmannDistribution(r.atoms, temperature_K=temperature)

        if restart and os.path.exists(os.path.join(output_dir, "repex.nc")):
            try:
                (energies_last_iteration, n_steps, coords, velocities, lambda_vals) = self._restart_from_latest()
                self.energies_last_iteration = energies_last_iteration
                self._current_iter = n_steps
                for idx, replica in enumerate(self.replicas):
                    atoms=replica.atoms
                    atoms.set_positions(coords[idx])
                    atoms.set_velocities(velocities[idx])
                    logger.debug(f"Setting replica {idx} lambda to {lambda_vals[idx]}")
                    atoms.calc.set_lambda(lambda_vals[idx])
            except OSError as e:
                logger.error(f"Could not restart from latest: {e}")
                self._initialise_storage_file(output_dir)

        # initialise the netcdf storage file
        else:
            self._initialise_storage_file(output_dir)

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
            N = sum_k N_k. note this is not N'
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
        try:  # Trap errors for MBAR being under sampled and the W_nk matrix not being normalized correctly
            u_kln = self.reporter.variables["u_kln"][:].transpose(1,2,0)
            k, l, n = u_kln.shape
            u_kn = self.reformat_energies_for_mbar(u_kln)

            mbar = MBAR(
                u_kn=u_kn,
                N_k=[n for _ in range(k)],
                verbose=False,
            )
            free_energy, err_free_energy = mbar.getFreeEnergyDifferences()
            logger.debug(f"Free energy: {free_energy} eV at iteration {self._current_iter}")
            # in some arbitrary units
            free_energy = free_energy[0, -1]
            err_free_energy = err_free_energy[0, -1]

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

    def propagate(self) -> None:
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
            self._propagate_replica, range(len(self.replicas)), send_results_to="all", sync_nodes=True
        )


        # set the new positions
        for idx, replica in enumerate(self.replicas):
            replica.atoms.set_positions(positions[idx])


    def _propagate_replica(self, idx) -> np.ndarray:
        replica=self.replicas[idx]
        logger.debug(f"Propagating replica with lambda = {replica.l:.2f}")
        # print first row of positions before and after
        # this is all stateful, the atoms object has stuff updated in place.
        replica.propagate(self.steps_per_iter)
        return replica.atoms.get_positions()

    @mpiplus.on_single_node(0, broadcast_result=True)
    def compute_energies(self) -> np.ndarray:
        # extract energies from each replica - get this from the atoms object
        all_energies = np.zeros([len(self.replicas), len(self.replicas)])
        logger.info("Computing energies for all replicas")

        for idx, replica in enumerate(self.replicas):
            # this is a little complex, I think the easiest, to avoid getting and setting lots of np arrays, is to have the calculator do an energy evaluation on those position over the range of lambda values, that should be its own method since it should be super easy to do
            for jdx, lmbda in enumerate([r.l for r in self.replicas]):
                replica.atoms.calc.set_lambda(lmbda)
                all_energies[idx, jdx] = replica.atoms.get_potential_energy()
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
        coords = np.zeros([len(self.replicas), len(self.replicas[0].atoms), 3])
        velocities = np.zeros_like(coords)
        lambda_vals = np.zeros_like(self.replicas)
        for idx, replica in enumerate(self.replicas):
            coords[idx] = replica.atoms.get_positions()
            velocities[idx] = replica.atoms.get_velocities()
            lambda_vals[idx] = replica.l
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
        self.reporter.createDimension("lambda", len(self.replicas))
        self.reporter.createDimension("replica", len(self.replicas))
        self.reporter.createDimension("iter_time", None)
        self.reporter.createDimension("n_atoms", len(self.replicas[0].atoms))
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
        iter_swap_info = np.zeros([len(self.replicas), len(self.replicas)])
        if no_mixing:
            logger.info("No replica mixing performed")
            return

        n_swap_attempts = len(self.replicas)**3
        n_successful_swaps = 0

        for _ in range(n_swap_attempts):
            # select two replicas as random to attempt to swap
            replica_i = np.random.randint(0, len(self.replicas))
            replica_j = np.random.randint(0, len(self.replicas))

            # attempt the swap
            result = self._attempt_swap(replica_i, replica_j)
            if result:
                n_successful_swaps += 1
                # record the new lambda value of the replica
                iter_swap_info[replica_i, replica_j] = self.replicas[replica_i].l

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

        # logP = 0 if the replicas if we end up with the same i and j
        log_p = -(u_ij + u_ji) + u_ii + u_jj
        # logger.debug(f"logP: {log_p}")
        #
        if log_p > 0 or np.random.random() < np.exp(log_p):
            # swap the replicas
            lambda_i = self.replicas[replica_i].atoms.calc.get_lambda()
            lambda_j = self.replicas[replica_j].atoms.calc.get_lambda()
            self.replicas[replica_i].atoms.calc.set_lambda(lambda_j)
            self.replicas[replica_j].atoms.calc.set_lambda(lambda_i)
            return True
        else:
            return False
        
    def _minimise_system(self, system_idx):
        atoms= self.replicas[system_idx].atoms
        logger.info(f"Minimising replica with lambda = {self.replicas[system_idx].l:.2f}")
        minimiser = LBFGS(atoms)
        minimiser.run(fmax=0.2)
        # return positions
        return atoms.get_positions()

    def minimise(self):
        min_positions = mpiplus.distribute(self._minimise_system, range(len(self.replicas)), send_results_to="all")
        # set the new positions
        for replica in self.replicas:
            print(replica.idx)
            replica.atoms.set_positions(min_positions[replica.idx])

        # write out the minimised positions
        for replica in self.replicas:
            replica.atoms.write(
                os.path.join(self.output_dir, f"minimised_replica_{replica.idx}.xyz"),
                append=True,
                parallel=False,
            )
