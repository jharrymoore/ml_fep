from ase import Atoms, units
import time
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
import logging
from ase.optimize import LBFGS
from datetime import date
import datetime
import os
import numpy as np
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# ioan's settings for organic systems - from ethanol water simulations
tstep   = 1.0*units.fs
ttime   = 50*units.fs
B_water = 2.0*units.GPa #vs 100*units.GPa recommended default
ptime   = 2500*units.fs

MD_dt = 100
TH_dt = 10
pres  = 1.013
densfact = (units.m/1.0e2)**3/units.mol


logger = logging.getLogger("mace_fep")

class NEQSystem:
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
        elif integrator == "NPT":
            raise NotImplementedError("NPT not implemented")
            # logging.info("Setting up NPT dynamics")
            # self.integrator = NPT(atoms=self.atoms,
            #                       timestep=timestep * units.fs,
            #                       temperature_K=temperature,
            #                       externalstress=pres*units.bar,
            #                       ttime=ttime,
            #                       pfactor=ptime**2*B_water)
            # self.integrator.set_fraction_traceless(0)
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
            steps_per_day = (self.report_interval/time_elapsed) * 86400
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
            current_lambda = float(self.integrator.atoms.calc.lambda_schedule.current_lambda)
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

        self.integrator.attach(write_frame, interval=100)
        self.integrator.attach(print_traj, interval=100)
        self.integrator.attach(recompute_nl, interval=20)
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
