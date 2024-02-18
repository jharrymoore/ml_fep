from ase import Atoms
from ase.md.langevin import Langevin
from ase import units
import time
import datetime
from datetime import date
import os
import numpy as np
import logging


logger = logging.getLogger("mace_fep")


densfact = (units.m/1.0e2)**3/units.mol

class Replica:
    def __init__(self,
                 atoms: Atoms, 
                 idx: int, 
                 l: float,
                 output_dir: str,
                 total_steps: int,
                 timestep: float = 1.0,
                 temperature: float = 300.0,
                 friction: float = 0.01,
                 write_interval: int = 100,
                 ):
        self.atoms = atoms
        self.idx = idx
        self.l = l
        self.checkpoint_time = time.time()
        self.total_steps=total_steps
        self.header = '   Time(fs)     Latency(ns/day)    Temperature(K)      Density(g/cm$^3$)        Energy(eV)        MSD(A$^2$)       COMSD(A$^2$)   Volume(cm$^3$)   Time remaining\n'
        self.output_dir=output_dir

        self.integrator = Langevin(
            self.atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature,
            friction=friction,
        )

        start_config_com = atoms.get_center_of_mass().copy()
        start_config_positions = atoms.positions.copy()

        def write_frame():
            self.integrator.atoms.write(
                os.path.join(output_dir, f"output_replica_{self.idx}.xyz"),
                append=True,
                parallel=False,
            )
        def update_nl():
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
            calc_epot = a.get_potential_energy()
            dhdl = a.get_potential_energy(force_consistent=True)
            calc_msd  = (((a.positions-a.get_center_of_mass())-(start_config_positions-start_config_com))**2).mean(0).sum(0)
            calc_drft = ((a.get_center_of_mass()-start_config_com)**2).sum(0)
            calc_volume = a.get_volume() * densfact
            a.info['step'] = self.integrator.nsteps
            a.info['time_fs'] = self.integrator.get_time()/units.fs
            a.info['time_ps'] = self.integrator.get_time()/units.fs/1000
            with open(os.path.join(output_dir, f"thermo_traj_{self.idx}.dat"), 'a') as thermo_traj:
                thermo_traj.write(('%12.1f'+' %17.6f'*7+'    %s'+'\n') % (calc_time, ns_per_day, calc_temp, calc_dens, calc_epot, calc_msd, calc_drft, calc_volume, time_remaining ))
                thermo_traj.flush()
            with open(os.path.join(output_dir, "dhdl.xvg"), 'a') as xvg:
                time_ps = calc_time 
                xvg.write(f"{time_ps:.4f} {dhdl:.6f}\n")
            self.checkpoint_time = time.time()
        self.integrator.attach(write_frame, interval=write_interval)
        self.integrator.attach(update_nl, interval=1)
        self.integrator.attach(print_traj, interval=write_interval)
    def propagate(self, n_steps: int):

        with open(os.path.join(self.output_dir, f"thermo_traj_{self.idx}.dat"), 'a') as thermo_traj:
            thermo_traj.write('# ASE Dynamics. Date: '+date.today().strftime("%d %b %Y")+'\n')
            thermo_traj.write(self.header)
            print('# ASE Dynamics. Date: '+date.today().strftime("%d %b %Y"))
        self.integrator.run(n_steps)





