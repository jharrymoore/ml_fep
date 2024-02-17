from ase import Atoms
from ase.md.langevin import Langevin
from ase import units
import os

class Replica:
    def __init__(self, atoms: Atoms, 
                 idx: int, 
                 l: float,
                 output_dir: str,
                 timestep: float = 1.0,
                 temperature: float = 300.0,
                 friction: float = 0.01,
                 write_interval: int = 100):
        self.atoms = atoms
        self.idx = idx
        self.l = l

        self.integrator = Langevin(
            self.atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature,
            friction=friction,
        )

        def write_frame():
            self.integrator.atoms.write(
                os.path.join(output_dir, f"output_replica_{self.idx}.xyz"),
                append=True,
                parallel=False,
            )
        self.integrator.attach(write_frame, interval=write_interval)
    def propagate(self, n_steps: int):
        self.integrator.run(n_steps)





