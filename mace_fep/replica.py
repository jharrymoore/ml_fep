from ase import Atoms
from ase.md.langevin import Langevin
from ase import units


class Replica:
    def __init__(self, atoms: Atoms, idx: int, l: float,
        timestep: float = 1.0,
        temperature: float = 300.0,
        friction: float = 0.01,
                 ):
        self.atoms = atoms
        self.idx = idx
        self.l = l

        self.integrator = Langevin(
            self.atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature,
            friction=friction,
        )
    def propagate(self, n_steps: int):
        self.integrator.run(n_steps)





