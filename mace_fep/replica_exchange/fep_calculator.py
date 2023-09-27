import logging
from ase.calculators.calculator import Calculator, all_changes
from typing import List
from mace.tools import torch_geometric, torch_tools, utils

import torch
import time
import numpy as np
from mace import data

logger = logging.getLogger("mace_fep")


class MACEFEPCalculator(Calculator):
    """MACE ASE Calculator"""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model_path: str,
        lmbda: float,
        stateA_idx: List[int],
        stateB_idx: List[int],
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="float64",
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.inter_results = {
            "energy": [0.0, 0.0, 0.0, 0.0, 0.0],
            "free_energy": [0.0, 0.0, 0.0, 0.0, 0.0],
            # force has units eng / len:
            "forces": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        self.results = {}

        self.model = torch.load(f=model_path, map_location=device)
        self.r_max = float(self.model.r_max)
        self.lmbda = lmbda
        self.original_lambda = lmbda
        # indices of the ligand atoms
        self.stateA_idx = stateA_idx
        self.stateB_idx = stateB_idx
        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        torch_tools.set_default_dtype(default_dtype)

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        t1 = time.time()

        # create 3 atoms objects, one of which is the original atoms object, then the solvent and solute aotms by indexing
        # state B idx is the state of the non-interacting ligand
        solvent_idx = [
            i for i in range(len(atoms)) if i not in self.stateA_idx + self.stateB_idx
        ]
        solvent_atoms = atoms[solvent_idx]
        stateA_solute = atoms[self.stateA_idx]
        stateA_all_atoms = stateA_solute + solvent_atoms
        stateB_solute = atoms[self.stateB_idx]
        stateB_all_atoms = stateB_solute + solvent_atoms
        all_atoms = [
            stateA_all_atoms,
            stateA_solute,
            stateB_all_atoms,
            stateB_solute,
            solvent_atoms,
        ]

        for idx, at in enumerate(all_atoms):
            # call to base-class to set atoms attribute
            Calculator.calculate(self, at)

            # prepare data
            config = data.config_from_atoms(at)
            data_loader = torch_geometric.dataloader.DataLoader(
                dataset=[
                    data.AtomicData.from_config(
                        config, z_table=self.z_table, cutoff=self.r_max
                    )
                ],
                batch_size=1,
                shuffle=False,
                drop_last=False,
            )
            batch = next(iter(data_loader)).to(self.device)

            # predict + extract data
            out = self.model(batch.to_dict(), compute_stress=False)
            energy = out["interaction_energy"].detach().cpu().item()
            forces = out["forces"].detach().cpu().numpy()

            # store results
            E = energy * self.energy_units_to_eV
            self.inter_results["energy"][idx] = E
            self.inter_results["free_energy"][idx] = E
            self.inter_results["forces"][idx] = forces * (
                self.energy_units_to_eV / self.length_units_to_A
            )
        stateA_isol_forces = np.concatenate(
            (self.inter_results["forces"][1], self.inter_results["forces"][4]), axis=0
        )
        stateB_isol_forces = np.concatenate(
            (self.inter_results["forces"][3], self.inter_results["forces"][4]), axis=0
        )

        final_forces = np.zeros((len(atoms), 3))
        final_forces[self.stateA_idx + solvent_idx] = self.lmbda * self.inter_results[
            "forces"
        ][0] + (1 - self.lmbda) * (stateA_isol_forces)
        final_forces[self.stateB_idx + solvent_idx] = (
            1 - self.lmbda
        ) * self.inter_results["forces"][2] + self.lmbda * (stateB_isol_forces)

        self.results = {
            "energy": self.lmbda * self.inter_results["energy"][0]
            + (1 - self.lmbda)
            * (self.inter_results["energy"][1] + self.inter_results["energy"][4])
            + (1 - self.lmbda) * self.inter_results["energy"][2]
            + self.lmbda
            * (self.inter_results["energy"][3] + self.inter_results["energy"][4]),
            "free_energy": self.lmbda * self.inter_results["free_energy"][0]
            + (1 - self.lmbda)
            * (
                self.inter_results["free_energy"][1]
                + self.inter_results["free_energy"][4]
            )
            + (1 - self.lmbda) * self.inter_results["free_energy"][2]
            + self.lmbda
            * (
                self.inter_results["free_energy"][3]
                + self.inter_results["free_energy"][4]
            ),
            # difference between the total forces and the sun is that due to the interactions bettween solute and solvent.
            "forces": final_forces,
        }
        t2 = time.time()
        logger.debug(f"Time taken for calculation: {t2-t1}")
        # get the final forces acting on the solute

    def set_lambda(self, lmbda: float) -> None:
        # Not thrilled about this, this allows us to change the value and run get_potential_energy.  I would love to be able to add a trait to say get_potential_energy_at_lambda but I would need to modify atoms.
        logger.debug(f"Setting lambda to {lmbda}, from {self.lmbda}")
        self.lmbda = lmbda

    def reset_lambda(self) -> None:
        logger.debug(f"Resetting lambda to {self.original_lambda}")
        self.lmbda = self.original_lambda
