from dataclasses import dataclass
import logging
import os
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import get_distances
from ase.io import write
from typing import List
from ase import neighborlist
from mace.tools import torch_geometric, torch_tools, utils

import torch
import time
import numpy as np
from mace import data

logger = logging.getLogger("mace_fep")


# create a datastructure to hold the arrays for the interaction energy components
# this is a dataclass, so we can access the attributes by name, but it is also a dictionary, so we can iterate over the keys


# need to be store the energies and forces of various sizes.  Can we just attach to the atoms object?

@dataclass
class InteractionEnergyComponents:
    stateA_and_buffered_solvent_shell: np.ndarray = np.zeros()
    stateA_solute: np.ndarray = np.zeroes()
    stateB_and_buffered_solvent_shell: np.ndarray = None
    stateB_solute: np.ndarray = None
    solvent: np.ndarray = None
    node_energies: np.ndarray = None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        # print("Setting %s to %s" % (key, value))
        setattr(self, key, value)

    def __iter__(self):
        for key in self.__dict__.keys():
            yield key

    


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
        # cutoff around the solute where there is a significant change to the
        cutoff_radius: float = 5.0,
        default_dtype="float64",
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.inter_results = {
            "energy": InteractionEnergyComponents(),
            "free_energy": InteractionEnergyComponents(),
            # force has units eng / len:
            "forces": InteractionEnergyComponents(),
            "node_energies": InteractionEnergyComponents(),
        }
        self.results = {}
        self.cutoff_radius = cutoff_radius

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

    def get_molecules_within_radius(
        self, atoms: Atoms, cutoff: float, core_indices: List[int]
    ) -> List[int]:
        """Given the ase atoms object, compute a list of atoms within the radius of the core atoms. Perform a nearest neighbours search to ensure no dangling bonds across the QM/MM boundary."""
        core_mask = np.zeros(len(atoms), dtype=bool)
        core_mask[core_indices] = True

        # compute geometric center of core atoms
        core_center = np.mean(atoms[core_mask].positions, axis=0)
        # compute distances from the core center to all other atoms
        # this should mean that if the core molecule drifts, we can still compute the QM region correctly, Periodicity is handled by the get_distances

        _, distance_matrix = get_distances(
            core_center, atoms.positions, atoms.cell, atoms.pbc
        )

        # find the column indices of the atoms within the cutoff
        indices = np.unique(np.argwhere(distance_matrix < cutoff)[:, 1])

        index_mask = np.zeros(len(atoms), dtype=bool)
        index_mask[indices] = True
        # set the core atoms to be false, just extract the solvent shell

        # write out the atoms within the cutoff
        # write("within_cutoff.xyz", atoms[indices])
        nl = neighborlist.NeighborList(
            cutoffs=neighborlist.natural_cutoffs(atoms),
            self_interaction=False,
            bothways=True,
        )
        nl.update(atoms)

        connectivity = nl.get_connectivity_matrix(sparse=False)

        # now do an exhaustive exploration of the connectivity graph to find all atoms within the cutoff
        all_connected = False
        while not all_connected:
            n_added = 0
            for idx, row in enumerate(connectivity):
                for jdx, elem in enumerate(row):
                    if elem:
                        if index_mask[idx] and not index_mask[jdx]:
                            index_mask[jdx] = True
                            n_added += 1
            print(f"Added {n_added} atoms")
            if n_added == 0:
                all_connected = True
                print("All atoms connected in QM region")

        index_mask[core_indices] = False

        qm_region_atoms = atoms[index_mask]
        # convert the index mask into a list of indices
        qm_region_indices = np.argwhere(index_mask).flatten()
        # write("qm_region.xyz", qm_region_atoms)
        return qm_region_atoms, qm_region_indices

    def get_buffer_atom_indices(
        self, atoms: Atoms, selection_indices: List[int], buffer_width: float
    ) -> List[int]:
        """
        Initialises system to perform qm calculation
        # TODO: we should merge this function with the above for calculating the initial qm region
        """
        # calculate the distances between all atoms and qm atoms
        # qm_distance_matrix is a [N_QM_atoms x N_atoms] matrix
        _, qm_distance_matrix = get_distances(
            atoms.positions[selection_indices],
            atoms.positions,
            atoms.cell,
            atoms.pbc,
        )

        buffer_mask = np.zeros(len(atoms), dtype=bool)

        # every r_qm is a matrix of distances
        # from an atom in qm region and all atoms with size [N_atoms]
        # if the atom is within the buffer distance of any QM atom, add to mask
        for r_qm in qm_distance_matrix:
            # This returns the boolean array with which to index the qm_buffer_mask

            buffer_mask[r_qm < buffer_width] = True
        # print out number of atoms in qm region and buffer region

        # now guess the connectivity from the atomic distances, using a neighbour list
        self.neighbourList.update(atoms)
        connectivity = neighborlist.get_connectivity_matrix(
            self.neighbourList.nl, sparse=False
        )
        # n_atoms x n_atoms
        # this should be n_atoms x n_atoms, identify all the atoms connected to the qm buffer region atoms, and include those if not already included
        # loop over all the atoms in the qm buffer mask, if any connections are not already in the qm buffer mask, add them
        # write out the initial buffer region
        qm_buffer_atoms = atoms[buffer_mask]
        all_connected = False
        while not all_connected:
            n_added = 0
            for idx, row in enumerate(connectivity):
                # atoms connected to atom idx
                for jdx, elem in enumerate(row):
                    if elem:
                        if buffer_mask[idx] and not buffer_mask[jdx]:
                            buffer_mask[jdx] = True
                            n_added += 1
                            # print("Adding atom %d to QM buffer mask, as a neighbour of %d " % (jdx, idx))
                            # print(connectivity[idx][jdx])
            if n_added == 0:
                all_connected = True
        # write out the xyz of just the qm buffer region
        qm_buffer_atoms = atoms[buffer_mask]
        return qm_buffer_atoms
        # qm_buffer_atoms.write("qm_buffer.xyz")
        # print the final indices of the qm buffer mask

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

        self.neighbourList = neighborlist.NeighborList(
            bothways=True,
            cutoffs=neighborlist.natural_cutoffs(atoms),
        )

        # create 3 atoms objects, one of which is the original atoms object, then the solvent and solute aotms by indexing
        # state B idx is the state of the non-interacting ligand
        # solvent is a huge number of atoms, most of which do not experience any change in their interactions due to the removal of the ligand, since they are so distant.  We should compute forces for a subset of atoms that are close to the solvent, and use the full system forces for everything else.

        stateA_solute = atoms[self.stateA_idx]

        # first solvation shell around ligand A + ligand B
        # TODO: this assumes the same solvation shell around both ligands for now
        solvation_shell_atoms, solvation_shell_idx = self.get_molecules_within_radius(
            atoms,
            cutoff=self.cutoff_radius,
            core_indices=self.stateA_idx + self.stateB_idx,
        )
        print(solvation_shell_idx)
        # solvation_shell_atoms = atoms[solvation_shell_idx]

        # solvation shell buffer - skin of 2A around the solvation shell
        buffer_solvent_atoms = self.get_buffer_atom_indices(
            atoms, solvation_shell_idx, buffer_width=2.0
        )

        # write out the solvation shell, buffer and ligand A
        write(
            "solvation_shell.xyz",
            solvation_shell_atoms + stateA_solute + buffer_solvent_atoms,
        )

        # these are the solvent indices that should not be affected by the ligand decoupling
        lig_and_solvation_shell = [self.statdA_idx]
        lig_and_solvation_shell.extend(self.stateB_idx)
        lig_and_solvation_shell.extend(solvation_shell_idx)

        # outer solvent atoms are those not in the primary solvation shell, ligandA or ligandB
        outer_solvent_idx = [
            i for i in range(len(atoms)) if i not in lig_and_solvation_shell
        ]
        outer_solvent_atoms = atoms[outer_solvent_idx]
        # redefine the solvent atoms to be the distant ones outside the solvent shell.

        # now the full state A calculation needs the outer solvation node energies added to it.
        # buffer atoms there to ensure the core solvation shell sees the same environment as the full solvation shell.  We discard the energies + forces in the buffer zone.
        stateA_and_buffered_solvent_shell = (
            stateA_solute + solvation_shell_atoms + buffer_solvent_atoms
        )
        stateB_solute = atoms[self.stateB_idx]
        stateB_and_buffered_solvent_shell = (
            stateB_solute + solvation_shell_atoms + buffer_solvent_atoms
        )
        all_atoms = [
            stateA_and_buffered_solvent_shell,
            stateA_solute,
            stateB_and_buffered_solvent_shell,
            stateB_solute,
            # the most expensive part
            outer_solvent_atoms,
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
            node_energies = out["node_energy"].detach().cpu().numpy()
            self.inter_results["node_energies"][idx] = node_energies
            # energy = out["interaction_energy"].detach().cpu().item()
            # forces = out["forces"].detach().cpu().numpy()

            # store results
            E = energy * self.energy_units_to_eV
            self.inter_results["energy"][idx] = E
            self.inter_results["free_energy"][idx] = E
            self.inter_results["forces"][idx] = forces * (
                self.energy_units_to_eV / self.length_units_to_A
            )


        # modify the stateA array in place to remove the buffer atoms
        self.inter_results["energy"].stateA_total = np.zeros(len(atoms) - len(self.stateB_idx))
        self.inter_results["energy"].stateA_total[self.stateA_idx + solvation_shell_idx] = self.inter_results["energy"].stateA_and_buffered_solvent_shell[self.stateA_idx + solvation_shell_idx] 
        self.inter_results["energy"].stateA_total[outer_solvent_idx] = self.inter_results["energy"].solvent


        # do the same for state B - n_atoms is total - stateA ligand atoms
        self.inter_results["energy"].stateB_total = np.zeros(len(atoms) - len(self.stateA_idx))
        self.inter_results["energy"].stateB_total[self.stateB_idx + solvation_shell_idx] = self.inter_results["energy"].stateB_and_buffered_solvent_shell[self.stateB_idx + solvation_shell_idx]
        self.inter_results["energy"].stateB_total[outer_solvent_idx] = self.inter_results["energy"].solvent



        # all atoms
        stateA_isol_forces = np.concatenate(
            (self.inter_results["forces"].stateA_and_buffered_solvent_shell[self.stateA_idx + solvation_shell_idx], self.inter_results["forces"].solvent), axis=0
        )
        stateB_isol_forces = np.concatenate(
            (self.inter_results["forces"].stateB_and_buffered_solvent_shell[self.stateB_idx + solvation_shell_idx], self.inter_results["forces"].solvent), axis=0
        )

        final_forces = np.zeros((len(atoms), 3))

        final_forces[self.stateA_idx + outer_solvent_idx] = self.lmbda * self.inter_results[
            "forces"
        ] + (1 - self.lmbda) * (stateA_isol_forces)


        final_forces[self.stateB_idx + solvent_idx] = (
            1 - self.lmbda
        ) * self.inter_results["forces"][2] + self.lmbda * (stateB_isol_forces)

        self.results = {
            "energy": self.lmbda * self.inter_results["energy"].stateA_total
            + (1 - self.lmbda)
            * (self.inter_results["energy"].stateA_solute + self.inter_results["energy"].solvent)
            + (1 - self.lmbda) * self.inter_results["energy"][2].stateB_total
            + self.lmbda
            * (self.inter_results["energy"].stateB_solute + self.inter_results["energy"].solvent),
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
            # return the inter force contributions
            "inter_forces": self.inter_results["forces"],
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
