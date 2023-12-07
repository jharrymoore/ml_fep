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
from mace_fep.data import AtomicData
from mace import data
from mace.data import get_neighborhood
from copy import deepcopy

logger = logging.getLogger("mace_fep")


# create a datastructure to hold the arrays for the interaction energy components
# this is a dataclass, so we can access the attributes by name, but it is also a dictionary, so we can iterate over the keys


class FullCalcAbsoluteMACEFEPCalculator(Calculator):
    """MACE ASE Calculator"""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model_path: str,
        lmbda: float,
        stateA_idx: List[int],
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        # cutoff around the solute where there is a significant change to the
        cutoff_radius: float = 5.0,
        default_dtype="float64",
        **kwargs):
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.cutoff_radius = cutoff_radius

        self.model = torch.load(f=model_path, map_location=device)
        self.r_max = float(self.model.r_max)
        self.lmbda = lmbda
        self.original_lambda = lmbda
        self.buffer_connectivity = None
        # indices of the ligand atoms
        self.stateA_idx = stateA_idx
        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        torch_tools.set_default_dtype(default_dtype)
        self.step_counter = 0
        self.nl_cache = {}

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        solvent_idx = [i for i in range(len(atoms)) if i not in self.stateA_idx]
        solvent_atoms = atoms[solvent_idx]
        stateA_solute = atoms[self.stateA_idx]
        stateA = stateA_solute + solvent_atoms
        all_atoms = [
            stateA,
            stateA_solute,
            solvent_atoms,
        ]

        for idx, at in enumerate(all_atoms):
            # call to base-class to set atoms attribute
            Calculator.calculate(self, at)

            # prepare data
            config = data.config_from_atoms(at)

            self.step_counter += 1

            config = data.config_from_atoms(at)
            # extract the neighbourlist from cache, unless we're every N steps, in which case update it
            if self.step_counter % 20 != 0 and idx in self.nl_cache.keys():
                edge_index, shifts, unit_shifts = self.nl_cache[idx]
                nl = (edge_index, shifts, unit_shifts)
            else:
                # logger.debug("Updating neighbourlist at step %d" % self.step_counter)
                nl = get_neighborhood(
                    positions=config.positions,
                    cutoff=self.r_max,
                    pbc=config.pbc,
                    cell=config.cell,
                )
                self.nl_cache[idx] = nl
            data_loader = torch_geometric.dataloader.DataLoader(
                dataset=[
                    AtomicData.from_config(
                        config, z_table=self.z_table, cutoff=self.r_max, nl=nl
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
            energy = out["interaction_energy"].detach().cpu().item()
            forces = out["forces"].detach().cpu().numpy()
            # print(node_energies )
            # attach to internal atoms object.  These should still be accessible after the loop
            at.arrays["node_energies"] = node_energies
            at.arrays["forces"] = (
                forces * self.energy_units_to_eV / self.length_units_to_A
            )
            at.info["energy"] = energy * self.energy_units_to_eV

        stateA_decoupled_forces = np.concatenate(
            (stateA_solute.arrays["forces"], solvent_atoms.arrays["forces"]),
            axis=0,
        )

        final_forces = self.lmbda * stateA.arrays["forces"] + (1 - self.lmbda) * (
            stateA_decoupled_forces
        )

        self.results = {
            "energy": self.lmbda * stateA.info["energy"]
            + (1 - self.lmbda)
            * (stateA_solute.info["energy"] + solvent_atoms.info["energy"]),
            "free_energy": self.lmbda * stateA.info["energy"]
            + (1 - self.lmbda)
            * (stateA_solute.info["energy"] + solvent_atoms.info["energy"]),
            # difference between the total forces and the sun is that due to the interactions bettween solute and solvent.
            "forces": final_forces,
        }
        t2 = time.time()
        # logger.debug(f"Time taken for calculation: {t2-t1}")
        # get the final forces acting on the solute

    def set_lambda(self, lmbda: float) -> None:
        # Not thrilled about this, this allows us to change the value and run get_potential_energy.  I would love to be able to add a trait to say get_potential_energy_at_lambda but I would need to modify atoms.
        # logger.debug(f"Setting lambda to {lmbda:.2f}, from {self.lmbda:.2f}")
        self.lmbda = lmbda

    def get_lambda(self) -> float:
        return self.lmbda

    

    def reset_lambda(self) -> None:
        logger.debug(f"Resetting lambda to {self.original_lambda:.2f}")
        self.lmbda = self.original_lambda


class FullCalcMACEFEPCalculator(Calculator):
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
        self.results = {}
        self.cutoff_radius = cutoff_radius

        self.model = torch.load(f=model_path, map_location=device)
        self.r_max = float(self.model.r_max)
        self.lmbda = lmbda
        self.original_lambda = lmbda
        self.buffer_connectivity = None
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

        self.step_counter = 0

        # cache of atoms MACE data objects - positions tensors need updating, nothing else will be changing on short timescales, so we don't need a new neighbour list every time
        self.nl_cache = {}

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """

        # solvent atoms indexed in the main atoms reference frame
        solvent_idx = [
            i for i in range(len(atoms)) if i not in self.stateA_idx + self.stateB_idx
        ]

        # TODO: doing these copy ops is expensive, really we just want to cache these in the calculator, and update the positions only
        # can we get away with not making copies of the system at each step? we want to compute properties on the same atoms, but maybe call them different things? 

        solvent_atoms = atoms[solvent_idx]
        stateA_solute = atoms[self.stateA_idx]
        stateA = stateA_solute + solvent_atoms
        stateB_solute = atoms[self.stateB_idx]
        stateB = stateB_solute + solvent_atoms
        all_atoms = [
            stateA,
            stateA_solute,
            stateB,
            stateB_solute,
            # solvent_atoms,
        ]

        for idx, at in enumerate(all_atoms):
            # call to base-class to set atoms attribute
            Calculator.calculate(self, at)
            # iterate the step counter
            self.step_counter += 1

            config = data.config_from_atoms(at)
            # extract the neighbourlist from cache, unless we're every N steps, in which case update it
            if self.step_counter % 20 != 0 and idx in self.nl_cache.keys():
                edge_index, shifts, unit_shifts = self.nl_cache[idx]
                nl = (edge_index, shifts, unit_shifts)
            else:
                # logger.debug("Updating neighbourlist at step %d" % self.step_counter)
                nl = get_neighborhood(
                    positions=config.positions,
                    cutoff=self.r_max,
                    pbc=config.pbc,
                    cell=config.cell,
                )
                self.nl_cache[idx] = nl

            # this data is not going to change, other than the positions, cache the
            data_loader = torch_geometric.dataloader.DataLoader(
                dataset=[
                    AtomicData.from_config(
                        config, z_table=self.z_table, cutoff=self.r_max, nl=nl
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
            energy = out["interaction_energy"].detach().cpu().item()
            forces = out["forces"].detach().cpu().numpy()
            # print(node_energies )
            # attach to internal atoms object.  These should still be accessible after the loop
            at.arrays["node_energies"] = node_energies
            at.arrays["forces"] = (
                forces * self.energy_units_to_eV / self.length_units_to_A
            )
            at.info["energy"] = energy * self.energy_units_to_eV

        final_forces = np.zeros((len(atoms), 3))

        final_forces[self.stateA_idx] = (
            self.lmbda * stateA.arrays["forces"][:len(self.stateA_idx)]
            + (1 - self.lmbda) * stateA_solute.arrays["forces"]
        )

        final_forces[self.stateB_idx] = self.lmbda * stateB_solute.arrays["forces"] + (
            1 - self.lmbda
        ) * (stateB.arrays["forces"][:len(self.stateB_idx)])

        # now the solute + isolated term +
        final_forces[solvent_idx] = self.lmbda * (
            stateA.arrays["forces"][len(self.stateA_idx) :]
        ) + (1 - self.lmbda) * (stateB.arrays["forces"][len(self.stateB_idx) :])

        # energy expression: isolated + \lambda * interaction(A) + (1-\lambda) * interaction(B)
        energy = self.lmbda * (stateA.info["energy"] + stateB_solute.info["energy"]) + (
            1 - self.lmbda
        ) * (stateB.info["energy"] + stateA_solute.info["energy"])

        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": final_forces,
        }

    def set_lambda(self, lmbda: float) -> None:
        # Not thrilled about this, this allows us to change the value and run get_potential_energy.  I would love to be able to add a trait to say get_potential_energy_at_lambda but I would need to modify atoms.
        logger.debug(f"Setting lambda to {lmbda:.2f}, from {self.lmbda:.2f}")
        self.lmbda = lmbda
    
    def get_lambda(self) -> float:
        return self.lmbda

    def reset_lambda(self) -> None:
        logger.debug(f"Resetting lambda to {self.original_lambda:.2f}")
        self.lmbda = self.original_lambda


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
        self.results = {}
        self.cutoff_radius = cutoff_radius

        self.model = torch.load(f=model_path, map_location=device)
        self.r_max = float(self.model.r_max)
        self.lmbda = lmbda
        self.original_lambda = lmbda
        self.buffer_connectivity = None
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
        self, atoms: Atoms, cutoff: float, core_center: np.ndarray
    ) -> List[int]:
        """Given the ase atoms object, compute a list of atoms within the radius of the core atoms. Perform a nearest neighbours search to ensure no dangling bonds across the QM/MM boundary."""

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
        if self.buffer_connectivity is None:
            nl = neighborlist.NeighborList(
                cutoffs=neighborlist.natural_cutoffs(atoms),
                self_interaction=False,
                bothways=True,
            )
            nl.update(atoms)
            self.buffer_connectivity = nl.get_connectivity_matrix(sparse=False)

        # now do an exhaustive exploration of the connectivity graph to find all atoms within the cutoff
        all_connected = False
        while not all_connected:
            n_added = 0
            for idx, row in enumerate(self.buffer_connectivity):
                for jdx, elem in enumerate(row):
                    if elem:
                        if index_mask[idx] and not index_mask[jdx]:
                            index_mask[jdx] = True
                            n_added += 1
            logger.debug(f"Added {n_added} atoms")
            if n_added == 0:
                all_connected = True
                logger.debug("All atoms connected in QM region")

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
        # TODO:
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
            # This returns nthe boolean array with which to idex the qm_buffer_mask

            buffer_mask[r_qm < buffer_width] = True

        # now guess the connectivity from the atomic distances, using a neighbour list
        if self.buffer_connectivity is None:
            nl = neighborlist.NeighborList(
                cutoffs=neighborlist.natural_cutoffs(atoms),
                self_interaction=False,
                bothways=True,
            )
            nl.update(atoms)
            # this matrix should not change with geometry, assuming the water molecules stay together.
            self.buffer_connectivity = nl.get_connectivity_matrix(sparse=False)
        # n_atoms x n_atoms
        # this should be n_atoms x n_atoms, identify all the atoms connected to the qm buffer region atoms, and include those if not already included
        # loop over all the atoms in the qm buffer mask, if any connections are not already in the qm buffer mask, add them
        # write out the initial buffer region
        qm_buffer_atoms = atoms[buffer_mask]
        all_connected = False
        while not all_connected:
            n_added = 0
            for idx, row in enumerate(self.buffer_connectivity):
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
        # final check that the qm atoms do not make it into the buffer region
        buffer_mask[selection_indices] = False
        qm_buffer_atoms = atoms[buffer_mask]
        # get the indices of the atoms in the buffer region
        buffer_indices = np.argwhere(buffer_mask).flatten()
        return qm_buffer_atoms, buffer_indices
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

        # self.neighbourList = neighborlist.NeighborList(
        #     bothways=True,
        #     cutoffs=neighborlist.natural_cutoffs(atoms),
        # )
        # self.neighbourList.update(atoms)
        # solvent is a huge number of atoms, most of which do not experience any change in their interactions due to the removal of the ligand, since they are so distant.  We should compute forces for a subset of atoms that are close to the solvent, and use the full system forces for everything else.

        # compute geometric center of core atoms
        # full_atoms_idx = [i for i in range(len(atoms))]
        ligand_center = np.mean(
            atoms[self.stateA_idx + self.stateB_idx].positions, axis=0
        )

        solvent_idx = [
            i for i in range(len(atoms)) if i not in self.stateA_idx + self.stateB_idx
        ]
        solvent_atoms = atoms[solvent_idx]

        # Note: these indices refer to the solvent region, not the full set of atoms
        solvation_shell_atoms, solvation_shell_idx = self.get_molecules_within_radius(
            solvent_atoms, cutoff=self.cutoff_radius, core_center=ligand_center
        )

        # solvation shell buffer - skin of 2A around the solvation shell
        buffer_solvent_atoms, buffer_atoms_idx = self.get_buffer_atom_indices(
            solvent_atoms, solvation_shell_idx, buffer_width=3.0
        )
        n_buffered_atoms = len(buffer_solvent_atoms)
        # write the buffered atoms out

        # calculate the outer solvation atoms indices, which are the solvent atoms not in the solvation shell
        outer_solvent_idx = [
            i for i in range(len(solvent_atoms)) if i not in solvation_shell_idx
        ]

        # write out the solvation shell, buffer and ligand A
        stateA_solute = atoms[self.stateA_idx]
        # write(
        #     "solvation_shell.xyz",
        #     solvation_shell_atoms, append=True
        # )

        # now the full state A calculation needs the outer solvation node energies added to it.
        # buffer atoms there to ensure the core solvation shell sees the same environment as the full solvation shell.  We discard the energies + forces in the buffer zone.
        stateA_and_buffered_solvent_shell = (
            stateA_solute + solvation_shell_atoms + buffer_solvent_atoms
        )
        # write the atoms to disk
        # write("stateA.xyz", stateA_and_buffered_solvent_shell)
        stateB_solute = atoms[self.stateB_idx]
        stateB_and_buffered_solvent_shell = (
            stateB_solute + solvation_shell_atoms + buffer_solvent_atoms
        )
        # write("stateB.xyz", stateB_and_buffered_solvent_shell)
        all_atoms = [
            stateA_and_buffered_solvent_shell,
            stateA_solute,
            stateB_and_buffered_solvent_shell,
            stateB_solute,
            # the most expensive part
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
            node_energies = out["node_energy"].detach().cpu().numpy()
            energy = out["interaction_energy"].detach().cpu().item()
            forces = out["forces"].detach().cpu().numpy()
            # print(node_energies )
            # attach to internal atoms object.  These should still be accessible after the loop
            at.arrays["node_energies"] = node_energies
            at.arrays["forces"] = (
                forces * self.energy_units_to_eV / self.length_units_to_A
            )
            at.info["energy"] = energy * self.energy_units_to_eV

        # slice the buffer atoms off the solvation shell atoms object
        # stateA_and_buffered was created by concatenating the buffer last.  Simply remove the last n_buffer atoms
        stateA = (
            stateA_and_buffered_solvent_shell[:-n_buffered_atoms]
            + solvent_atoms[outer_solvent_idx]
        )

        # this has taken forces from the outer solvation shell, and the inner solvation shell + the ligand together, instead of forces from the entire system, which is what we had previously

        # set the total energies by summing over the node energies for all atoms
        stateA.info["energy"] = np.sum(stateA.arrays["node_energies"])

        stateB = (
            stateB_and_buffered_solvent_shell[:-n_buffered_atoms]
            + solvent_atoms[outer_solvent_idx]
        )

        stateB.info["energy"] = np.sum(stateB.arrays["node_energies"])

        # now we have the full set of 6 atoms objects with energies and forces

        # all atoms
        stateA_decoupled_forces = np.concatenate(
            (stateA_solute.arrays["forces"], solvent_atoms.arrays["forces"]),
            axis=0,
        )
        stateB_decoupled_forces = np.concatenate(
            (stateB_solute.arrays["forces"], solvent_atoms.arrays["forces"]),
            axis=0,
        )

        final_forces = np.zeros((len(atoms), 3))
        # print(stateA.arrays["forces"].shape)
        # print(stateA_decoupled_forces.shape)
        # print(final_forces[self.stateA_idx + list(itemgetter(*solvation_shell_idx)(itemgetter(*solvent_idx)(full_atoms_idx))) + outer_solvent_idx].shape)

        # forces on the solvent due to INTERACTION with ligand A
        # select all indices in stateA that are not the ligand
        # this indexing works becase we construct the stateA/B objects by concatenation of existing atoms, preserving atom ordering
        # solvent_stateA_interaction_force = stateA.arrays["forces"][len(self.stateA_idx):] - solvent_atoms.arrays["forces"]
        # # forces on the solvent due to INTERACTION with ligand B
        # solvent_stateB_interaction_force = stateB.arrays["forces"][len(self.stateB_idx):] - solvent_atoms.arrays["forces"]

        # final_forces[
        #     self.stateA_idx + solvent_idx] += self.lmbda * stateA.arrays["forces"] + (1 - self.lmbda) * (
        #     stateA_decoupled_forces
        # )

        # # at lambda = 1, which is stable, the solvent forces get overwritten with their isolated component only, and the fully decoupled  ligand B interaction.  So in this case we are missing the interaction between
        # final_forces[self.stateB_idx + solvent_idx] = (
        #     1 - self.lmbda
        # ) * stateB.arrays["forces"] + self.lmbda * stateB_decoupled_forces

        final_forces[self.stateA_idx] = self.lmbda * stateA[self.stateA_idx].arrays[
            "forces"
        ] + (1 - self.lmbda) * (stateA_solute.arrays["forces"])
        final_forces[self.stateB_idx] = (1 - self.lmbda) * stateB[
            self.stateB_idx
        ].arrays["forces"] + self.lmbda * (stateB_solute.arrays["forces"])

        # then the solvent is the background plus the combination of the interactions with A and B
        # final_forces[solvent_idx] = self.lmbda * solvent_stateA_interaction_force + (1 - self.lmbda) * solvent_stateB_interaction_force + solvent_atoms.arrays["forces"]

        # get the total energies for state A by summing the relevant components

        self.results = {
            "energy": self.lmbda * stateA.info["energy"]
            + (1 - self.lmbda)
            * (stateA_solute.info["energy"] + solvent_atoms.info["energy"])
            + (1 - self.lmbda) * stateB.info["energy"]
            + self.lmbda
            * (stateB_solute.info["energy"] + solvent_atoms.info["energy"]),
            "free_energy": self.lmbda * stateA.info["energy"]
            + (1 - self.lmbda)
            * (stateA_solute.info["energy"] + solvent_atoms.info["energy"])
            + (1 - self.lmbda) * stateB.info["energy"]
            + self.lmbda
            * (stateB_solute.info["energy"] + solvent_atoms.info["energy"]),
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


# create a datastructure to hold the arrays for the interaction energy components
# this is a dataclass, so we can access the attributes by name, but it is also a dictionary, so we can iterate over the keys


class AbsoluteMACEFEPCalculator(Calculator):
    """MACE ASE Calculator"""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model_path: str,
        lmbda: float,
        stateA_idx: List[int],
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        # cutoff around the solute where there is a significant change to the
        cutoff_radius: float = 5.0,
        default_dtype="float64",
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.cutoff_radius = cutoff_radius

        self.model = torch.load(f=model_path, map_location=device)
        self.r_max = float(self.model.r_max)
        self.lmbda = lmbda
        self.original_lambda = lmbda
        self.buffer_connectivity = None
        # indices of the ligand atoms
        self.stateA_idx = stateA_idx
        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        torch_tools.set_default_dtype(default_dtype)

    def get_molecules_within_radius(
        self, atoms: Atoms, cutoff: float, core_center: np.ndarray
    ) -> List[int]:
        """Given the ase atoms object, compute a list of atoms within the radius of the core atoms. Perform a nearest neighbours search to ensure no dangling bonds across the QM/MM boundary."""

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
        # if self.buffer_connectivity is None:
        nl = neighborlist.NeighborList(
            cutoffs=neighborlist.natural_cutoffs(atoms),
            self_interaction=False,
            bothways=True,
        )
        nl.update(atoms)
        self.buffer_connectivity = nl.get_connectivity_matrix(sparse=False)

        # now do an exhaustive exploration of the connectivity graph to find all atoms within the cutoff
        all_connected = False
        while not all_connected:
            n_added = 0
            for idx, row in enumerate(self.buffer_connectivity):
                for jdx, elem in enumerate(row):
                    if elem:
                        if index_mask[idx] and not index_mask[jdx]:
                            index_mask[jdx] = True
                            n_added += 1
            logger.debug(f"Added {n_added} atoms")
            if n_added == 0:
                all_connected = True
                logger.debug("All atoms connected in QM region")

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
        # TODO:
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
            # This returns nthe boolean array with which to idex the qm_buffer_mask

            buffer_mask[r_qm < buffer_width] = True

        # now guess the connectivity from the atomic distances, using a neighbour list
        # if self.buffer_connectivity is None:
        nl = neighborlist.NeighborList(
            cutoffs=neighborlist.natural_cutoffs(atoms),
            self_interaction=False,
            bothways=True,
        )
        nl.update(atoms)
        # this matrix should not change with geometry, assuming the water molecules stay together.
        self.buffer_connectivity = nl.get_connectivity_matrix(sparse=False)
        # n_atoms x n_atoms
        # this should be n_atoms x n_atoms, identify all the atoms connected to the qm buffer region atoms, and include those if not already included
        # loop over all the atoms in the qm buffer mask, if any connections are not already in the qm buffer mask, add them
        # write out the initial buffer region
        qm_buffer_atoms = atoms[buffer_mask]
        all_connected = False
        while not all_connected:
            n_added = 0
            for idx, row in enumerate(self.buffer_connectivity):
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
        # final check that the qm atoms do not make it into the buffer region
        buffer_mask[selection_indices] = False
        qm_buffer_atoms = atoms[buffer_mask]
        # get the indices of the atoms in the buffer region
        buffer_indices = np.argwhere(buffer_mask).flatten()
        return qm_buffer_atoms, buffer_indices
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

        ligand_center = np.mean(atoms[self.stateA_idx].positions, axis=0)

        solvent_idx = [i for i in range(len(atoms)) if i not in self.stateA_idx]
        solvent_atoms = atoms[solvent_idx]

        # Note: these indices refer to the solvent region, not the full set of atoms
        solvation_shell_atoms, solvation_shell_idx = self.get_molecules_within_radius(
            solvent_atoms, cutoff=self.cutoff_radius, core_center=ligand_center
        )

        # solvation shell buffer - skin of 2A around the solvation shell
        buffer_solvent_atoms, buffer_atoms_idx = self.get_buffer_atom_indices(
            solvent_atoms, solvation_shell_idx, buffer_width=3.0
        )
        n_buffered_atoms = len(buffer_solvent_atoms)
        # write the buffered atoms out
        write("buffered_atoms.xyz", buffer_solvent_atoms, append=True)

        # calculate the outer solvation atoms indices, which are the solvent atoms not in the solvation shell
        outer_solvent_idx = [
            i for i in range(len(solvent_atoms)) if i not in solvation_shell_idx
        ]

        # write out the solvation shell, buffer and ligand A
        stateA_solute = atoms[self.stateA_idx]
        # write(
        #     "solvation_shell.xyz",
        #     solvation_shell_atoms, append=True
        # )

        # now the full state A calculation needs the outer solvation node energies added to it.
        # buffer atoms there to ensure the core solvation shell sees the same environment as the full solvation shell.  We discard the energies + forces in the buffer zone.
        stateA_and_buffered_solvent_shell = (
            stateA_solute + solvation_shell_atoms + buffer_solvent_atoms
        )
        # write the atoms to disk
        # write("stateA.xyz", stateA_and_buffered_solvent_shell)

        # write("stateB.xyz", stateB_and_buffered_solvent_shell)
        all_atoms = [
            stateA_and_buffered_solvent_shell,
            stateA_solute,
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
            node_energies = out["node_energy"].detach().cpu().numpy()
            energy = out["interaction_energy"].detach().cpu().item()
            forces = out["forces"].detach().cpu().numpy()
            # print(node_energies )
            # attach to internal atoms object.  These should still be accessible after the loop
            at.arrays["node_energies"] = node_energies
            at.arrays["forces"] = (
                forces * self.energy_units_to_eV / self.length_units_to_A
            )
            at.info["energy"] = energy * self.energy_units_to_eV

        # slice the buffer atoms off the solvation shell atoms object
        # stateA_and_buffered was created by concatenating the buffer last.  Simply remove the last n_buffer atoms
        # lets check on the stateA and buffered solvent shell
        write(
            "stateA_and_buffered_solvent_shell.xyz",
            stateA_and_buffered_solvent_shell,
            append=True,
        )

        stateA = (
            stateA_and_buffered_solvent_shell[:-n_buffered_atoms]
            + solvent_atoms[outer_solvent_idx]
        )

        # write state A to disk
        write("stateA.xyz", stateA, append=True)

        # this has taken forces from the outer solvation shell, and the inner solvation shell + the ligand together, instead of forces from the entire system, which is what we had previously

        # set the total energies by summing over the node energies for all atoms
        stateA.info["energy"] = np.sum(stateA.arrays["node_energies"])

        # stateB.info["energy"] = np.sum(stateB.arrays["node_energies"])

        # now we have the full set of 6 atoms objects with energies and forces

        # all atoms
        stateA_decoupled_forces = np.concatenate(
            (stateA_solute.arrays["forces"], solvent_atoms.arrays["forces"]),
            axis=0,
        )
        # stateB_decoupled_forces = np.concatenate(
        #     (
        #         stateB_solute.arrays["forces"],
        #         solvent_atoms.arrays["forces"]
        #     ),
        #     axis=0,
        # )

        final_forces = np.zeros((len(atoms), 3))
        # print(stateA.arrays["forces"].shape)
        # print(stateA_decoupled_forces.shape)
        # print(final_forces[self.stateA_idx + list(itemgetter(*solvation_shell_idx)(itemgetter(*solvent_idx)(full_atoms_idx))) + outer_solvent_idx].shape)

        # forces on the solvent due to INTERACTION with ligand A
        # print(self.stateA_idx)
        # select all indices in stateA that are not the ligand
        # this indexing works becase we construct the stateA/B objects by concatenation of existing atoms, preserving atom ordering
        # solvent_stateA_interaction_force = stateA.arrays["forces"][len(self.stateA_idx):] - solvent_atoms.arrays["forces"]
        # # forces on the solvent due to INTERACTION with ligand B
        # solvent_stateB_interaction_force = stateB.arrays["forces"][len(self.stateB_idx):] - solvent_atoms.arrays["forces"]

        print(stateA.arrays["forces"].shape)
        print(stateA_decoupled_forces.shape)
        final_forces = self.lmbda * stateA.arrays["forces"] + (1 - self.lmbda) * (
            stateA_decoupled_forces
        )

        # # at lambda = 1, which is stable, the solvent forces get overwritten with their isolated component only, and the fully decoupled  ligand B interaction.  So in this case we are missing the interaction between
        # final_forces[self.stateB_idx + solvent_idx] = (
        #     1 - self.lmbda
        # ) * stateB.arrays["forces"] + self.lmbda * stateB_decoupled_forces

        # final_forces[self.stateA_idx] = self.lmbda * stateA[self.stateA_idx].arrays["forces"] + (1 - self.lmbda) * (
        #     stateA_solute.arrays["forces"]
        # )

        # then the solvent is the background plus the combination of the interactions with A and B
        # final_forces[solvent_idx] = self.lmbda * solvent_stateA_interaction_force + (1 - self.lmbda) * solvent_stateB_interaction_force + solvent_atoms.arrays["forces"]

        # get the total energies for state A by summing the relevant components

        self.results = {
            "energy": self.lmbda * stateA.info["energy"]
            + (1 - self.lmbda)
            * (stateA_solute.info["energy"] + solvent_atoms.info["energy"]),
            "free_energy": self.lmbda * stateA.info["energy"]
            + (1 - self.lmbda)
            * (stateA_solute.info["energy"] + solvent_atoms.info["energy"]),
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
