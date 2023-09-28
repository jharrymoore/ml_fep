from mace_fep.replica_exchange.fep_calculator import MACEFEPCalculator
from ase.io import read, write


conf = read("input_files/methanol_ethanol_superimposed.xyz")


calc = MACEFEPCalculator(
    model_path="input_files/SPICE_sm_inv_neut_E0_swa.model",
    stateA_idx=[i for i in range(0, 6)],
    stateB_idx=[i for i in range(6, 15)],
    lmbda=1.0,
    device="cuda",
)


conf.set_calculator(calc)


inter_forces = conf.get_potential_energy()


# access the intra forces
intra_forces = calc.inter_results["forces"]


node_energies = calc.inter_results["node_energies"]
total_energies = calc.inter_results["energy"]


# pickle the intra forces to disk
import pickle

with open("inter_results.pkl", "wb") as f:
    pickle.dump(calc.inter_results, f)


# print(len(intra_forces))


# state_A_all_atoms = intra_forces[0]
# solvent_atoms = intra_forces[-2]

# print(state_A_all_atoms.shape)
# print(solvent_atoms.shape)
# state_A_solvent = state_A_all_atoms[6:,:]
