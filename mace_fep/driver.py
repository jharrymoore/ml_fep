import argparse
from typing import List
import os

import logging
import shutil
from ase.io import read

from ase.constraints import FixBondLength
from ase.md.langevin import Langevin

from mace_fep.replica_exchange.fep_calculator import MACEFEPCalculator


logger = logging.getLogger()

# pure MACE implementation of free energy perturbation, first to run solvation free energy calculations
# requires an overall neutarl waterbox


# Basic plan
# the hamiltonian of the sytsem can be decomposed into the solvent term, the solute term and the interaction term
# H = H_solute + H_solvent + H_interaction
# we can express the interaction energy as the difference between the total energy and the sum of the solute and solvent energies
# we can decompose the forces in the same way, and simulate the system at different values of lambda, where lambda controls the strength of the interaction energy term
# at every step, we need to evaluate forces and energies on the total system, the solute and the solvent separately and scale the interaction forces and energies by lambda
# then we can run multiple simulations and compute a free energy difference using the MBAR estimator


def run_mace_fep(
    lmbda: float,
    xyz_file: str,
    ligA_idx: List[int],
    ligB_idx: List[int],
    output: str,
    steps: int,
    interval: int,
    model_path: str,
):
    logger.info("Running MACE FEP")

    # access the node index the process is running on

    output_file = f"{output}/traj_{lmbda}.xyz"
    log_file = f"{output}/sim_{lmbda}.log"

    atoms = read(xyz_file)
    # minimise the atoms object
    # tmp_model = torch.load(model_path, map_location="cpu")
    # _, tmp_path = mkstemp(suffix=".pt")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tmp_model = tmp_model.to(device)
    # torch.save(tmp_model, tmp_path)
    # calc = MACECalculator(
    #     model_path=tmp_path,
    #     device="cuda",
    # )
    # atoms.set_calculator(calc)
    # opt = LBFGS(atoms)
    # logger.info("Minimising...")
    # opt.run(fmax=1e-2)

    # add a distance consttraint between the two ligands
    logger.info(f"Setting distance constraint between {ligA_idx} and {ligB_idx}")
    # find the oxygen atom in ligand A
    ligA_oxygen_idx = [i for i in ligA_idx if atoms[i].symbol == "O"][0]
    # find the oxygen atom in ligand B
    ligB_oxygen_idx = [i for i in ligB_idx if atoms[i].symbol == "O"][0]
    # constrain them at their current distance
    c = FixBondLength(ligA_oxygen_idx, ligB_oxygen_idx)
    atoms.set_constraint(c)

    logger.info(f"Read atoms {atoms}]")
    calculator = MACEFEPCalculator(
        model_path, device="cuda", lmbda=lmbda, stateA_idx=ligA_idx, stateB_idx=ligB_idx
    )

    atoms.set_calculator(calculator)

    integrator = Langevin(
        atoms,
        1 * units.fs,
        temperature_K=300,
        friction=1e-2,
        communicator=None,
        logfile=log_file,
        loginterval=interval,
    )

    def write_frame():
        logger.info("Writing frame, lambda = {} to file {}".format(lmbda, output_file))

        integrator.atoms.write(output_file, append=True, parallel=False)

    integrator.attach(write_frame, interval=interval)

    integrator.run(steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str)
    parser.add_argument("-lmbda", type=float, default=0.0)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument(
        "--model_path", type=str, default="input_files/SPICE_sm_inv_neut_E0_swa.model"
    )
    parser.add_argument("-s", "--steps", type=int, default=10000)
    parser.add_argument("-o", "--output", type=str, default="junk")
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("-c", "--clobber", action="store_true")
    # parser.add_argument("--idx", type=int, nargs="+")
    args = parser.parse_args()

    if args.clobber:
        shutil.rmtree(args.output)
        os.mkdir(args.output)
    else:
        os.makedirs(args.output, exist_ok=True)


if __name__ == "__main__":
    main()
