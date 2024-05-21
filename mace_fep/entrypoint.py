from copy import deepcopy
import ast
from typing import Tuple
from mace_fep.calculators import NEQ_MACE_AFE_Calculator_NEW, EQ_MACE_AFE_Calculator
from mace_fep.lambda_schedule import LambdaSchedule

import logging
import os
import numpy as np
from mace.tools import set_default_dtype
from ase.io import read
from ase import Atoms
from mace_fep.protocols import NonEquilibriumSwitching, ReplicaExchange
from mace_fep.replica import Replica
from mace_fep.utils import parse_arguments, setup_logger

log_level = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

# def initialize_system(init_lambda: float, fep_calc: Calculator, last_recorded_step: int):
#     # we just have the one Atoms object, and we propagate the lambda value with the trajectory
#     # set the lambda to the last value if the trajectory data file already exists
#
#     self.atoms.set_calculator(fep_calc)


def setup_atoms(
    restart: bool, xyz_file: str, output_dir: str, idx: int = 0
) -> Tuple[Atoms, int]:
    if not restart:
        atoms = read(xyz_file)
        last_recorded_step = 0
    else:
        last_traj = os.path.join(output_dir, f"output_replica_{idx}.xyz")
        atoms = read(last_traj, -1)
        last_recorded_step = atoms.info["step"]
        logging.info(f"Restarting from step {last_recorded_step} from {last_traj}")

        # we need to remove the frames from the dhdl + thermo file after the last timestep
        # or maybe we just remove the duplicates after the fact
        os.system(
            f"head -n {last_recorded_step+3} {output_dir}/thermo_traj_{idx}.dat > {output_dir}/thermo_traj_{idx}.temp"
        )
        os.system(
            f"mv {output_dir}/thermo_traj_{idx}.temp {output_dir}/thermo_traj_{idx}.dat"
        )

        os.system(
            f"head -n {last_recorded_step+2} {output_dir}/dhdl.xvg > {output_dir}/dhdl.temp"
        )
        os.system(f"mv {output_dir}/dhdl.temp {output_dir}/dhdl.xvg")
    return atoms, last_recorded_step


def main():
    args = parse_arguments().parse_args()
    logger = logging.getLogger("mace_fep")
    logger.setLevel(log_level[args.log_level])
    set_default_dtype(args.dtype)
    setup_logger(level=log_level[args.log_level], tag="mace_fep", directory=args.output)

    ligA_idx = range(args.ligA_idx)
    ligB_idx = range(args.ligB_idx) if args.ligB_idx is not None else None

    logger.info(f"ligA_idx: {ligA_idx}")
    logger.info(f"ligB_idx: {ligB_idx}")

    # make the output dir if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    constrain_atoms_idx = []
    if args.ligA_const is not None:
        constrain_atoms_idx.append(args.ligA_const)
    if args.ligB_const is not None:
        constrain_atoms_idx.append(args.ligB_const)

    atoms, last_recorded_step = setup_atoms(args.restart, args.file, args.output)
    if args.mode == "NEQ":
        # atoms, last_recorded_step = setup_atoms(args.restart, args.file, args.output)
        if args.lambdas is not None:
            assert len(lambdas) == 1, "Only one lambda value can be specified for NEQ"
        steps_remaining = args.steps - last_recorded_step
        if args.equilibrate:
            delta_lambda = 0.0
        else:
            delta_lambda = 1.0 / args.steps if not args.reverse else -1.0 / args.steps
        logger.info(f"Delta lambda: {delta_lambda}")
        lambda_schedule = LambdaSchedule(
            last_step=last_recorded_step,
            delta=delta_lambda,
            total_steps=args.steps,
            n_steps=steps_remaining,
            reverse=args.reverse,
            use_ssc=args.use_ssc,
            lambda_val=args.lambdas[0]
        )
        fep_calc = NEQ_MACE_AFE_Calculator_NEW(
            model_path=args.model_path,
            decouple_indices=ligA_idx,
            default_dtype=args.dtype,
            device=args.device,
            lambda_schedule=lambda_schedule,
        )
        atoms.set_calculator(fep_calc)

        sampler = NonEquilibriumSwitching(
            atoms=atoms,
            total_steps=args.steps,
            constrain_atoms_idx=constrain_atoms_idx,
            output_dir=args.output,
            report_interval=args.report_interval,
        )

    elif args.mode == "EQ":
        all_atoms = []
        # TODO: this takes the same model for all replicas, not the lambda dependent scheme
        model_paths = [args.model_path for _ in range(args.replicas)]
        lambdas = (
            np.linspace(0, 1, args.replicas)
            if args.lambdas is None
            else ast.literal_eval(args.lambdas)
        )
        logger.debug(f"Using lambda schedule {lambdas}")
        for idx, l in enumerate(lambdas):
            model = model_paths[idx]
            atoms = deepcopy(atoms)
            calc = EQ_MACE_AFE_Calculator(
                model_path=model,
                ligA_idx=ligA_idx,
                default_dtype=args.dtype,
                device=args.device,
                l=l,
            )
            atoms.set_calculator(calc)
            all_atoms.append(atoms)

        replicas = [
            Replica(
                atoms=ats,
                idx=idx,
                l=l,
                output_dir=args.output,
                write_interval=args.report_interval,
                total_steps=args.steps_per_iter * args.iters,
            )
            for idx, (ats, l) in enumerate(zip(all_atoms, lambdas))
        ]
        sampler = ReplicaExchange(
            replicas=replicas,
            steps_per_iter=args.steps_per_iter,
            iters=args.iters,
            output_dir=args.output,
            dtype=args.dtype,
            no_mixing=args.no_mixing,
            restart=args.restart,
        )
    else:
        raise ValueError(f"Did not recognise mode {args.mode}")

    if args.minimise and not args.restart:
        sampler.minimise()
    sampler.propagate()


if __name__ == "__main__":
    ()
