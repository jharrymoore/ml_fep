import argparse
from mace_fep.replica_exchange.fep_calculator import AbsoluteMACEFEPCalculator, FullCalcAbsoluteMACEFEPCalculator, FullCalcMACEFEPCalculator, NEQ_MACE_AFE_Calculator, NEQ_MACE_RFE_Calculator

from mace_fep.replica_exchange.replica_exchange import NonEquilibriumSwitching, ReplicaExchange
import logging
import os
from mace.tools import set_default_dtype

from mace_fep.utils import setup_logger

log_level = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--steps_per_iter", type=int, default=100)
    parser.add_argument(
        "--model_path", type=str, default="input_files/SPICE_sm_inv_neut_E0_swa.model"
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("-o", "--output", type=str, default="junk")
    parser.add_argument("--minimise", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--ligA_idx", type=int, help="open interval [0, ligA_idx) selects the ligand atoms for ligA", default=None)
    parser.add_argument("--ligB_idx", type=int, help="open interval [ligA_idx, ligB_idx) selects the ligand atoms for ligB", default=None)
    parser.add_argument("--ligA_const", help="atom to constrain in ligA", type=int)
    parser.add_argument("--ligB_const", help="atom to constrain in ligB", default=None, type=int)
    parser.add_argument("--mode", choices=["EQAbsolute", "EQRelative", "NEQAbsolute", "NEQRelative"])
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--no-mixing", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--equilibrate", action="store_true")
    parser.add_argument("--report_interval", type=int, default=100)
    args = parser.parse_args()
    logger = logging.getLogger("mace_fep")
    logger.setLevel(log_level[args.log_level])
    set_default_dtype(args.dtype)
    setup_logger(level=log_level[args.log_level], tag="mace_fep", directory=args.output)


    ligA_idx = [i for i in range(0, args.ligA_idx)]
    ligB_idx = [i for i in range(args.ligA_idx, args.ligB_idx)] if args.ligB_idx is not None else None
    

    logger.info(f"ligA_idx: {ligA_idx}")
    logger.info(f"ligB_idx: {ligB_idx}")

    # make the output dir if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)


    if args.mode == "absolute":
        fep_calc = FullCalcAbsoluteMACEFEPCalculator
    elif args.mode == "relative":
        fep_calc = FullCalcMACEFEPCalculator
    elif args.mode == "NEQAbsolute":
        fep_calc = NEQ_MACE_AFE_Calculator
    elif args.mode == "NEQRelative":
        fep_calc = NEQ_MACE_RFE_Calculator
    else:
        raise ValueError("mode must be absolute or relative")

    constrain_atoms_idx = []
    if args.ligA_const is not None:
        constrain_atoms_idx.append(args.ligA_const)
    if args.ligB_const is not None:
        constrain_atoms_idx.append(args.ligB_const)
    

    if args.mode in ["EQRelative", "EQAbsolute"]:
        sampler = ReplicaExchange(
            mace_model=args.model_path,
            output_dir=args.output,
            iters=args.iters,
            steps_per_iter=args.steps_per_iter,
            xyz_file=args.file,
            ligA_idx=ligA_idx,
            ligB_idx=ligB_idx,
            replicas=args.replicas,
            constrain_atoms_idx=constrain_atoms_idx,
            restart=args.restart,
            fep_calc=fep_calc,
            dtype=args.dtype,
            no_mixing=args.no_mixing
        )
    elif args.mode in ["NEQAbsolute", "NEQRelative"]:
        sampler = NonEquilibriumSwitching(
            mace_model=args.model_path,
            ligA_idx=ligA_idx,
            ligB_idx=ligB_idx,
            steps_per_iter=args.steps_per_iter,
            constrain_atoms_idx=constrain_atoms_idx,
            xyz_file=args.file,
            dtype=args.dtype,
            output_dir=args.output,
            reverse=args.reverse,
            fep_calc=fep_calc,
            restart = args.restart,
            equilibrate = args.equilibrate,
            interval=args.report_interval
            )
    else:
        raise ValueError(f"Did not recognise mode {args.mode}")


    if args.minimise and not args.restart:
        sampler.minimise()
    sampler.run()


if __name__ == "__main__":
    main()
