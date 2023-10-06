import argparse

from mace_fep.replica_exchange.replica_exchange import ReplicaExchange
import logging
import os

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
    args = parser.parse_args()
    logger = logging.getLogger("mace_fep")
    logger.setLevel(log_level[args.log_level])
    # parser.add_argument("--idx", type=int, nargs="+")

    ligA_idx = [i for i in range(0, 6)]
    ligB_idx = [i for i in range(6, 15)]
    # ligB_idx = []

    # make the output dir if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    sampler = ReplicaExchange(
        mace_model=args.model_path,
        output_dir=args.output,
        iters=args.iters,
        steps_per_iter=args.steps_per_iter,
        xyz_file=args.file,
        replicas=args.replicas,
        ligA_idx=ligA_idx,
        ligB_idx=ligB_idx,
        restart=args.restart,
    )

    if args.minimise:
        sampler.minimise()
    sampler.run()


if __name__ == "__main__":
    main()
