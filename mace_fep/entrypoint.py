import argparse

from mace_fep.replica_exchange.replica_exchange import ReplicaExchange
import logging

logger = logging.getLogger("mace_fep")
logger.setLevel(logging.INFO)


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
    # parser.add_argument("--idx", type=int, nargs="+")
    args = parser.parse_args()

    ligA_idx = [i for i in range(0, 6)]
    ligB_idx = [i for i in range(6, 15)]

    sampler = ReplicaExchange(
        mace_model=args.model_path,
        storage_file=args.output,
        iters=args.iters,
        steps_per_iter=args.steps_per_iter,
        xyz_file=args.file,
        replicas=args.replicas,
        ligA_idx=ligA_idx,
        ligB_idx=ligB_idx,
        minimise=args.minimise,
    )



    sampler.run()


if __name__ == "__main__":
    main()
