from copy import deepcopy

from ase.io.bundletrajectory import shutil
from mace_fep.lambda_schedule import LambdaSchedule
from mace_fep.calculators import EQ_MACE_AFE_Calculator, NEQ_MACE_AFE_Calculator
from mace_fep.protocols import NonEquilibriumSwitching, ReplicaExchange
from mace_fep.replica import Replica
import pytest
import os
from ase.io import read
import netCDF4 as nc
import torch
import numpy as np

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


@pytest.mark.parametrize("use_ssc", [True, False])
def test_run_neq(use_ssc: bool):
    os.makedirs("junk", exist_ok=True)
    lambda_schedule = LambdaSchedule(start=0.0, delta=0.1, n_steps=10, use_ssc=use_ssc)

    fep_calc = NEQ_MACE_AFE_Calculator(
        model_path=os.path.join(TEST_DIR, "l1_swa.model"),
        ligA_idx=[0, 1, 2, 3, 4, 5],
        lambda_schedule=lambda_schedule,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    atoms = read(os.path.join(TEST_DIR, "methanol_solvated.xyz"))
    atoms.set_calculator(fep_calc)

    sampler = NonEquilibriumSwitching(
        atoms=atoms,
        total_steps=10,
        constrain_atoms_idx=[],
        output_dir="junk",
    )

    sampler.propagate()

    assert os.path.isfile("junk/output_replica_0.xyz")
    assert np.isclose(sampler.integrator.atoms.calc.lambda_schedule.current_lambda, 1.1)


def ssc_lambda(lmbda: float) -> float:
    return 6 * lmbda**5 - 15 * lmbda**4 + 10 * lmbda**3


@pytest.mark.parametrize("use_ssc", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
def test_linear_lambda_schedule(use_ssc: bool, reverse: bool):
    if reverse:
        start = 1.0
        delta = -0.1
    else:
        start = 0.0
        delta = 0.1

    schedule = LambdaSchedule(start=start, delta=delta, n_steps=10, use_ssc=use_ssc)
    output_values = np.linspace(0, 1, 11) if not reverse else np.linspace(1, 0, 11)

    if use_ssc:
        output_values = [ssc_lambda(i) for i in output_values]

    for i in range(10):
        next(schedule)

        assert np.isclose(schedule.output_lambda, output_values[i])


@pytest.mark.parametrize("no_mixing", [True, False])
def test_repex(no_mixing):
    atoms = read(os.path.join(TEST_DIR, "methanol_solvated.xyz"))
    os.makedirs("junk", exist_ok=True)
    replicas = []

    for idx, l in enumerate(np.linspace(0, 1, 4)):
        atoms = deepcopy(atoms)
        calc = EQ_MACE_AFE_Calculator(
            model_path=os.path.join(TEST_DIR, "l1_swa.model"),
            ligA_idx=[0, 1, 2, 3, 4, 5],
            default_dtype="float32",
            device="cuda" if torch.cuda.is_available() else "cpu",
            l=l,
        )
        atoms.set_calculator(calc)
        replicas.append(
            Replica(atoms=atoms, idx=idx, l=l, output_dir="junk", total_steps=100)
        )

    sampler = ReplicaExchange(
        output_dir="junk",
        iters=10,
        steps_per_iter=10,
        replicas=replicas,
        restart=False,
        dtype=torch.float32,
        no_mixing=no_mixing,
    )
    sampler.propagate()

    assert np.isfinite(sampler.current_free_energy)



def test_repex_restart():
    atoms = read(os.path.join(TEST_DIR, "aniline.xyz"))
    os.makedirs("./junk/repex", exist_ok=True)
    shutil.copyfile(os.path.join(TEST_DIR, "repex.nc"), "./junk/repex/repex.nc")

    replicas = []

    for idx, l in enumerate(np.linspace(0, 1, 4)):
        atoms = deepcopy(atoms)
        calc = EQ_MACE_AFE_Calculator(
            model_path=os.path.join(TEST_DIR, "l1_swa.model"),
            ligA_idx=[0, 1, 2, 3, 4, 5],
            default_dtype="float32",
            device="cuda" if torch.cuda.is_available() else "cpu",
            l=l,
        )
        atoms.set_calculator(calc)
        replicas.append(
            Replica(atoms=atoms, idx=idx, l=l, output_dir="junk", total_steps=100)
        )

    sampler = ReplicaExchange(
        output_dir="./junk/repex",
        iters=10,
        steps_per_iter=10,
        replicas=replicas,
        restart=True,
        dtype=torch.float32,
        no_mixing=False,
    )


    # check energies match the final energies of the repex file


    ref_dataset = nc.Dataset(os.path.join(TEST_DIR, "repex.nc"), "r")
    print(ref_dataset.variables)
    ref_energies = ref_dataset.variables["u_kln"][:].transpose(1, 2, 0)[:,:,-1]
    ref_positions = ref_dataset.variables["positions"][-1, :, :, :] #[iter, replica, atom, xyz]
    ref_lambdas = ref_dataset.variables["lambda_vals"][:]
    ref_velocitiies = ref_dataset.variables["velocities"][-1, :, :, :] #[iter, replica, atom, xyz]


    print(sampler.energies_last_iteration)

    assert np.allclose(ref_energies, sampler.energies_last_iteration, atol=1e-5) 


    for replica, ref_pos in zip(sampler.replicas, ref_positions):
        print(replica.atoms.positions.shape, ref_pos.shape)
        assert np.allclose(replica.atoms.positions, ref_pos, atol=1e-5)

    for replica, ref_vel in zip(sampler.replicas, ref_velocitiies):
        assert np.allclose(replica.atoms.get_velocities(), ref_vel, atol=1e-5)

    restart_lambdas = [replica.atoms.calc.get_lambda() for replica in sampler.replicas]
    print(restart_lambdas, ref_lambdas)

    assert np.allclose(ref_lambdas, np.array(restart_lambdas), atol=1e-5)

    
    




