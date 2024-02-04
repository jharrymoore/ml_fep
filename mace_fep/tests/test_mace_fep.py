import pytest
import os
from mace_fep.entrypoint import entrypoint
from mace_fep.replica_exchange

@pytest.mark.parametrize("mode", ["EQAbsolute", "EQRelative", "NEQAbsolute", "NEQRelative"])
def test_entrypoint(mode, ligA_idx, ligB_idx, reverse: bool,  ):
    pass
