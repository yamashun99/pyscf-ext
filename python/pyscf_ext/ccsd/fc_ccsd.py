from pyscf_ext.ccsd.cvs_ccsd import CVSCCSD
import numpy as np


def fccvsccsd(mf, ncore):
    """
    Perform a frozen-core CCSD (Coupled-Cluster with Single and Double excitations) calculation
    and restore the amplitudes to the full size for further CCSD calculations.

    This function performs a frozen-core CCSD calculation using the specified ncore frozen orbitals
    and restores the resulting amplitudes to the full size, which are then used to initialize a new CCSD instance.

    Parameters:
    -----------
    mf : pyscf.scf.hf.RHF
        Mean-field object obtained from a Hartree-Fock calculation.
    ncore : int
        Number of core orbitals to freeze.

    Returns:
    --------
    ccsd : CCSDCVS
        A CCSDCVS instance with the restored amplitudes for a full CCSD calculation.

    Notes:
    ------
    This function freezes the core orbitals specified by ncore and performs the CCSD calculation accordingly.
    The resulting t1 and t2 amplitudes are restored to the full size, accounting for the frozen orbitals.
    """
    nocc = int(mf.mo_occ.sum() // 2)
    nvir = mf.mo_coeff.shape[1] - nocc

    # Perform frozen-core CCSD calculation
    ccsd_frozen = CVSCCSD(mf).run(frozen=ncore)

    # Get the frozen amplitudes
    t1_frozen, t2_frozen = ccsd_frozen.t1, ccsd_frozen.t2

    # Restore the frozen amplitudes to full size
    t1_restored = np.zeros((nocc, nvir))
    t2_restored = np.zeros((nocc, nocc, nvir, nvir))
    t1_restored[ncore:, :] = t1_frozen
    t2_restored[ncore:, ncore:, :, :] = t2_frozen

    # Create a new CCSD instance and set the restored amplitudes
    ccsd = CVSCCSD(mf)
    ccsd.t1, ccsd.t2 = t1_restored, t2_restored

    return ccsd
