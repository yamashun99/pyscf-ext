import numpy as np


def calculate_gamma_f_from_0(l, t, ncore):
    """
    Calculate gamma_jI and gamma_aI from the given l and t for the 0->f transition.
    """
    l1, l2 = l
    t1, t2 = t

    # l2_Ikab and gamma_jI calculation
    l2_Ikab = l2[:ncore, :, :, :]
    l_tilde_tilde_Ij = np.einsum("Ikab,jkab->Ij", l2_Ikab, t2) / 2
    gamma_jI = -np.einsum("Ij->jI", l_tilde_tilde_Ij)

    # l1_Ia and gamma_aI calculation
    l1_Ia = l1[:ncore, :]
    gamma_jI -= np.einsum("Ia, ja -> jI", l1_Ia, t1)
    gamm_aI = np.einsum("Ia->aI", l1_Ia)

    return gamma_jI, gamm_aI


def calculate_gamma_0_from_f(l, r, t, ncore):
    """
    Calculate gamma_Ji and gamma_Ia from the given l, r, and t for the f->0 transition.
    """
    l1, l2 = l
    r1, r2 = r
    t1, t2 = t

    # Calculation of gamma_Ji
    r2_Jkab = r2[:ncore, :, :, :]
    l_tilde_iJ = np.einsum("ikab,Jkab->iJ", l2, r2_Jkab) / 2
    r1_Ja = r1[:ncore, :]
    gamma_Ji = -np.einsum("iJ->Ji", l_tilde_iJ) - np.einsum("ia,Ja -> Ji", l1, r1_Ja)

    # Calculation of gamma_Ia
    r2_Ijab = r2[:ncore, :, :, :]
    l_tilde_tilde_ab = np.einsum("ijac,ijbc->ab", l2, t2) / 2
    gamma_Ia = (
        r1_Ja
        + np.einsum("Ijab,jb->Ia", (r2_Ijab - np.einsum("Ib,ja->Ijab", r1_Ja, t1)), l1)
        - np.einsum("ca,Ic->Ia", l_tilde_tilde_ab, r1_Ja)
        - np.einsum("jI,ja->Ia", l_tilde_iJ, t1)
    )

    return gamma_Ji, gamma_Ia


def f_f_from_0(root_index, ccsd, l_lambda, rvs, lvs, dipole_mo, ncore, nocc, energies):
    """
    Calculate the oscillator strength f_{0->root_index} from the ground state (0) to the excited state (root_index).
    """
    t = (ccsd.t1, ccsd.t2)

    # Convert left and right amplitudes to t1 and t2
    l1, l2 = ccsd.vector_to_amplitudes(lvs[root_index])
    r1, r2 = ccsd.vector_to_amplitudes(rvs[root_index])
    lam1, lam2 = ccsd.vector_to_amplitudes(l_lambda)

    # Calculate gamma values
    gamma_f_from_0_jI, gamma_f_from_0_aI = calculate_gamma_f_from_0((l1, l2), t, ncore)
    gamma_0_from_f_Ji, gamma_0_from_f_Ia = calculate_gamma_0_from_f(
        (lam1, lam2), (r1, r2), t, ncore
    )

    # Compute transition moments: mu_f_from_0_alpha and mu_0_from_f_alpha
    mu_f_from_0_alpha = np.einsum(
        "bjI,jI->b", dipole_mo[:, :nocc, :ncore], gamma_f_from_0_jI
    ) + np.einsum("baI,aI->b", dipole_mo[:, nocc:, :ncore], gamma_f_from_0_aI)
    mu_0_from_f_alpha = np.einsum(
        "bJi,Ji->b", dipole_mo[:, :ncore, :nocc], gamma_0_from_f_Ji
    ) + np.einsum("bIa,Ia->b", dipole_mo[:, :ncore, nocc:], gamma_0_from_f_Ia)

    # Calculate oscillator strength fval
    fval = (
        2.0
        / 3.0
        * energies[root_index]
        * np.einsum("b,b", mu_f_from_0_alpha, mu_0_from_f_alpha)
    )

    return fval
