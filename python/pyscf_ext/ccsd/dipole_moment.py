import numpy as np


def gamma_value_f_from_0(l, t, ncore):
    l1, l2 = l
    t1, t2 = t
    l2_Ikab = l2[:ncore, :, :, :]
    l_tilde_tilde_Ij = np.einsum("Ikab,jkab->Ij", l2_Ikab, t2) / 2
    l1_Ia = l1[:ncore, :]
    gamma_jI = -np.einsum("Ij->jI", l_tilde_tilde_Ij) - np.einsum(
        "Ia, ja -> jI", l1_Ia, t1
    )
    gamm_aI = np.einsum("Ia->aI", l1_Ia)

    return gamma_jI, gamm_aI


def gamma_value_0_from_f(l, r, t, ncore):
    l1, l2 = l
    r1, r2 = r
    t1, t2 = t
    r2_Jkab = r2[:ncore, :, :, :]
    l_tilde_iJ = np.einsum("ikab,Jkab->iJ", l2, r2_Jkab) / 2
    r1_Ja = r1[:ncore, :]
    gamma_Ji = -np.einsum("iJ->Ji", l_tilde_iJ) - np.einsum("ia,Ja -> Ji", l1, r1_Ja)
    r2_Ijab = r2[:ncore, :, :, :]
    l_tilde_tilde_ab = np.einsum("ijac,ijbc->ab", l2, t2) / 2
    l_tilde_iJ = np.einsum("ikab,Jkab->iJ", l2, r2_Jkab) / 2
    gamma_Ia = (
        r1_Ja
        + np.einsum("Ijab,jb->Ia", (r2_Ijab - np.einsum("Ib,ja->Ijab", r1_Ja, t1)), l1)
        - np.einsum("ca,Ic->Ia", l_tilde_tilde_ab, r1_Ja)
        - np.einsum("jI,ja->Ia", l_tilde_iJ, t1)
    )
    return gamma_Ji, gamma_Ia


def f_f_from_0(
    root_index,  # 励起状態のインデックス f
    ccsd,
    l_lambda,
    rvs,
    lvs,
    dipole_mo,
    ncore,
    nocc,
    energies,
):
    """
    0 (基底状態) から励起状態 root_index への
    オシレーター強度 f_{0->root_index} を返す。
    """
    t = (ccsd.t1, ccsd.t2)

    # 左右振幅を t1,t2 に変換
    l1, l2 = ccsd.vector_to_amplitudes(lvs[root_index])
    r1, r2 = ccsd.vector_to_amplitudes(rvs[root_index])
    lam1, lam2 = ccsd.vector_to_amplitudes(l_lambda)

    # gamma_f_from_0 および gamma_0_from_f を計算
    gamma_f_from_0_jI, gamma_f_from_0_aI = gamma_value_f_from_0((l1, l2), t, ncore)
    gamma_0_from_f_Ji, gamma_0_from_f_Ia = gamma_value_0_from_f(
        (lam1, lam2), (r1, r2), t, ncore
    )

    # 0 -> f の遷移モーメント (x,y,z 成分) : mu_f_from_0_alpha
    mu_f_from_0_alpha = np.einsum(
        "bjI,jI->b", dipole_mo[:, :nocc, :ncore], gamma_f_from_0_jI
    ) + np.einsum("baI,aI->b", dipole_mo[:, nocc:, :ncore], gamma_f_from_0_aI)

    # f -> 0 の遷移モーメント : mu_0_from_f_alpha
    mu_0_from_f_alpha = np.einsum(
        "bJi,Ji->b", dipole_mo[:, :ncore, :nocc], gamma_0_from_f_Ji
    ) + np.einsum("bIa,Ia->b", dipole_mo[:, :ncore, nocc:], gamma_0_from_f_Ia)

    # エネルギー差 energies[root_index] を使ってオシレーター強度を計算
    # fval = (2/3) * ΔE * (mu_f · mu_0_from_f)
    fval = (
        2.0
        / 3.0
        * energies[root_index]
        * np.einsum("b,b", mu_f_from_0_alpha, mu_0_from_f_alpha)
    )
    return fval
