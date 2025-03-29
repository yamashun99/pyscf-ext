from cvs_ccsd import CCSDCVS
import numpy as np


def fccvsccsd(
    mf,
    ncore,
):
    nocc = int(mf.mo_occ.sum() // 2)
    nvir = mf.mo_coeff.shape[1] - nocc

    # コアフローズンする軌道数
    ncore = 1

    # ccsd_frozen でフローズン CCSD を実行
    ccsd_frozen = CCSDCVS(mf).run(frozen=ncore)

    t1_frozen = ccsd_frozen.t1
    t2_frozen = ccsd_frozen.t2

    # フローズンで得られた振幅をフルサイズに埋め戻し
    t1_restored = np.zeros((nocc, nvir))
    t2_restored = np.zeros((nocc, nocc, nvir, nvir))
    t1_restored[ncore:, :] = t1_frozen
    t2_restored[ncore:, ncore:, :, :] = t2_frozen

    # 新しく CCSD インスタンスを作り直し，そこへ t1, t2 を設定
    ccsd = CCSDCVS(mf)
    ccsd.t1 = t1_restored
    ccsd.t2 = t2_restored

    return ccsd
