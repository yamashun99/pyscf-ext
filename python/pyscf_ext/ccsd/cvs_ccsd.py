from pyscf.cc.ccsd import CCSD as OriginalCCSD
from pyscf.cc import eom_rccsd
from pyscf.lib import logger

import numpy as np
from custom_ccsd.lib import cvs_linalg_helper as lib


def kernel_cvs(
    self,
    nroots=1,
    koopmans=False,
    guess=None,
    eris=None,
    imds=None,
    core_excitation_mask=None,
    **kwargs,
):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(self.stdout, self.verbose)
    if self.verbose >= logger.WARN:
        self.check_sanity()
    self.dump_flags()

    if imds is None:
        imds = self.make_imds(eris)

    matvec, diag = self.gen_matvec(imds, **kwargs)

    size = self.vector_size()
    nroots = min(nroots, size)
    if guess is not None:
        user_guess = True
        for g in guess:
            assert g.size == size
    else:
        user_guess = False
        guess = self.get_init_guess(nroots, koopmans, diag)

    def precond(r, e0, x0):
        return r / (e0 - diag + 1e-12)

    # GHF or customized RHF/UHF may be of complex type
    real_system = self._cc._scf.mo_coeff[0].dtype == np.double

    eig = lib.cvs_davidson_nosym1  # カスタマイズした関数を使用
    if user_guess or koopmans:
        assert len(guess) == nroots

        def eig_close_to_init_guess(w, v, nroots, envs):
            x0 = lib._gen_x0(envs["v"], envs["xs"])
            s = np.dot(np.asarray(guess).conj(), np.asarray(x0).T)
            snorm = np.einsum("pi,pi->i", s.conj(), s)
            idx = np.argsort(-snorm)[:nroots]
            return lib._eigs_cmplx2real(w, v, idx, real_system)

        conv, es, vs = eig(
            matvec,
            guess,
            precond,
            pick=eig_close_to_init_guess,
            tol=self.conv_tol,
            max_cycle=self.max_cycle,
            max_space=self.max_space,
            nroots=nroots,
            verbose=log,
            left=True,
            core_excitation_mask=core_excitation_mask,
        )
    else:

        def pickeig(w, v, nroots, envs):
            real_idx = np.where(abs(w.imag) < 1e-3)[0]
            return lib._eigs_cmplx2real(w, v, real_idx, real_system)

        conv, es, vs, lvs = eig(
            matvec,
            guess,
            precond,
            pick=pickeig,
            tol=self.conv_tol,
            max_cycle=self.max_cycle,
            max_space=self.max_space,
            nroots=nroots,
            verbose=log,
            left=True,
            core_excitation_mask=core_excitation_mask,
        )

    if self.verbose >= logger.INFO:
        for n, en, vn, convn in zip(range(nroots), es, vs, conv):
            r1, r2 = self.vector_to_amplitudes(vn)
            if isinstance(r1, np.ndarray):
                qp_weight = np.linalg.norm(r1) ** 2
            else:  # for EOM-UCCSD
                r1 = np.hstack([x.ravel() for x in r1])
                qp_weight = np.linalg.norm(r1) ** 2
            logger.info(
                self,
                "EOM-CCSD root %d E = %.16g  qpwt = %.6g  conv = %s",
                n,
                en,
                qp_weight,
                convn,
            )

    log.timer("EOM-CCSD", *cput0)
    self.e = es
    self.v = vs
    self.conv = conv
    return self.e, self.v, lvs, self.conv


class CCSDCVS(OriginalCCSD):
    def eomee_ccsd_singlet(
        self,
        nroots=1,
        koopmans=False,
        guess=None,
        eris=None,
        ncore=1,
    ):
        eom = EOMEESingletCVS(self)
        core_excitation_mask = self.make_core_excitation_mask(ncore)
        energies, eigenvecs, lvs, conv = eom.kernel(
            nroots,
            koopmans,
            guess,
            eris,
            core_excitation_mask=core_excitation_mask,
        )
        return energies, eigenvecs, lvs, conv

    def make_core_excitation_mask(self, ncore):
        nvir = self.nmo - self.nocc

        # t1のマスクを作成
        t1_mask = np.zeros((self.nocc, nvir), dtype=bool)
        t1_mask[:ncore, :] = True

        # t2のマスクを作成
        t2_mask = np.zeros((self.nocc, self.nocc, nvir, nvir), dtype=bool)
        t2_mask[:ncore, :, :, :] = True
        t2_mask[:, :ncore, :, :] = True

        return self.amplitudes_to_vector(t1_mask, t2_mask)


class EOMEESingletCVS(eom_rccsd.EOMEESinglet):
    kernel = kernel_cvs

    def get_init_guess(self, nroots=1, koopmans=True, diag=None, ncore=1):
        """
        コア励起の部分だけ1になる初期ベクトルを生成する。

        Parameters
        ----------
        nroots : int
            取り出す初期ベクトル（根）の数
        koopmans : bool
            （元コードの名残。ここでは特に使わず、常にコア励起をターゲットとする想定）
        diag : 1D array-like
            対角要素の配列（大きさは self.vector_size() と同じ）

        Returns
        -------
        guess : list of 1D numpy.ndarray
            コア励起成分のみを 1.0 にしたベクトルを最大 nroots 個返す
        """
        import numpy as np

        if diag is None:
            diag = self.get_diag()  # 何らかの対角要素取得関数

        # CC計算における占有軌道数, 全軌道数
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc

        # ここでは "コア" を、たとえば 0 ~ ncore-1 までの占有軌道とみなす想定
        # self.ncore が無い場合は、ユーザ側で定義する必要があります

        # コア軌道 i (0 <= i < ncore) と全仮想軌道 a (0 <= a < nvir) に対して
        # t1[i,a] = 1.0, 他は0 の振動ベクトルを作り、その1次元インデックスを取得
        core_indices = []
        for i in range(ncore):
            for a in range(nvir):
                # t1, t2 をゼロ初期化
                t1 = np.zeros((nocc, nvir), dtype=diag.dtype)
                t2 = np.zeros((nocc, nocc, nvir, nvir), dtype=diag.dtype)

                t1[i, a] = 1.0  # コア軌道 i -> 仮想軌道 a の励起成分だけ1にする

                # CCインスタンス mycc の amplitudes_to_vector で1次元ベクトルへ変換
                vec_1d = self.amplitudes_to_vector(t1, t2)

                # 上記ベクトル中の "1" が立っているインデックスを調べる
                # （通常は1つだけのはず）
                idx_nonzero = np.nonzero(vec_1d)[0]
                if len(idx_nonzero) == 1:
                    core_indices.append(idx_nonzero[0])
                else:
                    # 万が一複数や0個の場合、想定外なのでログを出す・あるいはスキップする等
                    pass

        # 対角要素 diag におけるコア励起インデックスを取り出し、エネルギー昇順でソート
        c_diag = diag[core_indices]
        i_sort = np.argsort(c_diag)
        # 取り出すルーツ数を調整
        nroots = min(nroots, len(i_sort))

        # ソート後に小さい順で nroots 個取り出し、それぞれに1.0を立てた初期ベクトルを作成
        size = self.vector_size()  # ベクトル全体の次元
        guess = []
        for k in i_sort[:nroots]:
            g = np.zeros(size, dtype=diag.dtype)
            g[core_indices[k]] = 1.0
            guess.append(g)

        return guess
