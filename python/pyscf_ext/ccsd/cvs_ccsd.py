from pyscf.cc.ccsd import CCSD as OriginalCCSD
from pyscf.cc import eom_rccsd
from pyscf.lib import logger
import numpy as np
from pyscf_ext.ccsd.lib import cvs_linalg_helper as lib


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
    """
    Perform a customized EOM-CCSD calculation with a CVS approximation.
    """
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(self.stdout, self.verbose)

    if self.verbose >= logger.WARN:
        self.check_sanity()
    self.dump_flags()

    imds = imds or self.make_imds(eris)
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

    real_system = self._cc._scf.mo_coeff[0].dtype == np.double
    eig = lib.cvs_davidson_nosym1

    if user_guess or koopmans:
        assert len(guess) == nroots

        def eig_close_to_init_guess(w, v, nroots, envs):
            x0 = lib._gen_x0(envs["v"], envs["xs"])
            s = np.dot(np.asarray(guess).conj(), np.asarray(x0).T)
            snorm = np.einsum("pi,pi->i", s.conj(), s)
            idx = np.argsort(-snorm)[:nroots]
            return lib._eigs_cmplx2real(
                w, v, idx, self._cc._scf.mo_coeff[0].dtype == np.double
            )

        conv, es, vs, lvs = eig(
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
            r1_norm = (
                np.linalg.norm(r1.ravel())
                if isinstance(r1, np.ndarray)
                else np.linalg.norm(np.hstack([x.ravel() for x in r1]))
            )
            logger.info(
                self,
                "EOM-CCSD root %d E = %.16g  qpwt = %.6g  conv = %s",
                n,
                en,
                r1_norm**2,
                convn,
            )

    log.timer("EOM-CCSD", *cput0)
    self.e, self.v, self.conv = es, vs, conv
    return self.e, self.v, lvs, self.conv


class CVSCCSD(OriginalCCSD):
    def eomee_ccsd_singlet(
        self, nroots=1, koopmans=False, guess=None, eris=None, ncore=1
    ):
        """
        Perform the EOM-CCSD singlet calculation with a CVS approximation.
        """
        eom = EOMEESingletCVS(self)
        core_excitation_mask = self.make_core_excitation_mask(ncore)
        energies, eigenvecs, lvs, conv = eom.kernel(
            nroots, koopmans, guess, eris, core_excitation_mask=core_excitation_mask
        )
        return energies, eigenvecs, lvs, conv

    def make_core_excitation_mask(self, ncore):
        """
        Create a mask for core excitations up to the ncore-th occupied orbital.
        """
        nvir = self.nmo - self.nocc
        t1_mask = np.zeros((self.nocc, nvir), dtype=bool)
        t1_mask[:ncore, :] = True
        t2_mask = np.zeros((self.nocc, self.nocc, nvir, nvir), dtype=bool)
        t2_mask[:ncore, :, :, :] = True
        t2_mask[:, :ncore, :, :] = True
        return self.amplitudes_to_vector(t1_mask, t2_mask)


class EOMEESingletCVS(eom_rccsd.EOMEESinglet):
    kernel = kernel_cvs

    def get_init_guess(self, nroots=1, koopmans=True, diag=None, ncore=1):
        """
        Generate initial guess vectors with core excitations set to 1.
        """
        if diag is None:
            diag = self.get_diag()

        nocc, nmo = self.nocc, self.nmo
        nvir = nmo - nocc
        core_indices = self._get_core_indices(ncore, nocc, nvir, diag)
        c_diag = diag[core_indices]
        i_sort = np.argsort(c_diag)
        nroots = min(nroots, len(i_sort))

        size = self.vector_size()
        guess = [
            self._generate_initial_vector(core_indices, k, size, diag)
            for k in i_sort[:nroots]
        ]

        return guess

    def _get_core_indices(self, ncore, nocc, nvir, diag):
        """
        Generate the indices corresponding to core excitations.
        """
        core_indices = []
        for i in range(ncore):
            for a in range(nvir):
                t1 = np.zeros((nocc, nvir), dtype=diag.dtype)
                t2 = np.zeros((nocc, nocc, nvir, nvir), dtype=diag.dtype)
                t1[i, a] = 1.0
                vec_1d = self.amplitudes_to_vector(t1, t2)
                idx_nonzero = np.nonzero(vec_1d)[0]
                if len(idx_nonzero) == 1:
                    core_indices.append(idx_nonzero[0])
        return core_indices

    def _generate_initial_vector(self, core_indices, k, size, diag):
        """
        Generate an initial guess vector with a core excitation set to 1.
        """
        g = np.zeros(size, dtype=diag.dtype)
        g[core_indices[k]] = 1.0
        return g
