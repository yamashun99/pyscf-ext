from pyscf.cc.ccsd import CCSD
from pyscf.cc import eom_rccsd
from pyscf.lib import logger
import numpy as np
from pyscf import lib


def kernel_left(
    eom,
    nroots=1,
    koopmans=False,
    guess=None,
    left=False,
    eris=None,
    imds=None,
    **kwargs,
):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(eom.stdout, eom.verbose)
    if eom.verbose >= logger.WARN:
        eom.check_sanity()
    eom.dump_flags()

    if imds is None:
        imds = eom.make_imds(eris)

    matvec, diag = eom.gen_matvec(imds, left=left, **kwargs)

    size = eom.vector_size()
    nroots = min(nroots, size)
    if guess is not None:
        user_guess = True
        for g in guess:
            assert g.size == size
    else:
        user_guess = False
        guess = eom.get_init_guess(nroots, koopmans, diag)

    def precond(r, e0, x0):
        return r / (e0 - diag + 1e-12)

    # GHF or customized RHF/UHF may be of complex type
    real_system = eom._cc._scf.mo_coeff[0].dtype == np.double

    eig = lib.davidson_nosym1
    if user_guess or koopmans:
        assert len(guess) == nroots

        def eig_close_to_init_guess(w, v, nroots, envs):
            x0 = lib.linalg_helper._gen_x0(envs["v"], envs["xs"])
            s = np.dot(np.asarray(guess).conj(), np.asarray(x0).T)
            snorm = np.einsum("pi,pi->i", s.conj(), s)
            idx = np.argsort(-snorm)[:nroots]
            return lib.linalg_helper._eigs_cmplx2real(w, v, idx, real_system)

        conv, es, vs, lvs = eig(
            matvec,
            guess,
            precond,
            pick=eig_close_to_init_guess,
            tol=eom.conv_tol,
            max_cycle=eom.max_cycle,
            max_space=eom.max_space,
            nroots=nroots,
            verbose=log,
            left=left,
        )
    else:

        def pickeig(w, v, nroots, envs):
            real_idx = np.where(abs(w.imag) < 1e-3)[0]
            return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)

        conv, es, vs, lvs = eig(
            matvec,
            guess,
            precond,
            pick=pickeig,
            tol=eom.conv_tol,
            max_cycle=eom.max_cycle,
            max_space=eom.max_space,
            nroots=nroots,
            verbose=log,
            left=left,
        )

    if eom.verbose >= logger.INFO:
        for n, en, vn, convn in zip(range(nroots), es, vs, conv):
            r1, r2 = eom.vector_to_amplitudes(vn)
            if isinstance(r1, np.ndarray):
                qp_weight = np.linalg.norm(r1) ** 2
            else:  # for EOM-UCCSD
                r1 = np.hstack([x.ravel() for x in r1])
                qp_weight = np.linalg.norm(r1) ** 2
            logger.info(
                eom,
                "EOM-CCSD root %d E = %.16g  qpwt = %.6g  conv = %s",
                n,
                en,
                qp_weight,
                convn,
            )
        log.timer("EOM-CCSD", *cput0)
    if nroots == 1:
        return conv[0], es[0].real, vs[0], lvs[0]
    else:
        return conv, es.real, vs, lvs


class CCSDLeft(CCSD):
    def eomee_ccsd_singlet(
        self, nroots=1, koopmans=False, guess=None, eris=None, imds=None, diag=None
    ):
        eom = EOMEESingletLeft(self)
        eom.converged, eom.e, eom.v, eom.lv = eom.kernel(
            nroots, koopmans, guess, eris=eris, imds=imds, diag=diag, left=True
        )
        return eom.e, eom.v, eom.lv


class EOMEESingletLeft(eom_rccsd.EOMEESinglet):
    kernel = kernel_left
