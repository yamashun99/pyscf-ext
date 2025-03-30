from pyscf.cc.ccsd import CCSD
from pyscf.cc.eom_uccsd import EOMEE, EOMEESpinKeep, EOMEESpinFlip
from pyscf.cc import eom_rccsd
from pyscf.lib import logger
import numpy as np
from pyscf import lib
from pyscf_ext.left_rccsd.ccsd_left import kernel_left


def eomee_ccsd(
    eom, nroots=1, koopmans=False, guess=None, eris=None, imds=None, diag=None
):
    if eris is None:
        eris = eom._cc.ao2mo()
    if imds is None:
        imds = eom.make_imds(eris)
    eom.converged, eom.e, eom.v, eom.lv = kernel_left(
        eom, nroots, koopmans, guess, imds=imds, diag=diag, left=True
    )
    return eom.e, eom.v, eom.lv


def eomsf_ccsd(
    eom, nroots=1, koopmans=False, guess=None, eris=None, imds=None, diag=None
):
    """Spin flip EOM-EE-CCSD"""
    return eomee_ccsd(eom, nroots, koopmans, guess, eris, imds, diag)


class EOMEESpinKeepLeft(EOMEESpinKeep):
    kernel = eomee_ccsd


class EOMEESpinFlipLeft(EOMEESpinFlip):
    kernel = eomsf_ccsd


def eeccsd_left(eom, nroots=1, koopmans=False, guess=None, eris=None, imds=None):
    """Calculate N-electron neutral excitations via EOM-EE-CCSD.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested
        koopmans : bool
            Calculate Koopmans'-like (1p1h) excitations only, targeting via
            overlap.
        guess : list of ndarray
            List of guess vectors to use for targeting via overlap.
    """
    if eris is None:
        eris = eom._cc.ao2mo()
    if imds is None:
        imds = eom.make_imds(eris)

    spinvec_size = eom.vector_size()
    nroots = min(nroots, spinvec_size)

    diag_ee, diag_sf = eom.get_diag(imds)
    guess_ee = []
    guess_sf = []
    if guess and guess[0].size == spinvec_size:
        raise NotImplementedError
        # TODO: initial guess from GCCSD EOM amplitudes
        # from pyscf.cc import addons
        # from pyscf.cc import eom_gccsd
        # orbspin = scf.addons.get_ghf_orbspin(eris.mo_coeff)
        # nmo = np.sum(eom.nmo)
        # nocc = np.sum(eom.nocc)
        # for g in guess:
        #    r1, r2 = eom_gccsd.vector_to_amplitudes_ee(g, nmo, nocc)
        #    r1aa = r1[orbspin==0][:,orbspin==0]
        #    r1ab = r1[orbspin==0][:,orbspin==1]
        #    if abs(r1aa).max() > 1e-7:
        #        r1 = addons.spin2spatial(r1, orbspin)
        #        r2 = addons.spin2spatial(r2, orbspin)
        #        guess_ee.append(eom.amplitudes_to_vector(r1, r2))
        #    else:
        #        r1 = spin2spatial_eomsf(r1, orbspin)
        #        r2 = spin2spatial_eomsf(r2, orbspin)
        #        guess_sf.append(amplitudes_to_vector_eomsf(r1, r2))
        #    r1 = r2 = r1aa = r1ab = g = None
        # nroots_ee = len(guess_ee)
        # nroots_sf = len(guess_sf)
    elif guess:
        for g in guess:
            if g.size == diag_ee.size:
                guess_ee.append(g)
            else:
                guess_sf.append(g)
        nroots_ee = len(guess_ee)
        nroots_sf = len(guess_sf)
    else:
        dee = np.sort(diag_ee)[:nroots]
        dsf = np.sort(diag_sf)[:nroots]
        dmax = np.sort(np.hstack([dee, dsf]))[nroots - 1]
        nroots_ee = np.count_nonzero(dee <= dmax)
        nroots_sf = np.count_nonzero(dsf <= dmax)
        guess_ee = guess_sf = None

    def eomee_sub(cls, nroots, guess, diag):
        ee_sub = cls(eom._cc)
        ee_sub.__dict__.update(eom.__dict__)
        e, v, lv = ee_sub.kernel(nroots, koopmans, guess, eris, imds, diag=diag)
        if nroots == 1:
            e, v, lv = [e], [v], [lv]
            ee_sub.converged = [ee_sub.converged]
        return list(ee_sub.converged), list(e), list(v), list(lv)

    e0 = e1 = []
    v0 = v1 = []
    lv0 = lv1 = []
    conv0 = conv1 = []
    if nroots_ee > 0:
        conv0, e0, v0, lv0 = eomee_sub(EOMEESpinKeepLeft, nroots_ee, guess_ee, diag_ee)
    if nroots_sf > 0:
        conv1, e1, v1, lv1 = eomee_sub(EOMEESpinFlipLeft, nroots_sf, guess_sf, diag_sf)

    e = np.hstack([e0, e1])
    idx = e.argsort()
    e = e[idx]
    conv = conv0 + conv1
    conv = [conv[x] for x in idx]
    v = v0 + v1
    v = [v[x] for x in idx]
    lv = lv0 + lv1
    lv = [lv[x] for x in idx]

    if nroots == 1:
        conv = conv[0]
        e = e[0]
        v = v[0]
        lv = lv[0]
    eom.converged = conv
    eom.e = e
    eom.v = v
    eom.lv = lv
    return eom.e, eom.v, eom.lv


class EOMEELeft(EOMEE):

    kernel = eeccsd_left
