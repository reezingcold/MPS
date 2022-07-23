'''
introducion: this package is created for time evolution via TDVP method.
'''

from .mps import MPS
from .tool_functions import rgetHbond, svd_bdc, update_leftenv, update_rightenv
from .tool_functions import lngetHeff, getHeffs, rngetHeff, getHeff, lgetHbond
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import expm_multiply

class oneTDVP(object):
    def __init__(self, initial_state: MPS, H, begin_time: float, dt: float):
        '''
        @initial_state: initial state in the form of mps
        @H: a function which returns Hamiltonian(MPO) over time
        @dt: time segment = evolution time/number of steps
        '''
        self.initial_state = initial_state.copy()
        self.initial_state.normalization()
        self.current_state = initial_state.copy()
        self.H = H
        self.begin_time = begin_time
        self.current_time = begin_time
        self.dt = dt
        self.current_H = H(self.current_time)
        if self.current_H.site_number != initial_state.site_number:
            raise Exception("hamiltonian and state should have same length.")
        else:
            self.N = initial_state.site_number
        #self.current_state.normalize(0)
        #generate all the right environment we need
        right_env = [self.current_state.transfer_matrix(x, self.current_H.get_data(x)) for x in range(1, self.N)]
        self.right = [0]*(self.N-1)
        self.right[self.N-2] = right_env[self.N-2]
        current_right = self.right[self.N-2]
        for i in range(self.N-3, -1, -1):
            current_right = right_env[i]@current_right
            self.right[i] = current_right
        self.left = [0]*(self.N-1)
    
    # for the very left side tensor, there is only LtoR sweep
    def evolve_leftside(self):
        state = self.current_state
        H = self.current_H
        if state.isGaugeCenter(0) == True:
            pass
        else:
            state.normalize(0)
        # evolve site tensor, with actual dt = dt/2
        right = self.right[0]
        a, b = state.get_dim(0)
        i, j, k = H.get_dim(0)
        right = np.reshape(right, (b, i, b))
        Heff = np.einsum('ijk,eig->jekg', H.get_data(0), right, optimize=True)
        Heff = np.reshape(Heff, (a*b, a*b))
        site_tensor = np.reshape(state.get_data(0), (a*b, 1))
        #evolved_tensor = linalg.expm(-1j*Heff*self.dt/2)@site_tensor
        evolved_tensor = expm_multiply(-1j*self.dt/2*Heff, site_tensor)
        evolved_tensor = np.reshape(evolved_tensor, (a, b))
        # evole bond back, also dt = dt/2
        Q, R = linalg.qr(evolved_tensor, mode='economic')
        state.replace(Q, 0)
        Qh = Q.conjugate()
        Heff = np.reshape(Heff, (a, b, a, b))
        Hbond = np.einsum('ab,ceag,cd->debg', Q, Heff, Qh, optimize=True)
        Hbond = np.reshape(Hbond, (min(a, b)*b, min(a, b)*b))
        R = np.reshape(R, (min(a, b)*b, 1))
        # evolved_bond = linalg.expm(1j*Hbond*self.dt/2)@R
        evolved_bond = expm_multiply(1j*self.dt/2*Hbond, R)
        evolved_bond = np.reshape(evolved_bond, (min(a, b), b))
        next_tensor = np.einsum('ab,bef->aef', evolved_bond, state.get_data(1), optimize=True)
        state.replace(next_tensor, 1) 
        self.left[0] = state.transfer_matrix(0, H.get_data(0), False)
    
    def evolve_middle_LtoR(self, site):
        state = self.current_state
        H = self.current_H
        left = self.left[site-1]
        # check whether we are at gauge center
        if state.isGaugeCenter(site) == True:
            pass
        else:
            state.normalize(site)
        # evolve site tensor, with dt = dt/2
        right = self.right[site]
        a, b, c = state.get_dim(site)
        i, j, k, l = H.get_dim(site)
        right = np.reshape(right, (c, j, c))
        #Heff = np.einsum('gim,ijkl,pjr->gkpmlr', left, H.get_data(site), right, optimize=True)
        Heff = getHeff(left, H.get_data(site), right)
        Heff = np.reshape(Heff, (a*k*c, a*l*c))
        site_tensor = np.reshape(state.get_data(site), (a*b*c, 1))
        #evolved_tensor = linalg.expm(-1j*Heff*self.dt/2)@site_tensor
        evolved_tensor = expm_multiply(-1j*self.dt/2*Heff, site_tensor)
        # evole bond back, also dt = dt/2
        evolved_tensor = np.reshape(evolved_tensor, (a*b, c))
        Q, R = linalg.qr(evolved_tensor, mode='economic')
        Q = np.reshape(Q, (a, b, min(a*b, c)))
        state.replace(Q, site)
        Qh = Q.conjugate()
        Heff = np.reshape(Heff, (a, b, c, a, b, c))
        #Hbond = np.einsum('abc,depabr,def->fpcr', Q, Heff, Qh, optimize=True)
        Hbond = lgetHbond(Q, Heff, Qh)
        Hbond = np.reshape(Hbond, (min(a*b, c)*c, min(a*b, c)*c))
        R = np.reshape(R, (min(a*b, c)*c, 1))
        #evolved_bond = linalg.expm(1j*Hbond*self.dt/2)@R
        evolved_bond = expm_multiply(1j*self.dt/2*Hbond, R)
        evolved_bond = np.reshape(evolved_bond, (min(a*b, c), c))
        if site != self.N-2:
            next_tensor = np.einsum('da,abc->dbc', evolved_bond, state.get_data(site+1), optimize=True)
        else:
            next_tensor = np.einsum('da,ab->db', evolved_bond, state.get_data(site+1), optimize=True)
        state.replace(next_tensor, site+1)
        self.left[site] = update_leftenv(left, H.get_data(site), Q)
    
    def evolve_rightside(self):
        state = self.current_state
        H = self.current_H
        if state.isGaugeCenter(self.N-1) == True:
            pass
        else:
            state.normalize(self.N-1)
        # evolve site tensor, with actual dt = dt
        left = self.left[self.N-2]
        a, b = state.get_dim(self.N-1)
        i, j, k = H.get_dim(self.N-1)
        Heff = np.einsum('eig,ijk->ejgk', left, H.get_data(self.N-1), optimize=True)
        Heff = np.reshape(Heff, (a*b, a*b))
        site_tensor = state.get_data(self.N-1)
        site_tensor = np.reshape(site_tensor, (a*b, 1))
        #evolved_tensor = linalg.expm(-1j*Heff*self.dt)@site_tensor
        evolved_tensor = expm_multiply(-1j*self.dt*Heff, site_tensor)
        evolved_tensor = np.reshape(evolved_tensor, (a, b))
        # evole bond back, also dt = dt/2
        R, Q = linalg.rq(evolved_tensor, mode='economic')
        state.replace(Q, self.N-1)
        Qh = Q.conjugate()
        Heff = np.reshape(Heff, (a, b, a, b))
        Hbond = np.einsum('ak,ejgk,cj->ecga', Q, Heff, Qh, optimize=True)
        Hbond = np.reshape(Hbond, (a*min(a, b), a*min(a, b)))
        R = np.reshape(R, (a*min(a, b), 1))
        #evolved_bond = linalg.expm(1j*Hbond*self.dt/2)@R
        evolved_bond = expm_multiply(1j*self.dt/2*Hbond, R)
        evolved_bond = np.reshape(evolved_bond, (a, min(a, b)))
        last_tensor = np.einsum('dea,ab->deb', state.get_data(self.N-2), evolved_bond, optimize=True)
        state.replace(last_tensor, self.N-2)
        self.right[self.N-2] = state.transfer_matrix(self.N-1, H.get_data(self.N-1), False)
    
    def evolve_middle_RtoL(self, site):
        state = self.current_state
        H = self.current_H
        if state.isGaugeCenter(site) == True:
            pass
        else:
            state.normalize(site)
        # evolve site tensor, with actual dt = dt/2
        left = self.left[site-1]
        a, b, c = state.get_dim(site)
        i, j, k, l = H.get_dim(site)
        left = np.reshape(left, (a, i, a))
        right = self.right[site]
        #Heff = np.einsum('gim,ijkl,njq->gknmlq', left, H.get_data(site), right, optimize=True)
        Heff = getHeff(left, H.get_data(site), right)
        Heff = np.reshape(Heff, (a*k*c, a*l*c))
        site_tensor = np.reshape(state.get_data(site), (a*b*c, 1))
        #evolved_tensor = linalg.expm(-1j*Heff*self.dt/2)@site_tensor
        evolved_tensor = expm_multiply(-1j*self.dt/2*Heff, site_tensor)
        evolved_tensor = np.reshape(evolved_tensor, (a, b*c))
        # evole bond back, also dt = dt/2
        R, Q = linalg.rq(evolved_tensor, mode='economic')
        Q = np.reshape(Q, (min(a, b*c), b, c))
        state.replace(Q, site)
        Qh = Q.conjugate()
        Heff = np.reshape(Heff, (a, k, c, a, l, c))
        #Hbond = np.einsum('alq,gknmlq,dkn->gdma', Q, Heff, Qh, optimize=True)
        Hbond = rgetHbond(Q, Heff, Qh)
        Hbond = np.reshape(Hbond, (min(a, b*c)*a, min(a, b*c)*a))
        R = np.reshape(R, (min(a, b*c)*a, 1))
        #evolved_bond = linalg.expm(1j*Hbond*self.dt/2)@R
        evolved_bond = expm_multiply(1j*self.dt/2*Hbond, R)
        evolved_bond = np.reshape(evolved_bond, (a, min(a, b*c)))
        if site != 1:
            last_tensor = np.einsum('dea,ab->deb', state.get_data(site-1), evolved_bond, optimize=True)
        else:
            last_tensor = np.einsum('da,ab->db', state.get_data(site-1), evolved_bond, optimize=True)
        state.replace(last_tensor, site-1)
        #self.right[site-1] = np.einsum('dial,l->dia', state.transfer_matrix(site, H.get_data(site), 'r'), np.reshape(right, (c*j*c, )), optimize=True)
        self.right[site-1] = update_rightenv(right, H.get_data(site), Q)

    def evolve_leftside_end(self):
        state = self.current_state
        H = self.current_H
        if state.isGaugeCenter(0) == True:
            pass
        else:
            state.normalize(0)
        # evolve site tensor, with actual dt = dt/2
        right = self.right[0]
        a, b = state.get_dim(0)
        i, j, k = H.get_dim(0)
        Heff = np.einsum('ijk,eig->jekg', H.get_data(0), right, optimize=True)
        Heff = np.reshape(Heff, (a*b, a*b))
        site_tensor = state.get_data(0)
        site_tensor = np.reshape(site_tensor, (a*b, 1))
        #evolved_tensor = linalg.expm(-1j*Heff*self.dt/2)@site_tensor
        evolved_tensor = expm_multiply(-1j*self.dt/2*Heff, site_tensor)
        evolved_tensor = np.reshape(evolved_tensor, (a, b))
        state.replace(evolved_tensor, 0)

    def evolve_a_step(self):
        self.evolve_leftside()
        for i in range(1, self.N-1):
            self.evolve_middle_LtoR(i)
        self.evolve_rightside()
        for j in range(self.N-2, 0, -1):
            self.evolve_middle_RtoL(j)
        self.evolve_leftside_end()
        self.current_time += self.dt
        self.current_H = self.H(self.current_time)

#------------------------------------------------------------------------------------------------------#

class twoTDVP(object):
    def __init__(self, initial_state: MPS, H, begin_time: float, dt: float, bond_D: int):
        '''
        @initial_state: initial state in the form of mps
        @H: a function which returns Hamiltonian(MPO) over time
        @dt: time segment = (evolution time)/(number of steps)
        '''
        self.initial_state = initial_state.copy()
        self.initial_state.normalization()
        self.current_state = initial_state.copy()
        self.current_state.bond_D = bond_D
        self.H = H
        self.begin_time = begin_time
        self.current_time = begin_time
        self.dt = dt
        self.current_H = self.H(self.current_time)
        self.bond_D = bond_D
        if self.current_H.site_number != initial_state.site_number:
            raise Exception("hamiltonian and state should have same length.")
        else:
            self.N = initial_state.site_number
        # generate all the right environment we need at this time step
        right_env = [self.current_state.transfer_matrix(x, self.current_H.get_data(x)) for x in range(2, self.N)]
        self.right = [0]*(self.N-2)
        self.right[self.N-3] = right_env[self.N-3]
        current_right = self.right[self.N-3]
        for i in range(self.N-4, -1, -1):
            current_right = right_env[i]@current_right
            self.right[i] = current_right
        self.left = [0]*(self.N-2)
    
    # for the very left side tensor, only RtoL sweep
    def evolve_leftside(self):
        state = self.current_state
        H = self.current_H
        if state.isGaugeCenter(0) == True:
            pass
        else:
            state.normalize(0)
        #evolve site 0 and site 1 together with actual dt = dt/2
        right = self.right[0]
        a, b = state.get_dim(0)
        e, f, g = state.get_dim(1)
        i, j, k = H.get_dim(0)
        w, x, y, z = H.get_dim(1)
        right = np.reshape(right, (g, x, g))
        Heffs = np.einsum('ijk,ixyz,pxr->jypkzr', H.get_data(0), H.get_data(1), right, optimize=True)
        Heffs = np.reshape(Heffs, (a*f*g, a*f*g))
        sites_tensor = np.einsum('ab,bfg->afg', state.get_data(0), state.get_data(1), optimize=True)
        sites_tensor = np.reshape(sites_tensor, (a*f*g, 1))
        #evolved_sites_tensor = linalg.expm(-1j*Heffs*self.dt/2)@sites_tensor
        evolved_sites_tensor = expm_multiply(-1j*Heffs*self.dt/2, sites_tensor)
        evolved_sites_tensor = np.reshape(evolved_sites_tensor, (a, f*g))
        U, Smat, V, trunc = svd_bdc(evolved_sites_tensor, self.bond_D)
        state.replace(U, 0)
        Uh = U.conjugate()
        Heffs = np.reshape(Heffs, (a, f, g, a, f, g))
        Heff = np.einsum('ab,cypazr,cd->dypbzr', U, Heffs, Uh, optimize=True)
        Heff = np.reshape(Heff, (trunc*f*g, trunc*f*g))
        #original_site_tensor = linalg.expm(1j*Heff*self.dt/2)@np.reshape(Smat@V, (trunc*f*g, 1))
        original_site_tensor = expm_multiply(1j*Heff*self.dt/2, np.reshape(Smat@V, (trunc*f*g, 1)))
        original_site_tensor = np.reshape(original_site_tensor, (trunc, f, g))
        state.replace(original_site_tensor, 1)
        self.left[0] = state.transfer_matrix(0, H.get_data(0), False)
    
    def evolve_LtoR(self, site):
        state = self.current_state
        H = self.current_H
        left = self.left[site-1]
        # check whether we are at gauge center
        if state.isGaugeCenter(site) == True:
            pass
        else:
            state.normalize(site)
        # evolve site tensor, with delta t = dt/2
        right = self.right[site]
        a, b, c = state.get_dim(site)
        g, h, m = state.get_dim(site+1)
        i, j, k, l = H.get_dim(site)
        w, x, y, z = H.get_dim(site+1)
        right = np.reshape(right, (m, x, m))
        #Heffs = np.einsum('tuv,ujkl,jxyz,qxs->tkyqvlzs', left, H.get_data(site), H.get_data(site+1), right, optimize=True)
        Heffs = getHeffs(left, H.get_data(site), H.get_data(site+1), right)
        Heffs = np.reshape(Heffs, (a*k*y*m, a*l*z*m))
        sites_tensor = np.einsum('abc,chm->abhm', state.get_data(site), state.get_data(site+1), optimize=True)
        sites_tensor = np.reshape(sites_tensor, (a*b*h*m, 1))
        #evolved_sites_tensor = linalg.expm(-1j*Heffs*self.dt/2)@sites_tensor
        evolved_sites_tensor = expm_multiply(-1j*Heffs*self.dt/2, sites_tensor)
        evolved_sites_tensor = np.reshape(evolved_sites_tensor, (a*b, h*m))
        U, Smat, V, trunc = svd_bdc(evolved_sites_tensor, self.bond_D)
        evolved_tensor = np.reshape(U, (a, b, trunc))
        state.replace(evolved_tensor, site)
        Heffs = np.reshape(Heffs, (a, k, y, m, a, l, z, m))
        #Heff = np.einsum('abc,tkyqabzs,tkf->fyqczs', evolved_tensor, Heffs, evolved_tensor.conjugate(), optimize=True)
        Heff = lngetHeff(evolved_tensor, Heffs)
        Heff = np.reshape(Heff, (trunc*y*m, trunc*z*m))
        #original_site_tensor = linalg.expm(1j*Heff*self.dt/2)@np.reshape(Smat@V, (trunc*h*m, 1))
        original_site_tensor = expm_multiply(1j*Heff*self.dt/2, np.reshape(Smat@V, (trunc*h*m, 1)))
        original_site_tensor = np.reshape(original_site_tensor, (trunc, h, m))
        state.replace(original_site_tensor, site+1)
        #self.left[site] = np.einsum('i,ifjc->fjc', np.reshape(left, (a*i*a, )), state.transfer_matrix(site, H.get_data(site), 'left'), optimize=True)
        self.left[site] = update_leftenv(left, H.get_data(site), evolved_tensor)

    def evolve_rightside(self):
        state = self.current_state
        H = self.current_H
        left = self.left[self.N-3]
        # check whether we are at gauge center
        if state.isGaugeCenter(self.N-2) == True:
            pass
        else:
            state.normalize(self.N-2)
        # evolve site tensor self.N-2, with dt = dt/2
        a, b, c = state.get_dim(self.N-2)
        g, h = state.get_dim(self.N-1)
        i, j, k, l = H.get_dim(self.N-2)
        x, y, z = H.get_dim(self.N-1)
        Heffs = np.einsum('qrs,rjkl,jyz->qkyslz', left, H.get_data(self.N-2), H.get_data(self.N-1), optimize=True)
        Heffs = np.reshape(Heffs, (a*k*y, a*l*z))
        sites_tensor = np.einsum('abc,ch->abh', state.get_data(self.N-2), state.get_data(self.N-1), optimize=True)
        sites_tensor = np.reshape(sites_tensor, (a*b*h, 1))
        #evolved_sites_tensor = linalg.expm(-1j*Heffs*self.dt/2)@sites_tensor
        evolved_sites_tensor = expm_multiply(-1j*Heffs*self.dt/2, sites_tensor)
        evolved_sites_tensor = np.reshape(evolved_sites_tensor, (a*b, h))
        U, Smat, V, trunc = svd_bdc(evolved_sites_tensor, self.bond_D)
        evolved_tensor = np.reshape(U, (a, b, trunc))
        state.replace(evolved_tensor, self.N-2)
        #Heffs = np.reshape(Heffs, (a, k, y, a, l, z))
        #Heff = np.einsum('abc,qkyabz,qkf->fycz', evolved_tensor, Heffs, evolved_tensor.conjugate())
        #Heff = np.reshape(Heff, (trunc*y, trunc*z))
        #original_site_tensor = linalg.expm(1j*Heff*self.dt/2)@np.reshape(Smat@V, (a*b*h, 1))
        #original_site_tensor = np.reshape(original_site_tensor, (a*b, h))
        state.replace(Smat@V, self.N-1)
        # evolve site tensor self.N-1, with dt = dt/2
        a, b, c = state.get_dim(self.N-2)
        g, h = state.get_dim(self.N-1)
        i, j, k, l = H.get_dim(self.N-2)
        x, y, z = H.get_dim(self.N-1)
        Heffs = np.einsum('qrs,rjkl,jyz->qkyslz', left, H.get_data(self.N-2), H.get_data(self.N-1), optimize=True)
        Heffs = np.reshape(Heffs, (a*k*y, a*l*z))
        sites_tensor = np.einsum('abc,ch->abh', state.get_data(self.N-2), state.get_data(self.N-1), optimize=True)
        sites_tensor = np.reshape(sites_tensor, (a*b*h, 1))
        evolved_sites_tensor = expm_multiply(-1j*Heffs*self.dt/2, sites_tensor)
        evolved_sites_tensor = np.reshape(evolved_sites_tensor, (a*b, h))
        U, Smat, V, trunc = svd_bdc(evolved_sites_tensor, self.bond_D)
        evolved_tensor = np.reshape(V, (trunc, h))
        state.replace(evolved_tensor, self.N-1)
        Vh = V.conjugate()
        Heffs = np.reshape(Heffs, (a, k, y, a, l, z))
        Heff = np.einsum('gh,qkyslh,ny->qknslg', V, Heffs, Vh, optimize=True)
        Heff = np.reshape(Heff, (a*b*h, a*b*h))
        original_site_tensor = expm_multiply(1j*Heff*self.dt/2, np.reshape(U@Smat, (a*b*trunc, 1)))
        original_site_tensor = np.reshape(original_site_tensor, (a, b, trunc))
        state.replace(original_site_tensor, self.N-2)
        self.right[self.N-3] = state.transfer_matrix(self.N-1, H.get_data(self.N-1), False)
    
    def evolve_RtoL(self, site):
        state = self.current_state
        H = self.current_H
        right = self.right[site-1]
        # check whether we are at gauge center
        if state.isGaugeCenter(site) == True:
            pass
        else:
            state.normalize(site)
        # evolve site tensor, with dt = dt/2
        left = self.left[site-2]
        a, b, c = state.get_dim(site-1)
        g, h, m = state.get_dim(site)
        i, j, k, l = H.get_dim(site-1)
        w, x, y, z = H.get_dim(site)
        left = np.reshape(left, (a, i, a))
        #Heffs = np.einsum('tuv,ujkl,jxyz,qxs->tkyqvlzs', left, H.get_data(site-1), H.get_data(site), right, optimize=True)
        Heffs = getHeffs(left, H.get_data(site-1), H.get_data(site), right)
        Heffs = np.reshape(Heffs, (a*k*y*m, a*l*z*m))
        sites_tensor = np.einsum('abc,chm->abhm', state.get_data(site-1), state.get_data(site), optimize=True)
        sites_tensor = np.reshape(sites_tensor, (a*b*h*m, 1))
        evolved_sites_tensor = expm_multiply(-1j*Heffs*self.dt/2, sites_tensor)
        evolved_sites_tensor = np.reshape(evolved_sites_tensor, (a*b, h*m))
        U, Smat, V, trunc = svd_bdc(evolved_sites_tensor, self.bond_D)
        evolved_tensor = np.reshape(V, (trunc, h, m))
        state.replace(evolved_tensor, site)
        Heffs = np.reshape(Heffs, (a, k, y, m, a, l, z, m))
        #Heff = np.einsum('ghm,tkyqvlhm,nyq->tknvlg', evolved_tensor, Heffs, evolved_tensor.conjugate(), optimize=True)
        Heff = rngetHeff(evolved_tensor, Heffs)
        Heff = np.reshape(Heff, (a*k*trunc, a*l*trunc))
        original_site_tensor = expm_multiply(1j*Heff*self.dt/2, np.reshape(U@Smat, (a*b*trunc, 1)))
        original_site_tensor = np.reshape(original_site_tensor, (a, b, trunc))
        state.replace(original_site_tensor, site-1)
        #self.right[site-2] = np.einsum('nwgx,x->nwg', state.transfer_matrix(site, H.get_data(site), 'right'), np.reshape(right, (m*x*m, )), optimize=True)
        self.right[site-2] = update_rightenv(right, H.get_data(site), evolved_tensor)
    
    def evolve_leftside_again(self):
        state = self.current_state
        H = self.current_H
        right = self.right[0]
        # check whether we are at gauge center
        if state.isGaugeCenter(1) == True:
            pass
        else:
            state.normalize(1)
        a, b = state.get_dim(0)
        e, f, g = state.get_dim(1)
        i, j, k = H.get_dim(0)
        w, x, y, z = H.get_dim(1)
        right = np.reshape(right, (g, x, g))
        Heffs = np.einsum('ijk,ixyz,pxr->jypkzr', H.get_data(0), H.get_data(1), right, optimize=True)
        Heffs = np.reshape(Heffs, (a*f*g, a*f*g))
        sites_tensor = np.einsum('ab,bfg->afg', state.get_data(0), state.get_data(1), optimize=True)
        sites_tensor = np.reshape(sites_tensor, (a*f*g, 1))
        evolved_sites_tensor = expm_multiply(-1j*Heffs*self.dt/2, sites_tensor)
        evolved_sites_tensor = np.reshape(evolved_sites_tensor, (a, f*g))
        U, Smat, V, trunc = svd_bdc(evolved_sites_tensor, self.bond_D)
        evolved_tensor = np.reshape(V, (trunc, f, g))
        state.replace(evolved_tensor, 1)
        #Heffs = np.reshape(Heffs, (a, f, g, a, f, g))
        #Heff = np.einsum('efg,jypkfg,hyp->jhke', evolved_tensor, Heffs, evolved_tensor.conjugate())
        #Heff = np.reshape(Heff, (a*f*g, a*f*g))
        #original_site_tensor = linalg.expm(1j*Heff*self.dt/2)@np.reshape(U@Smat, (a*f*g, 1))
        #original_site_tensor = np.reshape(original_site_tensor, (a, f*g))
        state.replace(U@Smat, 0)
        #evolve site tensor 0 again
        """
        a, b = state.get_dim(0)
        e, f, g = state.get_dim(1)
        i, j, k = H.get_dim(0)
        w, x, y, z = H.get_dim(1)
        Heffs = np.reshape(Heffs, (a*f*g, a*f*g))
        sites_tensor = np.einsum('ab,bfg->afg', state.get_data(0), state.get_data(1))
        sites_tensor = np.reshape(sites_tensor, (a*f*g, 1))
        evolved_sites_tensor = linalg.expm(-1j*Heffs*self.dt/2)@sites_tensor
        evolved_sites_tensor = np.reshape(evolved_sites_tensor, (a, f*g))
        U, Sval, V = linalg.svd(evolved_sites_tensor)
        Sval = Sval[:min(self.bond_D, len(Sval))]
        Smat = linalg.diagsvd(Sval, a, f*g)
        state.replace(U, 0)
        Uh = U.conjugate()
        Heffs = np.reshape(Heffs, (a, f, g, a, f, g))
        Heff = np.einsum('ab,cypazr,cd->dypbzr', U, Heffs, Uh)
        Heff = np.reshape(Heff, (a*f*g, a*f*g))
        original_site_tensor = linalg.expm(1j*Heff*self.dt/2)@np.reshape(Smat@V, (a*f*g, 1))
        original_site_tensor = np.reshape(original_site_tensor, (a, f, g))
        state.replace(original_site_tensor, 1)
        # return to right gauge
        state.update()
        state.normalize_site_R(1)
        """
    
    def evolve_a_step(self):
        self.evolve_leftside()
        for i in range(1, self.N-2):
            self.evolve_LtoR(i)
        self.evolve_rightside()
        for j in range(self.N-2, 1, -1):
            self.evolve_RtoL(j)
        self.evolve_leftside_again()
        self.current_state.update()
        self.current_time += self.dt
        self.current_H = self.H(self.current_time)