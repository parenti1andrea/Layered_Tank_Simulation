import numpy as np
import astropy.units as u
from .particle import Particle
from .constants import ELECTRON_MASS, K_BB
from .material import Material

class Electron(Particle):
    # Fixed mass for the muon in MeV

    def __init__(self, energy: u.Quantity):
        """
        Initialize an Electron particle with fixed mass.

        :param charge: Charge of the muon in elementary charge units (e)
        :param energy: Total energy of the muon in MeV
        """
        # Call the parent constructor, passing the fixed muon mass
        super().__init__(mass=ELECTRON_MASS, charge=1, energy=energy)

        self._e_delta = None
        self._delta_factor = None 

    def load_delta_parameter(self, material: Material):
        self._e_delta , self._delta_factor = np.loadtxt(material.delta_file, skiprows=8, usecols=(0,1), unpack=True) 


    def dedx_ion(self, material: Material):
        """
        Compute electron ionization energy loss per path length
        Returns in units of MeV / cm 
        """
        Z_A = material.Z_A
        rho = material.rho
        I = (material.I).to(u.MeV)

        delta = np.interp(self.kinetic_energy().value, self._e_delta, self._delta_factor) # Interplate table values 

        de_dx =  0.5 * K_BB *Z_A / self.beta()**2 * ( np.log(ELECTRON_MASS.to(u.MeV) * self.beta()**2 * self.gamma()**2 * (ELECTRON_MASS.to(u.MeV) *(self.gamma() -1)/2.)/I**2) + 
                             (1 - self.beta()**2) - (2*self.gamma()-1)/self.gamma()**2 *np.log(2) + 1/8. * ((self.gamma()-1)/self.gamma())**2 - delta) * rho

        return de_dx

    def dedx_rad(self, material: Material):
        """
        Compute electron radiative energy loss per path length 
        Bremsstrahlung
        Returns in units of MeV / cm 
        """
        X0 = material.X0
     
        de_dx = self.energy.to(u.MeV) / X0 
        return de_dx 

    def dedx_tot(self, material: Material):
        return self.dedx_rad(material) + self.dedx_ion(material)

    def dedx(self, material):
        return self.dedx_tot(material)  

    def __str__(self):
        return f"Electron (mass={self.mass}, charge={self.charge} e, energy={self.energy})"
