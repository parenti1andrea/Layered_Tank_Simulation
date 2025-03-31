import numpy as np
import astropy.units as u
from landaupy import landau

from .particle import Particle
from .constants import MUON_MASS, ELECTRON_MASS, K_BB
from .material import Material

class Muon(Particle):
    # Fixed mass for the muon in MeV

    def __init__(self, energy: u.Quantity):
        """
        Initialize a Muon particle with fixed mass.

        :param charge: Charge of the muon in elementary charge units (e)
        :param energy: Total energy of the muon in MeV
        """
        # Call the parent constructor, passing the fixed muon mass
        super().__init__(mass=MUON_MASS, charge=1, energy=energy)

    def dedx_bethe_bloch(self, material: Material):
        """
        Compute muon ionization energy loss per path length 
        Uses the Bethe-Bloch formula
        Returns in units of MeV / cm 
        """
        Z_A = material.Z_A
        rho = material.rho
        I = (material.I).to(u.MeV)

        Wmax = 2 * ELECTRON_MASS * (self.beta()*self.gamma())**2 / (1 + 2*self.gamma()*ELECTRON_MASS / MUON_MASS + (ELECTRON_MASS / MUON_MASS)**2  )

        de_dx = K_BB * Z_A * (1/self.beta()**2) * (0.5 *np.log(2*ELECTRON_MASS * self.beta()**2 * self.gamma()**2 * Wmax / I**2) - self.beta()**2 ) * rho 
        return de_dx 
    
    def de_landau(self, material: Material, dx):

        # Define function to compute energy loss from a Landau distribution at each step 
        Z_A = material.Z_A
        rho = material.rho
        I = (material.I).to(u.MeV)

        # Width of the Landau 
        width =  K_BB/2*Z_A*(rho/self.beta()**2) * np.abs(dx).to(u.cm) 

        # MPV of the Landau 
        mpv = width \
                *( np.log(2*ELECTRON_MASS*self.beta()**2*self.gamma()**2/I) \
                    +np.log(width/I) + 0.2
                             -self.beta()**2 )

        # Extract a value from the Landau distribution         
        de = landau.sample(x_mpv=mpv.value, xi=width.value, n_samples=1) 
        return de * u.MeV
    
    
    def dedx(self, material):
        return self.dedx_bethe_bloch(material)  # Use Bethe-Bloch for muons

    def __str__(self):
        return f"Muon(mass={self.mass}, charge={self.charge} e, energy={self.energy})"

