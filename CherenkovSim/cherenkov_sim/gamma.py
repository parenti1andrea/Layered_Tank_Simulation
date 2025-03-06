import numpy as np
import astropy.units as u
from .particle import Particle
from .constants import MUON_MASS, ELECTRON_MASS, K_BB
from .material import Material

class Gamma(Particle):
    # Fixed mass for the muon in MeV

    def __init__(self, energy: u.Quantity):
        """
        Initialize a Gamma particle with no mass.

        :param charge: Charge of the particle in elementary charge units (e)
        :param energy: Total energy of the photon in MeV
        """
        # Call the parent constructor, passing the fixed muon mass
        super().__init__(mass=0 * u.GeV, charge=0., energy=energy)

   
    def __str__(self):
        return f"Gamma(mass={self.mass}, charge={self.charge} e, energy={self.energy})"

