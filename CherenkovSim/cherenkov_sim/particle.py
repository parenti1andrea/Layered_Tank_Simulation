import numpy as np 
import astropy.units as u 

class Particle:
    def __init__(self, mass: u.Quantity, charge: float, energy: u.Quantity):
        """
        Base class for particles.

        :param mass: Mass of the particle in GeV/c^2
        :param charge: Charge of the particle in elementary charge units (e)
        :param energy: Total energy of the particle in GeV
        """
        self.mass = mass.to(u.GeV)
        self.charge = charge
        self.energy = energy.to(u.GeV)

        if self.energy < self.mass: 
            raise ValueError("Total energy cannot be less than the rest mass energy.")

    def momentum(self) -> u.Quantity:
        """
        Calculate the relativistic momentum of the particle in GeV.
        Uses the relation: p^2 = E^2 - m^2
        """
        p = np.sqrt(self.energy**2 - self.mass**2)
        return p.to(u.GeV)  # Ensuring it has correct units
    
    def kinetic_energy(self) -> u.Quantity:
        """
        Calculate the kinetic energyof the particle in GeV.
        Uses the relation: Ek = E - m
        """        
        ek = self.energy - self.mass 
        return ek.to(u.GeV)

    def beta(self) -> float:
        """
        Calculate the relativistic beta factor.
        Uses the relation: beta = p / E
        """        
        beta = self.momentum() / self.energy
        return beta 

    def gamma(self) -> float:
        """
        Calculate the relativistic gamma factor.
        Uses the relation: gamma = E / m
        """        
        if self.mass == 0 * u.MeV:
            return float('inf')  # Gamma for massless particles like photons
        else:
            return (self.energy / self.mass)


    def dedx(self, material):
        """This should be implemented in subclasses."""
        raise NotImplementedError("Energy loss function must be defined in subclasses.")

    def __str__(self):
        return f"Particle(mass={self.mass}, charge={self.charge} e, energy={self.energy})"
