import astropy.units as u

class Material:
    def __init__(self, name: str, Z_A: float, rho: u.Quantity, I: u.Quantity, n: float, X0: float, delta_file: str,  att_file: str):
        """
        Initialize a material with the required properties.

        :param name: Name of the material (e.g., "Aluminum")
        :param Z_A: Average <Z/A>. Atomic mass A has units g/mol 
        :param rho: Density of the material in g/cm³ (e.g., 2.7 g/cm³ for Aluminum)
        :param I: Mean excitation potential in eV (e.g., 166 eV for Aluminum)
        :param n: Refractive index 
        :papam X0: Radiation length
        :param delta_file: Path to the file with density effect parameter 
        :param att_file: Path to the file with mass attenuation factors for Compton scattering and pair production 
        """
        self.name = name
        self.Z_A = Z_A.to(u.mol / u.g)  # Average <Z/A>. Atomic mass A has units g/mol 
        self.rho = rho.to(u.g / u.cm**3)  # Density
        self.I = I.to(u.eV)  # Mean excitation energy
        self.n = n
        self.X0 = X0
        self.delta_file = delta_file
        self.att_file = att_file

    def __str__(self):
        return f"Material(name={self.name}, Z_A={self.Z_A}, rho={self.rho}, I={self.I}), n={self.n}"