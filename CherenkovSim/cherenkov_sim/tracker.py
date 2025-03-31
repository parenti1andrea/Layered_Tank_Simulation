import numpy as np
import astropy.units as u
import copy
from scipy.stats import poisson
from .particle import Particle
from .muon import Muon
from .electron import Electron
from .gamma import Gamma
from .material import Material
from .constants import ALPHA, R_TANK, H_TANK, ELECTRON_MASS, WORLD_R, WORLD_Z

class MuonTracker:
    def __init__(self, particle: Particle, material: Material, 
                 step_size: u.Quantity = 1.0 * u.cm,
                 position = np.array([0.0, 0.0, 0.0]) * u.cm, 
                 zenith: u.Quantity = 0.0 * u.deg, 
                 azimuth: u.Quantity = 0.0 * u.deg,
                 is_landau = True, 
                 is_verbose = False):
        """
        Initializes the muon tracking system.

        :param muon: The Muon object to track.
        :param material: The material the muon is traveling through.
        :param step_size: The step size for tracking (default is 1 cm).
        :param position: The initial position as a NumPy array or tuple (x, y, z) in cm.
        :param zenith: The zenith angle (θ) in degrees (0° = downward, 90° = horizontal).
        :param azimuth: The azimuth angle (φ) in degrees (0° = x-axis, 90° = y-axis).
        """
        self.particle = particle
        self.material = material
        self.step_size = step_size

        self.is_landau = is_landau

        self.is_verbose = is_verbose

        # Ensure position is a NumPy array and has units
        if isinstance(position, tuple) or isinstance(position, list):
            position = np.array(position)

        if not isinstance(position, np.ndarray):
            raise ValueError("Position must be a NumPy array or a tuple of (x, y, z).")

        if not hasattr(position, "unit"):
            position *= u.cm  # Assume cm if no units are provided

        self.position = position

        self.x = self.position[0]
        self.y = self.position[1]
        self.z = self.position[2]
        self.r = np.sqrt(self.x**2 + self.y**2)
        # Compute the direction vector from zenith and azimuth angles
        self.direction = self._compute_direction(zenith, azimuth)

        self.track = []  # Store position and energy history
        self.energy_track = []  # Store position and energy history
        self.cherenkov_photons = []
        self.cherenkov_photons_z = []
        self.cherenkov_photons_r = []
        self.path_length = []


    def _compute_direction(self, zenith, azimuth):
        """
        Converts zenith (θ) and azimuth (φ) angles into a unit direction vector.

        :param zenith: Angle from vertical (0° = downward, 90° = horizontal).
        :param azimuth: Angle from x-axis (0° = x, 90° = y, 180° = -x, 270° = -y).
        :return: A NumPy array representing the direction vector.
        """
        theta = zenith.to(u.rad).value  # Convert degrees to radians
        phi = azimuth.to(u.rad).value  # Convert degrees to radians

        # Convert spherical coordinates to Cartesian unit vector
        direction = np.array([
            np.sin(theta) * np.cos(phi),  # x-component
            np.sin(theta) * np.sin(phi),  # y-component
            np.cos(theta)                # z-component
        ])
        return direction / np.linalg.norm(direction)  # Ensure it's a unit vector]
    
    def _update_position(self, displacement):
        self.position += displacement
        self.x = self.position[0]
        self.y = self.position[1]
        self.z = self.position[2]
        self.r = np.sqrt(self.x**2 + self.y**2)

    def _is_in_tank(self):
        r_tank = R_TANK
        h_tank = H_TANK
        if self.z >= 0. * u.cm  and self.z <= h_tank and self.r <= r_tank: 
            return True 
        else: 
            return False 
      
    def cherenkov_yield(self, lam1,lam2): 
        return (2.*np.pi * ALPHA * (1. / lam1 - 1. / lam2) * ( 1. - 1. / (self.particle.beta()**2 * self.material.n**2 )) )
    
    def generate_cherenkov_photons(self, lambda1, lambda2):
        """
        Generates Cherenkov photons along a step based on the Poisson distribution.

        :param lambda1: Lower wavelength bound (e.g., 300 nm)
        :param lambda2: Upper wavelength bound (e.g., 600 nm)
        """
        # Compute expected Cherenkov photons 
        expected_photons = self.cherenkov_yield(lambda1.to(u.cm), lambda2.to(u.cm)) * np.abs( self.step_size.to(u.cm) )
        # Extract photons from a Poissonian distrubution 
        num_photons = poisson.rvs(expected_photons)

        if num_photons > 0: 
            # Distribute photon positions along the step 
            rng = np.random.default_rng()  
            fractions = rng.uniform(0, 1, size=num_photons)  # Faster random sampling

            photon_positions = self.position + np.outer(fractions, self.direction * self.step_size)
            photon_positions_x = self.position[0].value + np.outer(fractions, self.direction[0] * self.step_size.value)
            photon_positions_y = self.position[1].value + np.outer(fractions, self.direction[1] * self.step_size.value)
            photon_positions_z = self.position[2].value + np.outer(fractions, self.direction[2] * self.step_size.value)

            photon_positions_r = np.sqrt(photon_positions_x**2 + photon_positions_y**2)

            # Store photon positions
            self.cherenkov_photons.extend(photon_positions)
            self.cherenkov_photons_z.extend( photon_positions_z.flatten() )
            self.cherenkov_photons_r.extend( photon_positions_r.flatten() )


    def propagate(self, energy_threshold=0.1 * u.MeV, lambda1 = 300 * u.nm, lambda2 = 600 * u.nm):
        """
        Propagates the muon step by step until it loses all energy.

        :param energy_threshold: The minimum energy before stopping the muon.
        """
        if(self.is_verbose):print(f"Starting particle tracking at position {self.position} with energy {self.particle.energy:.4f}")

        l_tank = 0 * u.cm 

        while self.particle.energy > energy_threshold and self.z > 0 * u.cm and self.z < WORLD_Z and self.r < WORLD_R :

            # Update position using direction vector
            # Attention! Sign of step size determines the direction of propagation 
            displacement = self.direction * self.step_size
            self._update_position(displacement)

            # Print step info for debugging
            # print(f"Step: Position {self.position}, Energy {self.particle.energy:.4f}")
 

            while ( self.particle.beta() > 1./self.material.n ) and self._is_in_tank(): 

                l_tank += np.linalg.norm(displacement)

                # Compute energy loss in this step
                self.generate_cherenkov_photons(lambda1, lambda2)  # Generate Cherenkov photons

                if(self.is_landau):
                    energy_loss = self.particle.de_landau(self.material, self.step_size )

                else:
                    dE_dx = self.particle.dedx(self.material)  # Energy loss per cm
                    energy_loss = dE_dx * np.abs(self.step_size)
                
                self.particle.energy -= energy_loss  # Reduce muon energy

                if self.particle.energy <= 0:  # If energy is depleted, stop tracking
                    break

                # Save step in track history
                self.track.append(self.position.copy())
                self.energy_track.append(self.particle.energy.copy())

                # Print step info for debugging
                #print(f"Step: Position {self.position}, Energy {self.particle.energy:.4f}")

                displacement = self.direction * self.step_size
                self._update_position(displacement)

                #print(self.particle.momentum())

        if(self.is_verbose):print("Tracking complete. Particle stopped.")
        self.path_length.append(l_tank.value)




class ParticleTracker:
    def __init__(self, particle: Particle, material: Material, 
                 step_size: u.Quantity = 1.0 * u.cm,
                 position = np.array([0.0, 0.0, 0.0]) * u.cm, 
                 zenith: u.Quantity = 0.0 * u.deg, 
                 azimuth: u.Quantity = 0.0 * u.deg,
                 is_verbose = False):
        """
        Initializes the muon tracking system.

        :param muon: The Muon object to track.
        :param material: The material the muon is traveling through.
        :param step_size: The step size for tracking (default is 1 cm).
        :param position: The initial position as a NumPy array or tuple (x, y, z) in cm.
        :param zenith: The zenith angle (θ) in degrees (0° = downward, 90° = horizontal).
        :param azimuth: The azimuth angle (φ) in degrees (0° = x-axis, 90° = y-axis).
        """
        self.particle = particle
        self.material = material
        self.step_size = step_size

        self.is_verbose = is_verbose

        # Ensure position is a NumPy array and has units
        if isinstance(position, tuple) or isinstance(position, list):
            position = np.array(position)

        if not isinstance(position, np.ndarray):
            raise ValueError("Position must be a NumPy array or a tuple of (x, y, z).")

        if not hasattr(position, "unit"):
            position *= u.cm  # Assume cm if no units are provided

        self.position = position

        self.x = self.position[0]
        self.y = self.position[1]
        self.z = self.position[2]
        self.r = np.sqrt(self.x**2 + self.y**2)
        # Compute the direction vector from zenith and azimuth angles
        self.direction = self._compute_direction(zenith, azimuth)

        self.track = []  # Store position and energy history
        self.energy_track = []  # Store position and energy history
        self.cherenkov_photons = []
        self.cherenkov_photons_z = []
        self.cherenkov_photons_r = []
        self.path_length = []


    def _compute_direction(self, zenith, azimuth):
        """
        Converts zenith (θ) and azimuth (φ) angles into a unit direction vector.

        :param zenith: Angle from vertical (0° = downward, 90° = horizontal).
        :param azimuth: Angle from x-axis (0° = x, 90° = y, 180° = -x, 270° = -y).
        :return: A NumPy array representing the direction vector.
        """
        theta = zenith.to(u.rad).value  # Convert degrees to radians
        phi = azimuth.to(u.rad).value  # Convert degrees to radians

        # Convert spherical coordinates to Cartesian unit vector
        direction = np.array([
            np.sin(theta) * np.cos(phi),  # x-component
            np.sin(theta) * np.sin(phi),  # y-component
            np.cos(theta)                # z-component
        ])
        return direction / np.linalg.norm(direction)  # Ensure it's a unit vector]
    
    def _update_position(self, displacement):
        self.position += displacement
        self.x = self.position[0]
        self.y = self.position[1]
        self.z = self.position[2]
        self.r = np.sqrt(self.x**2 + self.y**2)

    def _is_in_tank(self):
        r_tank = R_TANK
        h_tank = H_TANK
        if self.z >= 0. * u.cm  and self.z <= h_tank and self.r <= r_tank: 
            return True 
        else: 
            return False 
      
    def cherenkov_yield(self, lam1,lam2): 
        return (2.*np.pi * ALPHA * (1. / lam1 - 1. / lam2) * ( 1. - 1. / (self.particle.beta()**2 * self.material.n**2 )) )
    
    def generate_cherenkov_photons(self, lambda1, lambda2):
        """
        Generates Cherenkov photons along a step based on the Poisson distribution.

        :param lambda1: Lower wavelength bound (e.g., 300 nm)
        :param lambda2: Upper wavelength bound (e.g., 600 nm)
        """
        # Compute expected Cherenkov photons 
        expected_photons = self.cherenkov_yield(lambda1.to(u.cm), lambda2.to(u.cm)) * np.abs( self.step_size.to(u.cm) )
        # Extract photons from a Poissonian distrubution 
        num_photons = poisson.rvs(expected_photons)

        if num_photons > 0: 
            # Distribute photon positions along the step 
            rng = np.random.default_rng()  
            fractions = rng.uniform(0, 1, size=num_photons)  # Faster random sampling

            photon_positions = self.position + np.outer(fractions, self.direction * self.step_size)
            photon_positions_x = self.position[0].value + np.outer(fractions, self.direction[0] * self.step_size.value)
            photon_positions_y = self.position[1].value + np.outer(fractions, self.direction[1] * self.step_size.value)
            photon_positions_z = self.position[2].value + np.outer(fractions, self.direction[2] * self.step_size.value)

            photon_positions_r = np.sqrt(photon_positions_x**2 + photon_positions_y**2)

            # Store photon positions
            self.cherenkov_photons.extend(photon_positions)
            self.cherenkov_photons_z.extend( photon_positions_z.flatten() )
            self.cherenkov_photons_r.extend( photon_positions_r.flatten() )


    def propagate(self, energy_threshold=0.1 * u.MeV, lambda1 = 300 * u.nm, lambda2 = 600 * u.nm):
        """
        Propagates the muon step by step until it loses all energy.

        :param energy_threshold: The minimum energy before stopping the muon.
        """
        if(self.is_verbose):print(f"Starting particle tracking at position {self.position} with energy {self.particle.energy:.4f}")

        l_tank = 0 * u.cm 

        while self.particle.energy > energy_threshold and self.z > 0 * u.cm and self.z < WORLD_Z and self.r < WORLD_R :

            # Update position using direction vector
            # Attention! Sign of step size determines the direction of propagation 
            displacement = self.direction * self.step_size
            self._update_position(displacement)

            # Print step info for debugging
            # print(f"Step: Position {self.position}, Energy {self.particle.energy:.4f}")
 

            while ( self.particle.beta() > 1./self.material.n ) and self._is_in_tank(): 

                l_tank += np.linalg.norm(displacement)

                dE_dx = self.particle.dedx(self.material)  # Energy loss per cm

                # Compute energy loss in this step
                self.generate_cherenkov_photons(lambda1, lambda2)  # Generate Cherenkov photons
                energy_loss = dE_dx * np.abs(self.step_size)
                self.particle.energy -= energy_loss  # Reduce muon energy

                if self.particle.energy <= 0:  # If energy is depleted, stop tracking
                    break

                # Save step in track history
                self.track.append(self.position.copy())
                self.energy_track.append(self.particle.energy.copy())

                # Print step info for debugging
                #print(f"Step: Position {self.position}, Energy {self.particle.energy:.4f}")

                displacement = self.direction * self.step_size
                self._update_position(displacement)

        if(self.is_verbose):print("Tracking complete. Particle stopped.")
        self.path_length.append(l_tank.value)


class GammaTracker:
    def __init__(self, particle: Gamma, material: Material, 
                 step_size: u.Quantity = 1.0 * u.cm,
                 position = np.array([0.0, 0.0, 0.0]) * u.cm, 
                 zenith: u.Quantity = 0.0 * u.deg, 
                 azimuth: u.Quantity = 0.0 * u.deg,
                 is_verbose = False):
        """
        Initializes the gamma tracking system.

        :param muon: The Gamma object to track.
        :param material: The material the gamma is traveling through.
        :param step_size: The step size for tracking (default is 1 cm).
        :param position: The initial position as a NumPy array or tuple (x, y, z) in cm.
        :param zenith: The zenith angle (θ) in degrees (0° = downward, 90° = horizontal).
        :param azimuth: The azimuth angle (φ) in degrees (0° = x-axis, 90° = y-axis).
        """
        self.particle = particle
        self.material = material
        self.step_size = step_size

        # Ensure position is a NumPy array and has units
        if isinstance(position, tuple) or isinstance(position, list):
            position = np.array(position)

        if not isinstance(position, np.ndarray):
            raise ValueError("Position must be a NumPy array or a tuple of (x, y, z).")

        if not isinstance(particle, Gamma):
            raise ValueError("Particle must be a gamma ray")


        if not hasattr(position, "unit"):
            position *= u.cm  # Assume cm if no units are provided

        self.position = position

        self.x = self.position[0]
        self.y = self.position[1]
        self.z = self.position[2]
        self.r = np.sqrt(self.x**2 + self.y**2)
        # Compute the direction vector from zenith and azimuth angles
        self.direction = self._compute_direction(zenith, azimuth)

        self._load_att_factor()
        
        self.is_Compton = False
        self.is_pair_production = False

        self.is_verbose = is_verbose

        self.electron_trackers_queue = []

        self.track = []  # Store position and energy history
        self.energy_track = []  # Store position and energy history
        self.cherenkov_photons = []
        self.cherenkov_photons_z = []
        self.cherenkov_photons_r = []
        self.path_length = []

    def _dsigma_dx_pp(self, x):
        return (1 - 4./3 * x * (1-x))
    
    def _dsigma_dcos_compton(self, energy, cos_theta):
    # Ignore normalization constant, just care about functional shape
        E = energy 
        theta_term = 1 + ( E.to(u.keV) * (1 - cos_theta) /ELECTRON_MASS.to(u.keV)  )
        return (1/theta_term**2) * (theta_term + 1. / theta_term - (1 - cos_theta**2) )

    def _compute_direction(self, zenith, azimuth):
        """
        Converts zenith (θ) and azimuth (φ) angles into a unit direction vector.

        :param zenith: Angle from vertical (0° = downward, 90° = horizontal).
        :param azimuth: Angle from x-axis (0° = x, 90° = y, 180° = -x, 270° = -y).
        :return: A NumPy array representing the direction vector.
        """
        theta = zenith.to(u.rad).value  # Convert degrees to radians
        phi = azimuth.to(u.rad).value  # Convert degrees to radians

        # Convert spherical coordinates to Cartesian unit vector
        direction = np.array([
            np.sin(theta) * np.cos(phi),  # x-component
            np.sin(theta) * np.sin(phi),  # y-component
            np.cos(theta)                # z-component
        ])
        return direction / np.linalg.norm(direction)  # Ensure it's a unit vector]
        
    def _update_position(self, displacement):
        self.position += displacement
        self.x = self.position[0]
        self.y = self.position[1]
        self.z = self.position[2]
        self.r = np.sqrt(self.x**2 + self.y**2)

    def _is_in_tank(self):
        r_tank = R_TANK
        h_tank = H_TANK
        if self.z >= 0. * u.cm  and self.z <= h_tank and self.r <= r_tank: 
            return True 
        else: 
            return False 
        
    def _load_att_factor(self): 
        self.att_energy , self.mu_compt, self.mu_pp = np.loadtxt(self.material.att_file, skiprows=3, usecols=(0,1,2), unpack=True)
      
    def _sample_gamma_process(self):
        e_gamma = self.particle.energy.to(u.MeV)
        mu_compton = np.interp(e_gamma.value, self.att_energy, self.mu_compt) # Interpolate table values
        mu_pp = np.interp(e_gamma.value, self.att_energy, self.mu_pp) # Interpolate table values

        lambda_compt = 1 / mu_compton * u.cm
        if(mu_pp == 0): lambda_pair = 1e6 * u.cm # Set to a very high value so that the process is never sampled 
        else: lambda_pair = 1 / mu_pp * u.cm
        
        l_compt = np.random.exponential(lambda_compt.value, size=1)[0] * u.cm  
        l_pair = np.random.exponential(lambda_pair.value, size=1)[0] * u.cm  

        min = np.minimum(l_compt, l_pair)
        if(min == l_compt):
            return l_compt, "Compton"
        if(min == l_pair):
            return l_pair, "PairProduction"

    def _generate_compton_pair(self, gamma):
        E_gamma = gamma.energy 
        rng = np.random.default_rng()  

        # Extract angle of scattered gamma ray 
        def extract_costheta(ene):
            while True:
                x_value = rng.uniform(-1, 1, size=1) 
                y_value = rng.uniform(0, self._dsigma_dcos_compton(ene,1), size=1) 
          
                if y_value < self._dsigma_dcos_compton(ene,x_value):
                    return x_value[0]
                
        cos_theta = extract_costheta(E_gamma)
        sin_theta = np.sqrt(1 - cos_theta*cos_theta)
        theta_gamma = np.arccos(cos_theta)
        phi_gamma = rng.uniform(0, 1, size=1) * np.pi * 2

        if(self.is_verbose):print("Gamma-ray Compton scattering angle ",theta_gamma*180./np.pi,' deg')

        # Compute energy of scattered gamma-ray
        E_gamma_prime = E_gamma / (1 + (E_gamma.to(u.MeV)/ELECTRON_MASS.to(u.MeV) * (1 - cos_theta)) )

        # Compute energy and scattering angle of electron    
        E_elec = E_gamma - E_gamma_prime + ELECTRON_MASS
        tan_theta_elec = sin_theta / (cos_theta - E_gamma / E_gamma_prime)
        theta_elec = np.arctan(tan_theta_elec)
        phi_elec = phi_gamma + np.pi

        n_gamma = self.direction
        nx = np.array([1.,0.,0.])
        e1 = np.cross(nx, n_gamma) / np.linalg.norm(np.cross(nx, n_gamma) )
        e2 = np.cross(n_gamma, e1)

        n_gamma_prime = e1 * np.cos(phi_gamma) * np.sin(theta_gamma) + e2 * np.sin(phi_gamma) * np.sin(theta_gamma) + n_gamma * np.cos(theta_gamma)
        n_elec = e1 * np.cos(phi_elec) * np.sin(theta_elec) + e2 * np.sin(phi_elec) * np.sin(theta_elec) + n_gamma * np.cos(theta_elec)

        return E_gamma_prime, n_gamma_prime, E_elec, n_elec

    def _generate_pair(self, gamma):
        E_gamma = gamma.energy 

        rng = np.random.default_rng()  

        # frac = 0,1 are not physical, the energy should be at least the mass of the particle 

        def extract_x_fraction():
            while True:
                x_value = rng.uniform(ELECTRON_MASS/E_gamma, 1 - ELECTRON_MASS/E_gamma, size=1) 
                y_value = rng.uniform(0, 1, size=1) 
                if y_value < self._dsigma_dx_pp(x_value):
                    return x_value[0]
                
        frac = extract_x_fraction()
        if(self.is_verbose):print('Fraction of energy to electron: ',frac)
        E_e_minus = copy.deepcopy(E_gamma) * frac
        E_e_plus = copy.deepcopy(E_gamma) -  copy.deepcopy(E_e_minus)

        electron = Electron(energy = E_e_minus)
        positron = Electron(energy = E_e_plus)

        return electron, positron

    def _generate_pair_directons(self, gamma):
        E_gamma = gamma.energy 
       
        n_gamma = self.direction
        nx = np.array([1.,0.,0.])
        e1 = np.cross(nx, n_gamma) / np.linalg.norm(np.cross(nx, n_gamma) )
        e2 = np.cross(n_gamma, e1)

        theta_pp = ELECTRON_MASS.to(u.MeV) / E_gamma.to(u.MeV) * u.rad
        if(self.is_verbose):print('Opening angle electron-positron pair: ',theta_pp.to(u.deg))

        # Compute azimuthal angle on the plane perpendicular to the direction of the photon 
        rng = np.random.default_rng()  
        phi_pp = rng.uniform(0, 1, size=1) * np.pi * 2

        n_e_minus = e1 * np.cos(phi_pp) * np.sin(theta_pp.value) + e2 * np.sin(phi_pp) * np.sin(theta_pp.value) + n_gamma * np.cos(theta_pp.value)
        n_e_plus = e1 * np.cos(phi_pp + np.pi) * np.sin(theta_pp.value) + e2 * np.sin(phi_pp + np.pi) * np.sin(theta_pp.value) + n_gamma * np.cos(theta_pp.value)

        return n_e_minus, n_e_plus
    
    def propagate(self, energy_threshold=0.1 * u.MeV):
        """
        Propagates the photon step by step until in undergoes pair production.

        :param energy_threshold: The minimum energy before stopping the photon.
        """
        if(self.is_verbose):print(f"Starting particle tracking at position {self.position} with energy {self.particle.energy:.2f}")

        l_int, str_process = self._sample_gamma_process()
        if(self.is_verbose):
            print("Sampled gamma-ray interacton length: ",l_int)
            print("Sampled interacton: ",str_process)

        l_tank = 0. * u.cm # Store length crossed inside the tank  
        l_tank_tot = 0 * u.cm

        while self.particle.energy > energy_threshold and self.z > 0 * u.cm and self.z < WORLD_Z and self.r < WORLD_R :

            # Update position using direction vector
            # Attention! Sign of step size determines the direction of propagation 

            displacement = self.direction * self.step_size
            self._update_position(displacement)

            while self._is_in_tank(): 

                displacement = self.direction * self.step_size
                self._update_position(displacement)
                l_tank += np.linalg.norm(displacement)
                l_tank_tot += np.linalg.norm(displacement)

                if(self.is_verbose):
                    print("Gamma ray energy ",self.particle.energy.to(u.MeV))
                    print("Path in the tank: ",l_tank)
                    print("Interaction path: ",l_int)
                    print("Interaction process: ",str_process)

                # When path inside the tank becomes greater than interaction length, trigger Compton scattering: 
                if(l_tank > l_int and str_process == 'Compton'):
                    if(self.is_verbose): print('\n*****************\nCompton scattering happening ', self.position)
                    self.is_Compton = True

                    e_gamma, n_gamma, e_electron, n_electron = self._generate_compton_pair(self.particle)

                    electron = Electron(energy = e_electron)
                    electron.load_delta_parameter(self.material)
                    theta_electron = np.arccos(n_electron[2]) 
                    phi_electron = np.arctan2(n_electron[1],n_electron[0])
                    electron_position = copy.deepcopy(self.position)

                    if(self.is_verbose):print('Scattered electron energy ',e_electron.to(u.MeV))

                    electron_tracker = ParticleTracker(electron, self.material, self.step_size, electron_position, theta_electron, phi_electron)
                    self.electron_trackers_queue.append(electron_tracker)

                    self.particle.energy = e_gamma
                    self.direction = n_gamma

                    # Terminate tracking if gama-ray energy is below energy Cherenkov threshold for electrons 
                    e_gamma_threshold = ELECTRON_MASS * 1 / np.sqrt(1. - 1./(self.material.n**2))
                    if(e_gamma.to(u.MeV) < e_gamma_threshold.to(u.MeV)): 
                        self.path_length.append(l_tank_tot.value)
                        if(self.is_verbose):print("Gamma ray energy is ",e_gamma,' < ',e_gamma_threshold)
                        return 

                    # Reset interaction lenght and sample another process 
                    l_tank = 0 * u.cm
                    l_int, str_process = self._sample_gamma_process()
                    if(self.is_verbose):
                        print("Sampled gamma-ray interacton length: ",l_int)
                        print("Sampled interacton: ",str_process)
                                

                # When path inside the tank becomes greater than interaction length, trigger pair production: 
                if(l_tank > l_int and self.particle.energy.to(u.MeV) > 2 * ELECTRON_MASS.to(u.MeV) and str_process == 'PairProduction'):
                    if(self.is_verbose): print('Pair production at ', self.position)
                    self.is_pair_production = True
                    electron, positron = self._generate_pair(self.particle)
                    n_electron, n_positron = self._generate_pair_directons(self.particle)

                    electron.load_delta_parameter(self.material)    
                    positron.load_delta_parameter(self.material)    

                    theta_electron = np.arccos(n_electron[2]) * u.rad
                    phi_electron = np.arctan2(n_electron[1],n_electron[0])* u.rad

                    theta_positron = np.arccos(n_positron[2]) * u.rad
                    phi_positron = np.arctan2(n_positron[1],n_positron[0]) * u.rad

                    electron_position = copy.deepcopy(self.position)
                    positron_position = copy.deepcopy(self.position)

                    electron_tracker = ParticleTracker(electron, self.material, self.step_size, electron_position, theta_electron, phi_electron)
                    positron_tracker = ParticleTracker(positron, self.material, self.step_size, positron_position, theta_positron, phi_positron)

                    self.electron_trackers_queue.append(electron_tracker)
                    self.electron_trackers_queue.append(positron_tracker)

                    self.path_length.append(l_tank_tot.value)
                    return  
            

        if(self.is_verbose): print("Tracking complete. Particle stopped.")
        if(self.is_pair_production == False and self.is_Compton == False): 
            self.path_length.append(l_tank_tot.value)
            if(self.is_verbose): print("No interaction happened.")
            return None

        if(self.is_pair_production == False and self.is_Compton == True): 
            self.path_length.append(l_tank_tot.value)
            if(self.is_verbose): print("Compton scattering, then gamma ray left the tank.")
            return None

class GammaTracker_OnlyPP:
    def __init__(self, particle: Gamma, material: Material, 
                 step_size: u.Quantity = 1.0 * u.cm,
                 position = np.array([0.0, 0.0, 0.0]) * u.cm, 
                 zenith: u.Quantity = 0.0 * u.deg, 
                 azimuth: u.Quantity = 0.0 * u.deg,
                 is_verbose = False):
        """
        Initializes the gamma tracking system.

        :param muon: The Gamma object to track.
        :param material: The material the gamma is traveling through.
        :param step_size: The step size for tracking (default is 1 cm).
        :param position: The initial position as a NumPy array or tuple (x, y, z) in cm.
        :param zenith: The zenith angle (θ) in degrees (0° = downward, 90° = horizontal).
        :param azimuth: The azimuth angle (φ) in degrees (0° = x-axis, 90° = y-axis).
        """
        self.particle = particle
        self.material = material
        self.step_size = step_size

        # Ensure position is a NumPy array and has units
        if isinstance(position, tuple) or isinstance(position, list):
            position = np.array(position)

        if not isinstance(position, np.ndarray):
            raise ValueError("Position must be a NumPy array or a tuple of (x, y, z).")

        if not isinstance(particle, Gamma):
            raise ValueError("Particle must be a gamma ray")


        if not hasattr(position, "unit"):
            position *= u.cm  # Assume cm if no units are provided

        self.position = position

        self.x = self.position[0]
        self.y = self.position[1]
        self.z = self.position[2]
        self.r = np.sqrt(self.x**2 + self.y**2)
        # Compute the direction vector from zenith and azimuth angles
        self.direction = self._compute_direction(zenith, azimuth)
        
        self.is_pair_production = False

        self.is_verbose = is_verbose

        self.track = []  # Store position and energy history
        self.energy_track = []  # Store position and energy history
        self.cherenkov_photons = []
        self.cherenkov_photons_z = []
        self.cherenkov_photons_r = []

    def _dsigma_dx(self, x):
        return (1 - 4./3 * x * (1-x))

    def _compute_direction(self, zenith, azimuth):
        """
        Converts zenith (θ) and azimuth (φ) angles into a unit direction vector.

        :param zenith: Angle from vertical (0° = downward, 90° = horizontal).
        :param azimuth: Angle from x-axis (0° = x, 90° = y, 180° = -x, 270° = -y).
        :return: A NumPy array representing the direction vector.
        """
        theta = zenith.to(u.rad).value  # Convert degrees to radians
        phi = azimuth.to(u.rad).value  # Convert degrees to radians

        # Convert spherical coordinates to Cartesian unit vector
        direction = np.array([
            np.sin(theta) * np.cos(phi),  # x-component
            np.sin(theta) * np.sin(phi),  # y-component
            np.cos(theta)                # z-component
        ])
        return direction / np.linalg.norm(direction)  # Ensure it's a unit vector]
    
        self.particles = [(particle, position)]
    
    def _update_position(self, displacement):
        self.position += displacement
        self.x = self.position[0]
        self.y = self.position[1]
        self.z = self.position[2]
        self.r = np.sqrt(self.x**2 + self.y**2)

    def _is_in_tank(self):
        r_tank = R_TANK
        h_tank = H_TANK
        if self.z >= 0. * u.cm  and self.z <= h_tank and self.r <= r_tank: 
            return True 
        else: 
            return False 

    def _generate_pair(self, gamma):
        E_gamma = gamma.energy 

        rng = np.random.default_rng()  

        # frac = 0,1 are not physical, the energy should be at least the mass of the particle 

        def extract_x_fraction():
            while True:
                x_value = rng.uniform(ELECTRON_MASS/E_gamma, 1 - ELECTRON_MASS/E_gamma, size=1) 
                y_value = rng.uniform(0, 1, size=1) 
                if y_value < self._dsigma_dx(x_value):
                    return x_value[0]
                
        frac = extract_x_fraction()
        if(self.is_verbose):print('Fraction of energy to electron: ',frac)
        E_e_minus = copy.deepcopy(E_gamma) * frac
        E_e_plus = copy.deepcopy(E_gamma) -  copy.deepcopy(E_e_minus)

        electron = Electron(energy = E_e_minus)
        positron = Electron(energy = E_e_plus)

        return electron, positron

    def _generate_pair_directons(self, gamma):
        E_gamma = gamma.energy 
       
        n_gamma = self.direction
        nx = np.array([1.,0.,0.])
        e1 = np.cross(nx, n_gamma) / np.linalg.norm(np.cross(nx, n_gamma) )
        e2 = np.cross(n_gamma, e1)

        theta_pp = ELECTRON_MASS.to(u.MeV) / E_gamma.to(u.MeV) * u.rad
        if(self.is_verbose):print('Opening angle electron-positron pair: ',theta_pp.to(u.deg))

        # Compute azimuthal angle on the plane perpendicular to the direction of the photon 
        rng = np.random.default_rng()  
        phi_pp = rng.uniform(0, 1, size=1) * np.pi * 2

        n_e_minus = e1 * np.cos(phi_pp) * np.sin(theta_pp.value) + e2 * np.sin(phi_pp) * np.sin(theta_pp.value) + n_gamma * np.cos(theta_pp.value)
        n_e_plus = e1 * np.cos(phi_pp + np.pi) * np.sin(theta_pp.value) + e2 * np.sin(phi_pp + np.pi) * np.sin(theta_pp.value) + n_gamma * np.cos(theta_pp.value)

        return n_e_minus, n_e_plus
    
    def propagate(self, energy_threshold=0.1 * u.MeV):
        """
        Propagates the photon step by step until in undergoes pair production.

        :param energy_threshold: The minimum energy before stopping the photon.
        """
        if(self.is_verbose):print(f"Starting particle tracking at position {self.position} with energy {self.particle.energy:.2f}")

        # Define pair production interaction length as 9/7 of radiation length 
        int_length = self.material.X0 * 9./7 
        l_int = np.random.exponential(int_length.value, size=1)[0] * u.cm  

        if(self.is_verbose):print("Sampled pair production interacton length: ",l_int)
        l_tank = 0. * u.cm # Store length crossed inside the tank  


        while self.particle.energy > energy_threshold and self.z > 0 * u.cm and self.z < WORLD_Z and self.r < WORLD_R:

            # Update position using direction vector
            # Attention! Sign of step size determines the direction of propagation 

            displacement = self.direction * self.step_size
            self._update_position(displacement)

            while self._is_in_tank(): 

                displacement = self.direction * self.step_size
                self._update_position(displacement)
                l_tank += np.linalg.norm(displacement)
                if(self.is_verbose):print("Path in the tank: ",l_tank)

                # When path inside the tank becomes greater than interaction length, trigger pair production: 
                if(l_tank > l_int and self.particle.energy.to(u.MeV) > 2 * ELECTRON_MASS.to(u.MeV)):
                    if(self.is_verbose): print('Pair production at ', self.position)
                    self.is_pair_production = True
                    electron, positron = self._generate_pair(self.particle)
                    n_electron, n_positron = self._generate_pair_directons(self.particle)

                    electron.load_delta_parameter(self.material)    
                    positron.load_delta_parameter(self.material)    

                    theta_electron = np.arccos(n_electron[2]) * u.rad
                    phi_electron = np.arctan(n_electron[1]/n_electron[0])* u.rad

                    theta_positron = np.arccos(n_positron[2]) * u.rad
                    phi_positron = np.arctan2(n_positron[1],n_positron[0]) * u.rad

                    electron_position = copy.deepcopy(self.position)
                    positron_position = copy.deepcopy(self.position)

                    electron_tracker = ParticleTracker(electron, self.material, self.step_size, electron_position, theta_electron, phi_electron)
                    positron_tracker = ParticleTracker(positron, self.material, self.step_size, positron_position, theta_positron, phi_positron)

                    return electron_tracker, positron_tracker 
                                

        if(self.is_verbose): print("Tracking complete. Particle stopped.")
        if(self.is_pair_production == False): 
            if(self.is_verbose): print("No pair production happened.")
            return None

