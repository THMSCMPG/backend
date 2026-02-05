# ============================================================================
# PHYSICS & SIMULATION ENGINE
# ============================================================================
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Required for server-side rendering
import matplotlib.pyplot as plt


@dataclass
class SimState:
    """Global state for the ML Orchestrator demo."""
    time: float = 0.0
    fidelity_history: List[int] = field(default_factory=list)
    
    def step(self):
        self.time += 1.0
        fid = int((self.time // 10) % 3)
        self.fidelity_history.append(fid)
        return fid


state_manager = SimState()


class AURA_Physics_Solver:
    """
    2D finite-difference thermal solver for a photovoltaic panel.

    Temperature unit contract:
      - Input  `ambient`: KELVIN  (frontend slider range 280–330 K)
      - Internal field `self.T`: KELVIN
      - Output stats: CELSIUS (converted at the API response boundary)
    """
    SIGMA          = 5.67e-8   # Stefan-Boltzmann constant [W/m²·K⁴]
    RHO            = 2400.0    # Density [kg/m³]
    CP             = 900.0     # Specific heat [J/(kg·K)]
    H_BASE         = 10.0      # Base convective coefficient [W/(m²·K)]
    H_WIND         = 5.0       # Wind-dependent convective increment
    T_SKY_OFFSET   = 10.0      # Sky temperature offset below ambient [K]
    BETA           = 0.004     # Temperature coefficient of efficiency [1/K]
    T_REF          = 298.15    # Reference temperature [K]

    def __init__(self, nx=20, ny=20):
        self.nx, self.ny = nx, ny
        self.dx = 0.1                               # Grid spacing [m]
        self.T  = np.ones((ny, nx)) * self.T_REF    # Initial field [K]

    def solve(self, solar, wind, ambient, fidelity,
              cell_efficiency, thermal_conductivity, absorptivity, emissivity):
        """
        Advance the temperature field through `steps` sub-iterations.
        `ambient` is in KELVIN.
        """
        steps = [5, 20, 100][fidelity]
        dt    = 0.1   # Sub-step size [s]
        h_conv = self.H_BASE + self.H_WIND * wind
        alpha_th = thermal_conductivity / (self.RHO * self.CP)
        q_abs = absorptivity * solar
        q_elec = cell_efficiency * q_abs
        q_thermal = q_abs - q_elec
        T_sky = ambient - self.T_SKY_OFFSET   # Sky radiative sink [K]
        scale = dt / (self.RHO * self.CP * self.dx)

        for _ in range(steps):
            q_conv = h_conv * (self.T - ambient)
            q_rad  = emissivity * self.SIGMA * (self.T**4 - T_sky**4)
            T_pad     = np.pad(self.T, 1, mode='edge')
            laplacian = (
                T_pad[1:-1,  2:]   +
                T_pad[1:-1, :-2]   +
                T_pad[ 2:,  1:-1]  +
                T_pad[:-2,  1:-1]  
                - 4.0 * self.T
            ) / self.dx**2
            q_cond = alpha_th * laplacian
            q_net  = q_thermal - q_conv - q_rad + q_cond
            self.T = self.T + q_net * scale
        return self.T

    def compute_power_metrics(self, solar, cell_efficiency, absorptivity):
        q_abs = absorptivity * solar
        eta_local = cell_efficiency * (1.0 - self.BETA * (self.T - self.T_REF))
        eta_local = np.maximum(eta_local, 0.0)
        power_density = q_abs * eta_local
        A_cell = self.dx * self.dx
        power_total = float(np.sum(power_density) * A_cell)
        eff_avg = float(np.mean(eta_local))
        return power_total, eff_avg


# ============================================================================
# BTE-NS COUPLED SOLVER IMPLEMENTATION
# ============================================================================

class BTESolver:
    """
    Boltzmann Transport Equation solver for electron/phonon transport.
    Simplified model using relaxation time approximation.
    """
    
    # Physical constants
    KB = 1.380649e-23      # Boltzmann constant [J/K]
    HBAR = 1.054571817e-34 # Reduced Planck constant [J·s]
    Q_E = 1.602176634e-19  # Elementary charge [C]
    
    def __init__(self, nx=20, ny=20):
        self.nx, self.ny = nx, ny
        self.dx = 1e-6  # Grid spacing [m] - micron scale
        
        # Carrier concentrations [1/m³]
        self.n_electrons = np.ones((ny, nx)) * 1e21
        self.n_holes = np.ones((ny, nx)) * 1e21
        
        # Current densities [A/m²]
        self.J_e = np.zeros((ny, nx, 2))  # Electron current (x,y components)
        self.J_h = np.zeros((ny, nx, 2))  # Hole current (x,y components)
        
        # Relaxation times [s]
        self.tau_e = 1e-12  # Electron scattering time
        self.tau_h = 1e-12  # Hole scattering time
        
    def solve(self, T_field, E_field, fidelity_level):
        """
        Solve BTE for carrier transport given temperature and electric fields.
        
        Args:
            T_field: Temperature field [K] (ny, nx)
            E_field: Electric field [V/m] (ny, nx, 2) - (x,y components)
            fidelity_level: 0=low, 1=medium, 2=high
            
        Returns:
            dict with carrier densities and currents
        """
        iterations = [10, 50, 200][fidelity_level]
        dt = [1e-11, 1e-12, 1e-13][fidelity_level]
        
        # Mobility calculation (temperature dependent)
        mu_e = self._calculate_mobility(T_field, carrier='electron')
        mu_h = self._calculate_mobility(T_field, carrier='hole')
        
        for _ in range(iterations):
            # Drift-diffusion approximation of BTE
            # J = q*n*mu*E + q*D*grad(n)
            
            # Diffusion coefficients (Einstein relation)
            D_e = mu_e * self.KB * T_field / self.Q_E
            D_h = mu_h * self.KB * T_field / self.Q_E
            
            # Compute gradients
            grad_n_e = self._compute_gradient(self.n_electrons)
            grad_n_h = self._compute_gradient(self.n_holes)
            
            # Drift current
            J_drift_e = self.Q_E * self.n_electrons[:, :, np.newaxis] * mu_e[:, :, np.newaxis] * E_field
            J_drift_h = self.Q_E * self.n_holes[:, :, np.newaxis] * mu_h[:, :, np.newaxis] * E_field
            
            # Diffusion current
            J_diff_e = self.Q_E * D_e[:, :, np.newaxis] * grad_n_e
            J_diff_h = self.Q_E * D_h[:, :, np.newaxis] * grad_n_h
            
            # Total current
            self.J_e = J_drift_e + J_diff_e
            self.J_h = J_drift_h - J_diff_h  # Opposite sign for holes
            
            # Update carrier densities (continuity equation)
            div_J_e = self._compute_divergence(self.J_e)
            div_J_h = self._compute_divergence(self.J_h)
            
            self.n_electrons += dt * (-div_J_e / self.Q_E)
            self.n_holes += dt * (-div_J_h / self.Q_E)
            
            # Apply boundary conditions
            self.n_electrons = np.maximum(self.n_electrons, 1e15)
            self.n_holes = np.maximum(self.n_holes, 1e15)
        
        return {
            'n_electrons': self.n_electrons.copy(),
            'n_holes': self.n_holes.copy(),
            'J_e': self.J_e.copy(),
            'J_h': self.J_h.copy(),
            'J_total': self.J_e + self.J_h
        }
    
    def _calculate_mobility(self, T_field, carrier='electron'):
        """Temperature-dependent mobility [m²/(V·s)]"""
        T_ref = 300.0
        mu_ref = 0.14 if carrier == 'electron' else 0.05  # Reference mobility
        
        # Power-law temperature dependence
        return mu_ref * (T_field / T_ref) ** (-1.5)
    
    def _compute_gradient(self, field):
        """Compute 2D gradient using central differences"""
        grad = np.zeros((self.ny, self.nx, 2))
        
        # X-direction
        grad[:, 1:-1, 0] = (field[:, 2:] - field[:, :-2]) / (2 * self.dx)
        grad[:, 0, 0] = (field[:, 1] - field[:, 0]) / self.dx
        grad[:, -1, 0] = (field[:, -1] - field[:, -2]) / self.dx
        
        # Y-direction
        grad[1:-1, :, 1] = (field[2:, :] - field[:-2, :]) / (2 * self.dx)
        grad[0, :, 1] = (field[1, :] - field[0, :]) / self.dx
        grad[-1, :, 1] = (field[-1, :] - field[-2, :]) / self.dx
        
        return grad
    
    def _compute_divergence(self, vector_field):
        """Compute divergence of 2D vector field"""
        div = np.zeros((self.ny, self.nx))
        
        # d(Jx)/dx + d(Jy)/dy
        div[:, 1:-1] += (vector_field[:, 2:, 0] - vector_field[:, :-2, 0]) / (2 * self.dx)
        div[1:-1, :] += (vector_field[2:, :, 1] - vector_field[:-2, :, 1]) / (2 * self.dx)
        
        return div


class NSSolver:
    """
    Navier-Stokes solver for fluid flow (air cooling around solar panel).
    Incompressible flow with Boussinesq approximation.
    """
    
    def __init__(self, nx=20, ny=20):
        self.nx, self.ny = nx, ny
        self.dx = 0.01  # Grid spacing [m] - cm scale
        
        # Velocity field [m/s]
        self.u = np.zeros((ny, nx))  # x-velocity
        self.v = np.zeros((ny, nx))  # y-velocity
        
        # Pressure field [Pa]
        self.p = np.zeros((ny, nx))
        
        # Air properties (at 300K)
        self.rho = 1.177    # Density [kg/m³]
        self.nu = 1.57e-5   # Kinematic viscosity [m²/s]
        self.alpha_air = 2.2e-5  # Thermal diffusivity [m²/s]
        self.beta_th = 3.4e-3    # Thermal expansion coefficient [1/K]
        self.g = 9.81       # Gravitational acceleration [m/s²]
        
    def solve(self, T_field, wind_speed, ambient_T, fidelity_level):
        """
        Solve incompressible Navier-Stokes equations.
        
        Args:
            T_field: Temperature field [K]
            wind_speed: External wind velocity [m/s]
            ambient_T: Ambient temperature [K]
            fidelity_level: 0=low, 1=medium, 2=high
            
        Returns:
            dict with velocity and pressure fields
        """
        iterations = [20, 100, 500][fidelity_level]
        dt = [0.01, 0.001, 0.0001][fidelity_level]
        
        # Initialize with wind boundary condition
        self.u[:, 0] = wind_speed
        
        for _ in range(iterations):
            # Store old velocities
            u_old = self.u.copy()
            v_old = self.v.copy()
            
            # Compute buoyancy force (Boussinesq approximation)
            F_buoy = self.g * self.beta_th * (T_field - ambient_T)
            
            # Advection terms (semi-Lagrangian)
            u_adv = self._advect(self.u, u_old, v_old, dt)
            v_adv = self._advect(self.v, u_old, v_old, dt)
            
            # Diffusion terms (viscosity)
            u_diff = self._diffuse(u_adv, self.nu, dt)
            v_diff = self._diffuse(v_adv, self.nu, dt)
            
            # Add buoyancy to vertical velocity
            v_diff += F_buoy * dt
            
            # Project to divergence-free field
            self.u, self.v = self._project(u_diff, v_diff)
            
            # Apply boundary conditions
            self._apply_boundary_conditions(wind_speed)
        
        return {
            'u': self.u.copy(),
            'v': self.v.copy(),
            'p': self.p.copy(),
            'velocity_magnitude': np.sqrt(self.u**2 + self.v**2)
        }
    
    def _advect(self, field, u, v, dt):
        """Semi-Lagrangian advection"""
        result = np.zeros_like(field)
        
        for j in range(self.ny):
            for i in range(self.nx):
                # Backward trace
                x = i - dt * u[j, i] / self.dx
                y = j - dt * v[j, i] / self.dx
                
                # Bilinear interpolation
                x = np.clip(x, 0, self.nx - 1.001)
                y = np.clip(y, 0, self.ny - 1.001)
                
                i0, j0 = int(x), int(y)
                i1, j1 = min(i0 + 1, self.nx - 1), min(j0 + 1, self.ny - 1)
                
                sx, sy = x - i0, y - j0
                
                result[j, i] = (
                    (1 - sx) * (1 - sy) * field[j0, i0] +
                    sx * (1 - sy) * field[j0, i1] +
                    (1 - sx) * sy * field[j1, i0] +
                    sx * sy * field[j1, i1]
                )
        
        return result
    
    def _diffuse(self, field, nu, dt):
        """Implicit diffusion solver"""
        a = dt * nu / (self.dx ** 2)
        iterations = 20
        
        result = field.copy()
        for _ in range(iterations):
            result_new = result.copy()
            result_new[1:-1, 1:-1] = (
                field[1:-1, 1:-1] + a * (
                    result[1:-1, 2:] + result[1:-1, :-2] +
                    result[2:, 1:-1] + result[:-2, 1:-1]
                )
            ) / (1 + 4 * a)
            result = result_new
        
        return result
    
    def _project(self, u, v):
        """Project velocity field to be divergence-free"""
        # Compute divergence
        div = np.zeros((self.ny, self.nx))
        div[1:-1, 1:-1] = (
            (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * self.dx) +
            (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * self.dx)
        )
        
        # Solve Poisson equation for pressure
        p = np.zeros_like(div)
        for _ in range(50):
            p_new = p.copy()
            p_new[1:-1, 1:-1] = (
                div[1:-1, 1:-1] + 
                p[1:-1, 2:] + p[1:-1, :-2] +
                p[2:, 1:-1] + p[:-2, 1:-1]
            ) / 4
            p = p_new
        
        self.p = p
        
        # Subtract pressure gradient
        u_corrected = u.copy()
        v_corrected = v.copy()
        
        u_corrected[1:-1, 1:-1] -= (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * self.dx)
        v_corrected[1:-1, 1:-1] -= (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * self.dx)
        
        return u_corrected, v_corrected
    
    def _apply_boundary_conditions(self, wind_speed):
        """No-slip and inflow boundaries"""
        # Inflow (left)
        self.u[:, 0] = wind_speed
        self.v[:, 0] = 0
        
        # Outflow (right) - zero gradient
        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = self.v[:, -2]
        
        # Top/bottom - no slip
        self.u[0, :] = 0
        self.u[-1, :] = 0
        self.v[0, :] = 0
        self.v[-1, :] = 0


class CoupledSolver:
    """
    Coupled BTE-NS solver with thermal feedback.
    Integrates electron transport, fluid dynamics, and heat transfer.
    """
    
    def __init__(self, nx=20, ny=20):
        self.nx, self.ny = nx, ny
        
        # Initialize subsystems
        self.thermal_solver = AURA_Physics_Solver(nx, ny)
        self.bte_solver = BTESolver(nx, ny)
        self.ns_solver = NSSolver(nx, ny)
        
    def solve(self, fidelity_level, solar_irradiance, ambient_temperature,
              wind_speed, cell_efficiency, thermal_conductivity,
              absorptivity, emissivity, config=None):
        """
        Main coupled solve routine.
        
        Args:
            fidelity_level: 0=low, 1=medium, 2=high
            solar_irradiance: Solar input [W/m²]
            ambient_temperature: Ambient temp [K]
            wind_speed: Wind velocity [m/s]
            cell_efficiency: PV cell efficiency [0-1]
            thermal_conductivity: Material k [W/(m·K)]
            absorptivity: Solar absorption [0-1]
            emissivity: Thermal emission [0-1]
            config: Additional configuration dict
            
        Returns:
            dict with all solution fields and statistics
        """
        config = config or {}
        coupling_iterations = config.get('coupling_iterations', [2, 5, 10][fidelity_level])
        
        # Initialize electric field (simple 1D junction model)
        E_field = self._initialize_electric_field()
        
        results = {}
        
        for iteration in range(coupling_iterations):
            # 1. Solve thermal problem
            T_field = self.thermal_solver.solve(
                solar_irradiance, wind_speed, ambient_temperature,
                fidelity_level, cell_efficiency, thermal_conductivity,
                absorptivity, emissivity
            )
            
            # 2. Solve BTE for carrier transport
            bte_results = self.bte_solver.solve(T_field, E_field, fidelity_level)
            
            # 3. Solve Navier-Stokes for convection
            ns_results = self.ns_solver.solve(
                T_field, wind_speed, ambient_temperature, fidelity_level
            )
            
            # 4. Update electric field based on carrier densities
            E_field = self._update_electric_field(bte_results)
            
            # Store last iteration results
            if iteration == coupling_iterations - 1:
                results = {
                    'temperature_field': T_field,
                    'bte_results': bte_results,
                    'ns_results': ns_results,
                    'electric_field': E_field
                }
        
        # Compute statistics
        power_total, eff_avg = self.thermal_solver.compute_power_metrics(
            solar_irradiance, cell_efficiency, absorptivity
        )
        
        # Calculate additional metrics
        J_total_magnitude = np.sqrt(
            np.sum(bte_results['J_total']**2, axis=2)
        )
        
        results['statistics'] = {
            'temp_max': float(np.max(T_field)),
            'temp_min': float(np.min(T_field)),
            'temp_avg': float(np.mean(T_field)),
            'power_total': power_total,
            'efficiency_avg': eff_avg,
            'current_density_max': float(np.max(J_total_magnitude)),
            'velocity_max': float(np.max(ns_results['velocity_magnitude'])),
            'carrier_density_avg': float(np.mean(bte_results['n_electrons']))
        }
        
        return results
    
    def _initialize_electric_field(self):
        """Initialize electric field for p-n junction"""
        E_field = np.zeros((self.ny, self.nx, 2))
        
        # Simple 1D junction field in x-direction
        for i in range(self.nx):
            if i < self.nx // 2:
                E_field[:, i, 0] = 1e4  # V/m
            else:
                E_field[:, i, 0] = -1e4
        
        return E_field
    
    def _update_electric_field(self, bte_results):
        """Update electric field based on charge distribution"""
        # Poisson equation: div(E) = rho/epsilon
        n_e = bte_results['n_electrons']
        n_h = bte_results['n_holes']
        
        # Net charge density
        Q_E = 1.602176634e-19
        epsilon = 8.854e-12 * 11.7  # Silicon relative permittivity
        
        rho_charge = Q_E * (n_h - n_e)
        
        # Simple field update (full Poisson solve would be more accurate)
        E_field = np.zeros((self.ny, self.nx, 2))
        E_field[:, :, 0] = rho_charge / epsilon * 1e-6  # Simplified
        
        return E_field


class VisualizationGenerator:
    """Generate visualizations for simulation results"""
    
    @staticmethod
    def generate_temperature_heatmap(T_field):
        """Generate temperature distribution heatmap"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Convert to Celsius
        T_celsius = T_field - 273.15
        
        im = ax.imshow(T_celsius, cmap='hot', origin='lower',
                      interpolation='bilinear', aspect='auto')
        
        ax.set_title('Temperature Distribution', fontsize=14, weight='bold')
        ax.set_xlabel('X Position (grid cells)')
        ax.set_ylabel('Y Position (grid cells)')
        
        cbar = plt.colorbar(im, ax=ax, label='Temperature (°C)')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    @staticmethod
    def generate_current_density_plot(J_field):
        """Generate current density vector field"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ny, nx, _ = J_field.shape
        
        # Calculate magnitude
        J_mag = np.sqrt(J_field[:, :, 0]**2 + J_field[:, :, 1]**2)
        
        # Heatmap of magnitude
        im = ax.imshow(J_mag, cmap='viridis', origin='lower', aspect='auto')
        
        # Overlay vector field (subsample for clarity)
        skip = max(1, nx // 15)
        X, Y = np.meshgrid(range(0, nx, skip), range(0, ny, skip))
        U = J_field[::skip, ::skip, 0]
        V = J_field[::skip, ::skip, 1]
        
        ax.quiver(X, Y, U, V, color='white', alpha=0.7)
        
        ax.set_title('Current Density Distribution', fontsize=14, weight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        plt.colorbar(im, ax=ax, label='Current Density (A/m²)')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    @staticmethod
    def generate_velocity_field_plot(u, v):
        """Generate velocity field visualization"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate magnitude
        vel_mag = np.sqrt(u**2 + v**2)
        
        # Heatmap
        im = ax.imshow(vel_mag, cmap='coolwarm', origin='lower', aspect='auto')
        
        # Streamlines
        ny, nx = u.shape
        Y, X = np.mgrid[0:ny, 0:nx]
        ax.streamplot(X.T, Y.T, u.T, v.T, color='black', 
                     density=1.5, linewidth=0.5, arrowsize=0.8)
        
        ax.set_title('Air Flow Velocity Field', fontsize=14, weight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        plt.colorbar(im, ax=ax, label='Velocity (m/s)')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"


def generate_plot(temperature_field):
    """Standalone plot generation function for backwards compatibility"""
    return VisualizationGenerator.generate_temperature_heatmap(temperature_field)