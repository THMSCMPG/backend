import os
import io
import time
import base64
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Required for server-side rendering
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from flask_mail import Mail, Message
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple

# ============================================================================
# INITIALIZATION & CORE CONFIG
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# FIXED: Updated CORS configuration with all required headers and origins
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://thmscmpg.github.io",
            "https://thmscmpg.github.io/backend-hub",
            "https://thmscmpg.github.io/CircuitNotes",
            "https://thmscmpg.github.io/AURA-MF",
            "http://localhost:4000",
            "http://127.0.0.1:4000",
            "http://localhost:8000",
            "http://127.0.0.1:8000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"],
        "supports_credentials": False,
        "expose_headers": ["Content-Type"]
    }
})
    
# SMTP Setup
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME=os.environ.get("MAIL_USERNAME"),
    MAIL_PASSWORD=os.environ.get("MAIL_PASSWORD"),
    MAIL_DEFAULT_SENDER=os.environ.get("MAIL_USERNAME")
)

mail = Mail(app)
CONTACT_RECIPIENT = os.environ.get("CONTACT_EMAIL", os.environ.get("MAIL_USERNAME", "admin@example.com"))

# ============================================================================
# PHYSICS & SIMULATION ENGINE
# ============================================================================

@dataclass
class SimState:
    """Global state for the ML Orchestrator demo."""
    time: float = 0.0
    fidelity_history: List[int] = field(default_factory=list)
    
    def step(self):
        self.time += 1.0
        # Cycle through LF (0), MF (1), HF (2) every 10 steps for demo
        fid = int((self.time // 10) % 3)
        self.fidelity_history.append(fid)
        return fid

state_manager = SimState()


class AURA_Physics_Solver:
    """
    2D Thermal Finite-Difference Solver for a photovoltaic panel.

    Energy balance per grid cell per timestep
    ------------------------------------------
        Q_abs     = alpha_s * G                         [W/m²]  absorbed solar
        Q_elec    = eta_e * Q_abs                       [W/m²]  extracted as electricity
        Q_thermal = Q_abs - Q_elec                      [W/m²]  remainder heats the cell
        Q_conv    = h_conv * (T - T_inf)                [W/m²]  convective loss
        Q_rad     = eps * sigma * (T⁴ - T_sky⁴)        [W/m²]  radiative loss
        Q_cond    = alpha_th * laplacian(T)             [W/m²]  lateral conduction

        dT = (Q_thermal - Q_conv - Q_rad + Q_cond) * dt * scale

    Seven user-controllable parameters (one per frontend slider)
    -------------------------------------------------------------
        G           solar irradiance       [W/m²]   800–1200
        T_inf       ambient temperature    [K]      280–330
        u           wind speed             [m/s]    0–10
        eta_e       cell efficiency        [–]      0.10–0.30
        k           thermal conductivity   [W/(m·K)] 100–200
        alpha_s     solar absorptivity     [–]      0.85–0.98
        eps         surface emissivity     [–]      0.80–0.95

    Fixed material / correlation constants
    ---------------------------------------
        SIGMA    5.67e-8   Stefan-Boltzmann [W/(m²·K⁴)]
        RHO      2400      density of PV laminate [kg/m³]
        CP       900       specific heat [J/(kg·K)]
        H_BASE   10        natural-convection baseline [W/(m²·K)]
        H_WIND   5         forced-convection scaling per m/s [W/(m²·K·s/m)]
        T_SKY_OFFSET  10   clear-sky approximation: T_sky = T_inf − 10 K
        BETA     0.004     linear power derating coeff [1/K] (standard Si cell)
        T_REF    298.15    IEC reference temperature [K] (25 °C)
    """

    SIGMA          = 5.67e-8   # Stefan-Boltzmann constant
    RHO            = 2400.0    # laminate density          [kg/m³]
    CP             = 900.0     # specific heat             [J/(kg·K)]
    H_BASE         = 10.0      # natural-convection base   [W/(m²·K)]
    H_WIND         = 5.0       # forced-convection scale   [W/(m²·K) per m/s]
    T_SKY_OFFSET   = 10.0      # T_sky = T_inf - this      [K]
    BETA           = 0.004     # power derating coeff      [1/K]
    T_REF          = 298.15    # reference temperature     [K]

    def __init__(self, nx=20, ny=20):
        self.nx, self.ny = nx, ny
        self.dx = 0.1                              # grid spacing [m]
        self.T  = np.ones((ny, nx)) * self.T_REF   # start at 25 °C

    def solve(self, solar, wind, ambient, fidelity,
              cell_efficiency, thermal_conductivity, absorptivity, emissivity):
        """
        Explicit finite-difference time loop.

        Parameters
        ----------
        solar               float   irradiance G            [W/m²]
        wind                float   wind speed u            [m/s]
        ambient             float   ambient temp T_inf      [K]
        fidelity            int     0=LF / 1=MF / 2=HF     controls step count
        cell_efficiency     float   electrical η_e          [–]
        thermal_conductivity float  in-plane k              [W/(m·K)]
        absorptivity        float   solar α_s               [–]
        emissivity          float   surface ε               [–]

        Returns
        -------
        self.T   ndarray   final temperature field         [K]
        """
        # --- fidelity → number of time steps --------------------------------
        steps = [5, 20, 100][fidelity]
        dt    = 0.1   # [s]

        # --- derived quantities (computed once, outside the loop) -----------

        # Convective transfer coefficient: h = h_base + h_wind · u
        h_conv = self.H_BASE + self.H_WIND * wind

        # Thermal diffusivity: α_th = k / (ρ · Cp)   [m²/s]
        alpha_th = thermal_conductivity / (self.RHO * self.CP)

        # Absorbed solar flux: Q_abs = α_s · G        [W/m²]
        q_abs = absorptivity * solar

        # Electrical extraction: Q_elec = η_e · Q_abs [W/m²]
        q_elec = cell_efficiency * q_abs

        # Net solar input to the thermal budget
        q_thermal = q_abs - q_elec                    # [W/m²]

        # Clear-sky temperature for radiative exchange
        T_sky = ambient - self.T_SKY_OFFSET           # [K]

        # Combined update scale: dt / (ρ · Cp · dx)
        #   dx used as effective slab thickness (thin-panel limit)
        scale = dt / (self.RHO * self.CP * self.dx)

        # --- time loop ------------------------------------------------------
        for _ in range(steps):

            # 1. Convective loss
            q_conv = h_conv * (self.T - ambient)

            # 2. Radiative loss  Q_rad = ε · σ · (T⁴ − T_sky⁴)
            q_rad  = emissivity * self.SIGMA * (self.T**4 - T_sky**4)

            # 3. Lateral conduction via 5-point Laplacian stencil
            #    Neumann (zero-flux) boundary via edge-padding
            T_pad     = np.pad(self.T, 1, mode='edge')
            laplacian = (
                T_pad[1:-1,  2:]   +   # east
                T_pad[1:-1, :-2]   +   # west
                T_pad[ 2:,  1:-1]  +   # south
                T_pad[:-2,  1:-1]      # north
                - 4.0 * self.T
            ) / self.dx**2
            q_cond = alpha_th * laplacian

            # 4. Net flux → temperature update
            q_net  = q_thermal - q_conv - q_rad + q_cond
            self.T = self.T + q_net * scale

        return self.T

    def compute_power_metrics(self, solar, cell_efficiency, absorptivity):
        """
        Compute instantaneous power output and efficiency at current T field.

        Returns
        -------
        power_total  float   total electrical power [W]
        eff_avg      float   effective PV efficiency [–] (accounting for T derating)
        """
        # Absorbed solar flux per cell: Q_abs = α_s · G  [W/m²]
        q_abs = absorptivity * solar

        # Electrical conversion: η_e(T) = η_ref · [1 − β·(T − T_ref)]
        eta_local = cell_efficiency * (1.0 - self.BETA * (self.T - self.T_REF))

        # Clip to ensure η ≥ 0 (high temps → some cells might turn off)
        eta_local = np.maximum(eta_local, 0.0)

        # Power per cell: P = Q_abs · η_e(T)   [W/m²]
        power_density = q_abs * eta_local

        # Panel area:  A_cell = dx · dx   [m²]
        A_cell = self.dx * self.dx

        # Total power over entire panel:
        power_total = float(np.sum(power_density) * A_cell)  # [W]

        # Average effective efficiency (domain-averaged):
        eff_avg = float(np.mean(eta_local))

        return power_total, eff_avg


def generate_plot(temperature_field):
    """Generate base64-encoded heatmap of temperature field."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Convert Kelvin to Celsius for display
    temp_celsius = temperature_field - 273.15
    
    im = ax.imshow(temp_celsius, cmap='hot', origin='lower', interpolation='bilinear')
    ax.set_title('PV Panel Temperature Distribution', fontsize=14, weight='bold')
    ax.set_xlabel('Position (grid cells)')
    ax.set_ylabel('Position (grid cells)')
    
    cbar = plt.colorbar(im, ax=ax, label='Temperature (°C)')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ============================================================================
# API ROUTES
# ============================================================================

# FIXED: Now accepts both GET and POST methods
@app.route('/api/simulate', methods=['GET', 'POST', 'OPTIONS'])
def handle_simulation():
    if request.method == 'OPTIONS':
        return '', 204

    t_start = time.time()

    # FIXED: Handle both GET (dashboard auto-update) and POST (user-triggered)
    if request.method == 'GET':
        # Dashboard auto-update with default parameters
        data = {}
    else:
        # User-triggered simulation with custom parameters
        data = request.json if request.is_json else {}

    # ---- read all 7 parameters with defaults matching the slider defaults --
    solar                = float(data.get('solar',                1000.0))
    wind                 = float(data.get('wind',                 2.0   ))
    ambient              = float(data.get('ambient',              298.15))
    cell_efficiency      = float(data.get('cell_efficiency',      0.20  ))
    thermal_conductivity = float(data.get('thermal_conductivity', 130.0 ))
    absorptivity         = float(data.get('absorptivity',         0.95  ))
    emissivity           = float(data.get('emissivity',           0.90  ))

    # ---- clamp every value to its slider's physical range -------------------
    solar                = float(np.clip(solar,                800.0,  1200.0))
    wind                 = float(np.clip(wind,                 0.0,    10.0  ))
    ambient              = float(np.clip(ambient,              280.0,  330.0 ))
    cell_efficiency      = float(np.clip(cell_efficiency,      0.10,   0.30  ))
    thermal_conductivity = float(np.clip(thermal_conductivity, 100.0,  200.0 ))
    absorptivity         = float(np.clip(absorptivity,         0.85,   0.98  ))
    emissivity           = float(np.clip(emissivity,           0.80,   0.95  ))

    logger.info(
        f"Simulation: solar={solar} wind={wind} ambient={ambient} "
        f"eta_e={cell_efficiency} k={thermal_conductivity} "
        f"alpha_s={absorptivity} eps={emissivity}"
    )

    # ---- multi-fidelity state tick ------------------------------------------
    current_fid = state_manager.step()

    # ---- run solver ---------------------------------------------------------
    solver       = AURA_Physics_Solver()
    result_field = solver.solve(
        solar, wind, ambient, current_fid,
        cell_efficiency, thermal_conductivity, absorptivity, emissivity
    )

    # ---- post-solve power metrics ------------------------------------------
    power_total, eff_avg = solver.compute_power_metrics(
        solar, cell_efficiency, absorptivity
    )

    # ---- wall-clock runtime ------------------------------------------------
    runtime_ms = (time.time() - t_start) * 1000.0

    # ---- ML-orchestrator heuristics (demo) ---------------------------------
    confidence = 0.98 - (current_fid * 0.05) + (np.random.random() * 0.02)
    residual   = [1e-3, 1e-5, 1e-8][current_fid]

    return jsonify({
        "temperature_field": result_field.tolist(),
        "visualization":     generate_plot(result_field),
        "fidelity_level":    current_fid,
        "fidelity_name":     ["Low (LF)", "Medium (MF)", "High (HF)"][current_fid],
        "ml_confidence":     round(confidence, 4),
        "energy_residuals":  residual,
        "timestamp":         state_manager.time,
        "stats": {
            "max_t":      round(float(np.max(result_field))  - 273.15, 2),
            "min_t":      round(float(np.min(result_field))  - 273.15, 2),
            "avg_t":      round(float(np.mean(result_field)) - 273.15, 2),
            "power_total": round(power_total, 2),       # [W]
            "eff_avg":     round(eff_avg * 100, 2),     # converted to % for display
            "runtime_ms":  round(runtime_ms, 1)         # [ms]
        }
    })

@app.route('/api/contact', methods=['POST', 'OPTIONS'])
def handle_contact():
    if request.method == 'OPTIONS':
        return '', 204
        
    data = request.get_json()
    if not data or data.get('website_hp'):
        return jsonify({"status": "success"}), 200

    logger.info(f"Contact form submission from: {data.get('email')}")

    try:
        msg = Message(
            subject=f"Contact from {data.get('name')} - THMSCMPG Portfolio",
            recipients=[CONTACT_RECIPIENT],
            body=f"From: {data.get('name')} <{data.get('email')}>\n\nMessage:\n{data.get('message')}"
        )
        mail.send(msg)
        logger.info("Email sent successfully")
        return jsonify({"status": "success", "message": "Message sent!"})
    except Exception as e:
        logger.error(f"Contact Error: {e}")
        return jsonify({"status": "error", "message": "Email service failed"}), 500

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 204
        
    return jsonify({
        "status": "active", 
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/')
def docs():
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AURA-MF Backend API</title>
            <style>
                body { 
                    font-family: system-ui; 
                    max-width: 800px; 
                    margin: 50px auto; 
                    padding: 20px;
                    background: #f5f5f5;
                }
                .container {
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 { color: #667eea; }
                .status { 
                    background: #e8f5e9; 
                    padding: 15px; 
                    border-radius: 5px;
                    margin: 20px 0;
                    border-left: 4px solid #4CAF50;
                }
                .endpoint {
                    background: #f5f5f5;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 5px;
                    font-family: monospace;
                }
                a { color: #667eea; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 AURA-MF Backend API</h1>
                <div class="status">
                    ✅ Status: <strong>Running</strong><br>
                    🕐 Uptime: {{ time }} simulation steps
                </div>
                
                <h2>Available Endpoints</h2>
                <div class="endpoint">GET /api/health - Health check</div>
                <div class="endpoint">POST /api/contact - Contact form submission</div>
                <div class="endpoint">GET/POST /api/simulate - Physics simulation</div>
                
                <h2>Frontend Sites</h2>
                <p>
                    <a href="https://thmscmpg.github.io" target="_blank">Portfolio (THMSCMPG)</a><br>
                    <a href="https://thmscmpg.github.io/CircuitNotes" target="_blank">CircuitNotes</a><br>
                    <a href="https://thmscmpg.github.io/AURA-MF" target="_blank">AURA-MF</a>
                </p>
                
                <p style="margin-top: 40px; color: #666; font-size: 0.9em;">
                    Backend for THMSCMPG GitHub Pages • Powered by Flask + Render
                </p>
            </div>
        </body>
        </html>
    """, time=state_manager.time)

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin and origin.startswith('https://thmscmpg.github.io'):
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Accept')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
