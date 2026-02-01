import os
import io
import time
import base64
import logging
import numpy as np
import threading
import requests
import matplotlib
matplotlib.use('Agg')  # Required for server-side rendering
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple

# ============================================================================
# INITIALIZATION & CORE CONFIG
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Formspree configuration (set FORMSPREE_ID env var on Render)
FORMSPREE_ID = os.environ.get("FORMSPREE_ID")
FORMSPREE_URL = f"https://formspree.io/f/{FORMSPREE_ID}" if FORMSPREE_ID else None

# CORS: allow all three frontend origins + the bridge hub itself
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://thmscmpg.github.io",              # Portal + all sub-paths
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

def generate_plot(temperature_field):
    fig, ax = plt.subplots(figsize=(6, 5))
    temp_celsius = temperature_field - 273.15
    im = ax.imshow(temp_celsius, cmap='hot', origin='lower', interpolation='bilinear')
    ax.set_title('PV Panel Temperature Distribution', fontsize=14, weight='bold')
    ax.set_xlabel('Position (grid cells)')
    ax.set_ylabel('Position (grid cells)')
    plt.colorbar(im, ax=ax, label='Temperature (°C)')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/simulate', methods=['GET', 'POST', 'OPTIONS'])
def handle_simulation():
    if request.method == 'OPTIONS':
        return '', 204
    t_start = time.time()
    data = request.json if request.is_json else {} if request.method == 'POST' else {}
    
    # Parameter extraction with clipping to valid ranges.
    # ambient is expected in KELVIN from the frontend (slider range 280–330).
    solar                = float(np.clip(float(data.get('solar', 1000.0)), 800.0, 1200.0))
    wind                 = float(np.clip(float(data.get('wind', 2.0)), 0.0, 10.0))
    ambient              = float(np.clip(float(data.get('ambient', 298.15)), 280.0, 330.0))  # Kelvin
    cell_efficiency      = float(np.clip(float(data.get('cell_efficiency', 0.20)), 0.10, 0.30))
    thermal_conductivity = float(np.clip(float(data.get('thermal_conductivity', 130.0)), 100.0, 200.0))
    absorptivity         = float(np.clip(float(data.get('absorptivity', 0.95)), 0.85, 0.98))
    emissivity           = float(np.clip(float(data.get('emissivity', 0.90)), 0.80, 0.95))

    current_fid = state_manager.step()
    solver = AURA_Physics_Solver()
    result_field = solver.solve(solar, wind, ambient, current_fid,
                                cell_efficiency, thermal_conductivity, absorptivity, emissivity)
    power_total, eff_avg = solver.compute_power_metrics(solar, cell_efficiency, absorptivity)
    runtime_ms = (time.time() - t_start) * 1000.0

    # Response: temperature_field is in Kelvin; stats are converted to Celsius here.
    return jsonify({
        "temperature_field": result_field.tolist(),
        "visualization": generate_plot(result_field),
        "fidelity_level": current_fid,
        "fidelity_name": ["Low (LF)", "Medium (MF)", "High (HF)"][current_fid],
        "ml_confidence": round(0.98 - (current_fid * 0.05), 4),
        "energy_residuals": [1e-3, 1e-5, 1e-8][current_fid],
        "timestamp": state_manager.time,
        "stats": {
            # All temperature values converted to Celsius for frontend display
            "max_t":      round(float(np.max(result_field)) - 273.15, 2),
            "min_t":      round(float(np.min(result_field)) - 273.15, 2),
            "avg_t":      round(float(np.mean(result_field)) - 273.15, 2),
            "power_total": round(power_total, 2),
            "eff_avg":    round(eff_avg * 100, 2),
            "runtime_ms": round(runtime_ms, 1)
        }
    })

# ─── Contact Form ───────────────────────────────────────────────────────────

def send_via_formspree(data):
    if not FORMSPREE_URL:
        logger.error("❌ FORMSPREE_ID not found in environment variables")
        return False
    
    payload = {
        "name":     data.get("name"),
        "email":    data.get("email"),
        "message": data.get("message"),
        "_subject": f"Portfolio Contact from {data.get('name', 'Anonymous')}"
    }
    
    try:
        response = requests.post(
            FORMSPREE_URL,
            json=payload,
            headers={"Accept": "application/json"},
            timeout=10
        )
        return response.ok
    except Exception as e:
        logger.error(f"❌ Connection to Formspree failed: {e}")
        return False

@app.route('/api/contact', methods=['POST', 'OPTIONS'])
def handle_contact():
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json or {}

    # Server-side honeypot check: if website_hp is present and non-empty,
    # silently discard the submission (do not tell the bot it failed).
    if data.get('website_hp', '').strip():
        logger.warning(f"🚫 Honeypot triggered — discarding submission from {data.get('email', 'unknown')}")
        # Return 200 to not reveal to bots that filtering occurred
        return jsonify({"message": "Message received!"}), 200

    logger.info(f"Contact form submission from: {data.get('email')}")
    
    success = send_via_formspree(data)
    
    if success:
        return jsonify({"message": "Message received!"}), 200
    else:
        return jsonify({"message": "Failed to send message via provider"}), 500

# ─── Health Check ───────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 204
    return jsonify({
        "status": "active", 
        "version": "1.3.0",
        "timestamp": datetime.now().isoformat()
    })

# ─── Root Docs Page ─────────────────────────────────────────────────────────

@app.route('/')
def docs():
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head><title>AURA-MF Backend API</title><style>
            body { font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #667eea; }
            .status { background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #4CAF50; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; font-family: monospace; }
            a { color: #667eea; text-decoration: none; }
        </style></head>
        <body><div class="container">
            <h1>🚀 AURA-MF Backend API</h1>
            <div class="status">✅ Status: <strong>Running</strong><br>🕐 Uptime: {{ time }} simulation steps</div>
            <h2>Endpoints</h2>
            <div class="endpoint">GET /api/health — Health check</div>
            <div class="endpoint">POST /api/contact — Contact form (via Formspree, honeypot-protected)</div>
            <div class="endpoint">GET/POST /api/simulate — Physics simulation</div>
        </div></body></html>
    """, time=state_manager.time)

# ─── CORS After-Request Hook ────────────────────────────────────────────────

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
    app.run(host='0.0.0.0', port=port, debug=False)
