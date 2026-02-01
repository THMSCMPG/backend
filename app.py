import os
import io
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

CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://thmscmpg.github.io",
            "https://thmscmpg.github.io/backend-hub",
            "https://thmscmpg.github.io/CircuitNotes",
            "https://thmscmpg.github.io/AURA-MF",
            "http://localhost:4000",
            "http://127.0.0.1:4000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": False
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
    """2D Thermal Finite Difference Solver."""
    SIGMA = 5.67e-8  # Stefan-Boltzmann
    
    def __init__(self, nx=20, ny=20):
        self.nx, self.ny = nx, ny
        self.dx = 0.1 # meters
        self.T = np.ones((ny, nx)) * 298.15 # Start at 25°C

    def solve(self, solar, wind, ambient, fidelity):
        """
        Runs a simplified heat balance: 
        dT/dt = (Q_solar - Q_conv - Q_rad + Q_cond) / (m * Cp)
        """
        # Multi-fidelity resolution scaling
        steps = [5, 20, 100][fidelity] 
        dt = 0.1
        
        # Physical constants
        h_conv = 10.0 + (5.0 * wind)
        emissivity = 0.9
        alpha = 1.3e-4 # Thermal diffusivity
        
        for _ in range(steps):
            # 1. Convection & Radiation
            q_conv = h_conv * (self.T - ambient)
            q_rad = emissivity * self.SIGMA * (self.T**4 - (ambient-10)**4)
            
            # 2. Conduction (Laplacian)
            T_pad = np.pad(self.T, 1, mode='edge')
            laplacian = (T_pad[1:-1, 2:] + T_pad[1:-1, :-2] + 
                         T_pad[2:, 1:-1] + T_pad[:-2, 1:-1] - 4*self.T) / self.dx**2
            
            # 3. Net Flux
            # Q_solar is reduced by 20% efficiency (energy converted to electricity)
            q_net = (solar * 0.8) - q_conv - q_rad + (alpha * laplacian)
            
            # 4. Update (Simplified mass/Cp factor)
            self.T += q_net * dt * 0.001
            
        return np.clip(self.T, 250, 400)

# ============================================================================
# VISUALIZATION UTILITY
# ============================================================================

def generate_plot(temp_data):
    """Generates a heatmap and returns it as a Base64 string."""
    plt.figure(figsize=(5, 4))
    plt.imshow(temp_data - 273.15, cmap='magma', origin='lower')
    plt.colorbar(label='Temp (°C)')
    plt.title("AURA-MF Thermal Field")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/simulate', methods=['GET', 'POST', 'OPTIONS'])
def handle_simulation():
    # Handle preflight
    if request.method == 'OPTIONS':
        return '', 204
        
    # 1. Get Inputs (Default or Request)
    data = request.json if request.is_json else {}
    solar = float(data.get('solar', 1000.0))
    wind = float(data.get('wind', 2.0))
    ambient = float(data.get('ambient', 298.15))
    
    logger.info(f"Simulation request: solar={solar}, wind={wind}, ambient={ambient}")
    
    # 2. Update Multi-Fidelity State
    current_fid = state_manager.step()
    
    # 3. Run Solver
    solver = AURA_Physics_Solver()
    result_field = solver.solve(solar, wind, ambient, current_fid)
    
    # 4. Generate Analytics
    confidence = 0.98 - (current_fid * 0.05) + (np.random.random() * 0.02)
    residual = [1e-3, 1e-5, 1e-8][current_fid]
    
    return jsonify({
        "temperature_field": result_field.tolist(),
        "visualization": generate_plot(result_field),
        "fidelity_level": current_fid,
        "fidelity_name": ["Low (LF)", "Medium (MF)", "High (HF)"][current_fid],
        "ml_confidence": round(confidence, 4),
        "energy_residuals": residual,
        "timestamp": state_manager.time,
        "stats": {
            "max_t": round(np.max(result_field) - 273.15, 2),
            "avg_t": round(np.mean(result_field) - 273.15, 2)
        }
    })

@app.route('/api/contact', methods=['POST', 'OPTIONS'])
def handle_contact():
    # Handle preflight
    if request.method == 'OPTIONS':
        return '', 204
        
    data = request.get_json()
    if not data or data.get('website_hp'): # Honeypot check
        return jsonify({"status": "success"}), 200 # Silent drop for bots

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
    # Handle preflight
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
                <div class="endpoint">POST /api/simulate - Physics simulation</div>
                
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

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    # Allow any thmscmpg.github.io subdomain
    if origin and origin.startswith('https://thmscmpg.github.io'):
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
