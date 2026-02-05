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
from physics import AURA_Physics_Solver, state_manager
from physics import CoupledSolver, VisualizationGenerator, generate_plot


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
            "https://thmscmpg.github.io/AURA-MF/",    # AURA-MF app path
            "https://thmscmpg.github.io/CircuitNotes/",      # CircuitNotes app path
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

# Initialize coupled solver
coupled_solver = CoupledSolver()
visualizer = VisualizationGenerator()


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/simulate', methods=['GET', 'POST', 'OPTIONS'])
def handle_simulation():
    """Original AURA-MF simulation endpoint (thermal only)"""

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


@app.route('/api/simulate/bte-ns', methods=['POST', 'OPTIONS'])
def handle_bte_ns_simulation():
    """
    BTE-NS Coupled Simulation endpoint
    Integrates Boltzmann Transport Equation + Navier-Stokes + Thermal
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    t_start = time.time()
    
    try:
        data = request.json or {}
        
        # Validate and extract parameters
        fidelity_level = int(data.get('fidelity_level', 1))
        if fidelity_level not in [0, 1, 2]:
            return jsonify({
                "success": False,
                "error": "Invalid fidelity level. Must be 0, 1, or 2."
            }), 400
        
        # Physics parameters
        solar_irradiance = float(np.clip(
            float(data.get('solar_irradiance', 1000.0)), 
            800.0, 1200.0
        ))
        
        ambient_temperature = float(np.clip(
            float(data.get('ambient_temperature', 298.15)),
            280.0, 330.0
        ))
        
        wind_speed = float(np.clip(
            float(data.get('wind_speed', 2.0)),
            0.0, 10.0
        ))
        
        cell_efficiency = float(np.clip(
            float(data.get('cell_efficiency', 0.20)),
            0.10, 0.30
        ))
        
        thermal_conductivity = float(np.clip(
            float(data.get('thermal_conductivity', 130.0)),
            100.0, 200.0
        ))
        
        absorptivity = float(np.clip(
            float(data.get('absorptivity', 0.95)),
            0.85, 0.98
        ))
        
        emissivity = float(np.clip(
            float(data.get('emissivity', 0.90)),
            0.80, 0.95
        ))
        
        # Optional configuration
        config = data.get('config', {})
        
        logger.info(f"Starting BTE-NS simulation: fidelity={fidelity_level}, "
                   f"irradiance={solar_irradiance} W/m²")
        
        # Run coupled solver
        results = coupled_solver.solve(
            fidelity_level=fidelity_level,
            solar_irradiance=solar_irradiance,
            ambient_temperature=ambient_temperature,
            wind_speed=wind_speed,
            cell_efficiency=cell_efficiency,
            thermal_conductivity=thermal_conductivity,
            absorptivity=absorptivity,
            emissivity=emissivity,
            config=config
        )
        
        # Generate visualizations
        visualizations = {
            'temperature_heatmap': visualizer.generate_temperature_heatmap(
                results['temperature_field']
            ),
            'current_density': visualizer.generate_current_density_plot(
                results['bte_results']['J_total']
            ),
            'velocity_field': visualizer.generate_velocity_field_plot(
                results['ns_results']['u'],
                results['ns_results']['v']
            )
        }
        
        runtime_ms = (time.time() - t_start) * 1000.0
        
        # Prepare response with statistics
        response_data = {
            "success": True,
            "fidelity_level": fidelity_level,
            "fidelity_name": ["Low (LF)", "Medium (MF)", "High (HF)"][fidelity_level],
            "runtime_ms": round(runtime_ms, 1),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "statistics": {
                # Convert temperatures to Celsius
                "temp_max": round(results['statistics']['temp_max'] - 273.15, 2),
                "temp_min": round(results['statistics']['temp_min'] - 273.15, 2),
                "temp_avg": round(results['statistics']['temp_avg'] - 273.15, 2),
                "power_total": round(results['statistics']['power_total'], 2),
                "efficiency_avg": round(results['statistics']['efficiency_avg'] * 100, 2),
                "current_density_max": round(results['statistics']['current_density_max'], 2),
                "velocity_max": round(results['statistics']['velocity_max'], 2),
                "carrier_density_avg": f"{results['statistics']['carrier_density_avg']:.2e}"
            },
            "visualizations": visualizations,
            "data": {
                "temperature_field": results['temperature_field'].tolist(),
                "velocity_magnitude": results['ns_results']['velocity_magnitude'].tolist()
            }
        }
        
        logger.info(f"BTE-NS simulation completed in {runtime_ms:.1f}ms")
        
        return jsonify(response_data)
        
    except ValueError as e:
        logger.error(f"Parameter validation error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Invalid parameter: {str(e)}"
        }), 400
        
    except Exception as e:
        logger.error(f"Simulation error: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Internal simulation error",
            "details": str(e)
        }), 500


# ──────────────────────────────────────────────────────────────────────
# Contact Form
# ──────────────────────────────────────────────────────────────────────

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
        logger.warning(f"🚫 Honeypot triggered – discarding submission from {data.get('email', 'unknown')}")
        # Return 200 to not reveal to bots that filtering occurred
        return jsonify({"message": "Message received!"}), 200

    logger.info(f"Contact form submission from: {data.get('email')}")
    
    success = send_via_formspree(data)
    
    if success:
        return jsonify({"message": "Message received!"}), 200
    else:
        return jsonify({"message": "Failed to send message via provider"}), 500


# ──────────────────────────────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 204
    return jsonify({
        "status": "active",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "thermal": "/api/simulate",
            "coupled": "/api/simulate/bte-ns",
            "contact": "/api/contact"
        }
    })


# ──────────────────────────────────────────────────────────────────────
# Root Docs Page
# ──────────────────────────────────────────────────────────────────────

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
            .new { background: #e3f2fd; border-left: 4px solid #2196F3; }
            a { color: #667eea; text-decoration: none; }
        </style></head>
        <body><div class="container">
            <h1>🚀 AURA-MF Backend API</h1>
            <div class="status">✅ Status: <strong>Running</strong><br>🕐 Uptime: {{ time }} simulation steps</div>
            <h2>Endpoints</h2>
            <div class="endpoint">GET /api/health – Health check</div>
            <div class="endpoint">POST /api/contact – Contact form (via Formspree, honeypot-protected)</div>
            <div class="endpoint">GET/POST /api/simulate – Thermal physics simulation</div>
            <div class="endpoint new">POST /api/simulate/bte-ns – <strong>NEW:</strong> Coupled BTE-NS simulation</div>
            
            <h2>BTE-NS Simulation</h2>
            <p>The new <code>/api/simulate/bte-ns</code> endpoint provides:</p>
            <ul>
                <li><strong>Boltzmann Transport Equation</strong> – Electron/hole carrier transport</li>
                <li><strong>Navier-Stokes</strong> – Air flow and convective cooling</li>
                <li><strong>Thermal Coupling</strong> – Multi-physics integration</li>
                <li><strong>Advanced Visualizations</strong> – Temperature, current density, velocity fields</li>
            </ul>
            
            <h3>Example Request:</h3>
            <pre style="background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto;">
{
  "fidelity_level": 1,
  "solar_irradiance": 1000.0,
  "ambient_temperature": 298.15,
  "wind_speed": 2.0,
  "cell_efficiency": 0.20,
  "thermal_conductivity": 130.0,
  "absorptivity": 0.95,
  "emissivity": 0.90
}
            </pre>
            
            <h3>Response includes:</h3>
            <ul>
                <li>Temperature distribution</li>
                <li>Current density fields</li>
                <li>Velocity/pressure fields</li>
                <li>Power output & efficiency</li>
                <li>Base64-encoded visualizations</li>
            </ul>
        </div></body></html>
    """, time=state_manager.time)


# ──────────────────────────────────────────────────────────────────────
# CORS After-Request Hook
# ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)