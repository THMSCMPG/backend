import os
import logging
import requests
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from datetime import datetime


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


# ============================================================================
# API ROUTES
# ============================================================================

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
# Error Logging
# ──────────────────────────────────────────────────────────────────────

@app.route('/api/log-error', methods=['POST', 'OPTIONS'])
def handle_error_log():
    """Frontend error telemetry endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json or {}
        error = data.get('error', 'Unknown error')
        context = data.get('context', {})
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        logger.error(f"Frontend error [{timestamp}]: {error} | Context: {context}")
        
        return jsonify({"message": "Error logged"}), 200
    except Exception as e:
        logger.error(f"Failed to log frontend error: {str(e)}")
        # Always return 200 for telemetry endpoints
        return jsonify({"message": "Error logged"}), 200


# ──────────────────────────────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 204
    return jsonify({
        "status": "active",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "contact": "/api/contact",
            "health": "/api/health"
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
            .info { background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #2196F3; }
            a { color: #667eea; text-decoration: none; }
        </style></head>
        <body><div class="container">
            <h1>🚀 AURA-MF Backend API</h1>
            <div class="status">✅ Status: <strong>Running</strong><br>📡 Mode: Communication-Only Backend (v3.0.0)</div>
            
            <div class="info">
                <strong>Architecture Update:</strong> Physics simulation now runs entirely client-side in JavaScript. 
                This backend provides communication and logging services only.
            </div>
            
            <h2>Available Endpoints</h2>
            <div class="endpoint">GET /api/health – Health check and status</div>
            <div class="endpoint">POST /api/contact – Contact form submission (via Formspree, honeypot-protected)</div>
            <div class="endpoint">POST /api/log-error – Frontend error telemetry logging</div>
            
            <h2>Contact Form Endpoint</h2>
            <p>The <code>/api/contact</code> endpoint handles contact form submissions from the frontend:</p>
            <ul>
                <li><strong>Method:</strong> POST</li>
                <li><strong>Content-Type:</strong> application/json</li>
                <li><strong>Honeypot Protection:</strong> Server-side filtering enabled</li>
                <li><strong>Provider:</strong> Formspree integration</li>
            </ul>
            
            <h3>Example Request:</h3>
            <pre style="background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto;">
{
  "name": "John Doe",
  "email": "john@example.com",
  "message": "Hello, I have a question...",
  "website_hp": ""
}
            </pre>
            
            <h2>Error Logging Endpoint</h2>
            <p>The <code>/api/log-error</code> endpoint accepts error telemetry from the frontend:</p>
            <ul>
                <li><strong>Method:</strong> POST</li>
                <li><strong>Content-Type:</strong> application/json</li>
                <li><strong>Purpose:</strong> Server-side logging of frontend errors</li>
            </ul>
            
            <h3>Example Request:</h3>
            <pre style="background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto;">
{
  "error": "TypeError: Cannot read property 'x' of undefined",
  "context": {
    "component": "SimulationEngine",
    "userAgent": "Mozilla/5.0..."
  },
  "timestamp": "2026-02-08T01:48:00.000Z"
}
            </pre>
            
            <h2>Health Check Response</h2>
            <pre style="background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto;">
{
  "status": "active",
  "version": "3.0.0",
  "timestamp": "2026-02-08T01:48:00.000000",
  "endpoints": {
    "contact": "/api/contact",
    "health": "/api/health"
  }
}
            </pre>
        </div></body></html>
    """)


# ──────────────────────────────────────────────────────────────────────
# CORS After-Request Hook
# ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)