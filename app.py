"""
app.py — Solar Power Output Forecasting Web App
Run: python app.py
Visit: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib
import os

app = Flask(__name__)

# ── Load model & artefacts ─────────────────────────────────────────────────
MODEL_PATH   = "best_solar_model.pkl"
COLUMNS_PATH = "feature_columns.pkl"
SCALER_PATH  = "feat_scaler.pkl"          # see note below — save this too

model          = joblib.load(MODEL_PATH)   if os.path.exists(MODEL_PATH)   else None
feature_cols   = joblib.load(COLUMNS_PATH) if os.path.exists(COLUMNS_PATH) else []
feat_scaler    = joblib.load(SCALER_PATH)  if os.path.exists(SCALER_PATH)  else None

# ── HTML (single-file — no templates folder needed) ───────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Solar Power Forecast</title>
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <style>
    :root{
      --sun:#FFB830; --sky:#0A1628; --mid:#0F2040;
      --card:#122040; --border:#1E3A5F;
      --text:#E8F0FE; --muted:#6B8BBF; --accent:#FFB830;
    }
    *{box-sizing:border-box;margin:0;padding:0}
    body{
      font-family:'IBM Plex Mono',monospace;
      background:var(--sky); color:var(--text);
      min-height:100vh;
      background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(255,184,48,.18) 0%, transparent 65%);
    }

    /* ── Header ── */
    header{
      text-align:center; padding:3rem 1rem 2rem;
      border-bottom:1px solid var(--border);
    }
    .sun-icon{
      width:64px;height:64px;margin:0 auto 1rem;
      background:var(--sun);border-radius:50%;
      box-shadow:0 0 0 12px rgba(255,184,48,.15),
                 0 0 0 24px rgba(255,184,48,.07);
      animation:pulse 3s ease-in-out infinite;
    }
    @keyframes pulse{0%,100%{box-shadow:0 0 0 12px rgba(255,184,48,.15),0 0 0 24px rgba(255,184,48,.07)}
                     50%{box-shadow:0 0 0 18px rgba(255,184,48,.2),0 0 0 36px rgba(255,184,48,.08)}}
    header h1{font-family:'Syne',sans-serif;font-size:clamp(1.6rem,4vw,2.6rem);font-weight:800;letter-spacing:-1px}
    header h1 span{color:var(--sun)}
    header p{color:var(--muted);margin-top:.5rem;font-size:.85rem}

    /* ── Main layout ── */
    main{max-width:900px;margin:2.5rem auto;padding:0 1.5rem 4rem}

    /* ── Section labels ── */
    .section-label{
      font-family:'Syne',sans-serif;font-size:.7rem;font-weight:600;
      letter-spacing:.15em;text-transform:uppercase;color:var(--sun);
      margin-bottom:1rem;padding-bottom:.4rem;border-bottom:1px solid var(--border);
    }

    /* ── Grid of inputs ── */
    .grid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));margin-bottom:2rem}

    .field label{
      display:block;font-size:.72rem;color:var(--muted);
      margin-bottom:.35rem;letter-spacing:.04em;
    }
    .field input{
      width:100%;background:var(--mid);border:1px solid var(--border);
      border-radius:6px;padding:.6rem .85rem;color:var(--text);
      font-family:'IBM Plex Mono',monospace;font-size:.9rem;
      transition:border-color .2s,box-shadow .2s;
    }
    .field input:focus{
      outline:none;border-color:var(--sun);
      box-shadow:0 0 0 3px rgba(255,184,48,.15);
    }
    .field input::placeholder{color:#3a5070}

    /* ── Submit ── */
    .btn-wrap{display:flex;justify-content:center;margin-top:.5rem}
    button[type=submit]{
      font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;
      background:var(--sun);color:var(--sky);border:none;
      padding:.85rem 3.5rem;border-radius:8px;cursor:pointer;
      transition:transform .15s,box-shadow .15s;letter-spacing:.03em;
    }
    button[type=submit]:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(255,184,48,.4)}
    button[type=submit]:active{transform:translateY(0)}

    /* ── Result card ── */
    #result{
      display:none;margin-top:2.5rem;
      background:var(--card);border:1px solid var(--border);
      border-radius:12px;padding:2rem;text-align:center;
      animation:fadeUp .4s ease;
    }
    @keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
    #result .result-label{font-size:.7rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase}
    #result .result-value{
      font-family:'Syne',sans-serif;font-size:3.5rem;font-weight:800;
      color:var(--sun);margin:.4rem 0 .25rem;
    }
    #result .result-unit{color:var(--muted);font-size:.85rem}
    #result.error .result-value{color:#ff6b6b;font-size:1.2rem}

    /* ── Spinner ── */
    .spinner{display:none;width:24px;height:24px;
      border:3px solid rgba(255,184,48,.3);border-top-color:var(--sun);
      border-radius:50%;animation:spin .7s linear infinite;margin:0 auto}
    @keyframes spin{to{transform:rotate(360deg)}}
  </style>
</head>
<body>

<header>
  <div class="sun-icon"></div>
  <h1>Solar <span>Power</span> Forecast</h1>
  <p>XGBoost · Norway Hourly PV Dataset · Final Year B.E./B.Tech Project</p>
</header>

<main>
  <form id="predict-form">

    <!-- ── Sensor readings ── -->
    <div class="section-label">Sensor Readings</div>
    <div class="grid">
      <div class="field">
        <label>Irradiation (W/m²)</label>
        <input type="number" name="Irradiation" step="0.001" placeholder="e.g. 0.42" required/>
      </div>
      <div class="field">
        <label>Module Temperature (°C)</label>
        <input type="number" name="Module_Temperature" step="0.01" placeholder="e.g. 35.2" required/>
      </div>
      <div class="field">
        <label>Ambient Temperature (°C)</label>
        <input type="number" name="Ambient_Temperature" step="0.01" placeholder="e.g. 24.5"/>
      </div>
    </div>

    <!-- ── Temporal context ── -->
    <div class="section-label">Temporal Context</div>
    <div class="grid">
      <div class="field">
        <label>Hour (0–23)</label>
        <input type="number" name="Hour" min="0" max="23" placeholder="e.g. 12" required/>
      </div>
      <div class="field">
        <label>Day of Week (0=Mon)</label>
        <input type="number" name="DayOfWeek" min="0" max="6" placeholder="e.g. 2" required/>
      </div>
      <div class="field">
        <label>Month (1–12)</label>
        <input type="number" name="Month" min="1" max="12" placeholder="e.g. 6" required/>
      </div>
      <div class="field">
        <label>Day of Year (1–365)</label>
        <input type="number" name="DayOfYear" min="1" max="365" placeholder="e.g. 160" required/>
      </div>
      <div class="field">
        <label>Week of Year (1–53)</label>
        <input type="number" name="WeekOfYear" min="1" max="53" placeholder="e.g. 23"/>
      </div>
    </div>

    <!-- ── Lag features ── -->
    <div class="section-label">Previous AC Power Readings (kW)</div>
    <div class="grid">
      {% for lag in [1,2,3,6,12,24] %}
      <div class="field">
        <label>AC Power — {{ lag }}h ago</label>
        <input type="number" name="AC_lag_{{ lag }}" step="0.01" placeholder="e.g. 1500" required/>
      </div>
      {% endfor %}
    </div>

    <!-- ── Rolling stats ── -->
    <div class="section-label">Rolling Statistics</div>
    <div class="grid">
      {% for w in [3,6,12,24] %}
      <div class="field">
        <label>Rolling Mean {{ w }}h</label>
        <input type="number" name="roll_mean_{{ w }}" step="0.01" placeholder="e.g. 1450"/>
      </div>
      <div class="field">
        <label>Rolling Std {{ w }}h</label>
        <input type="number" name="roll_std_{{ w }}" step="0.01" placeholder="e.g. 200"/>
      </div>
      {% endfor %}
    </div>

    <!-- ── EWM ── -->
    <div class="section-label">Exponential Weighted Mean</div>
    <div class="grid">
      <div class="field">
        <label>EWM span=6</label>
        <input type="number" name="ewm_6" step="0.01" placeholder="e.g. 1480"/>
      </div>
      <div class="field">
        <label>EWM span=24</label>
        <input type="number" name="ewm_24" step="0.01" placeholder="e.g. 1420"/>
      </div>
    </div>

    <div class="btn-wrap">
      <button type="submit">⚡ Predict AC Power</button>
    </div>
  </form>

  <!-- Result -->
  <div id="result">
    <div class="spinner" id="spinner"></div>
    <div id="result-inner">
      <div class="result-label">Predicted AC Power Output</div>
      <div class="result-value" id="result-value">—</div>
      <div class="result-unit">kilowatts (kW)</div>
    </div>
  </div>
</main>

<script>
const form   = document.getElementById('predict-form');
const box    = document.getElementById('result');
const valEl  = document.getElementById('result-value');
const spinner = document.getElementById('spinner');
const inner  = document.getElementById('result-inner');

form.addEventListener('submit', async e => {
  e.preventDefault();

  // Collect form data
  const raw = Object.fromEntries(new FormData(form).entries());
  const data = {};
  for (const [k, v] of Object.entries(raw)) {
    data[k] = v === '' ? 0 : parseFloat(v);
  }

  // Compute derived features from raw inputs
  const h = data['Hour'], m = data['Month'], doy = data['DayOfYear'];
  data['Hour_sin']  = Math.sin(2 * Math.PI * h / 24);
  data['Hour_cos']  = Math.cos(2 * Math.PI * h / 24);
  data['Month_sin'] = Math.sin(2 * Math.PI * m / 12);
  data['Month_cos'] = Math.cos(2 * Math.PI * m / 12);
  data['DOY_sin']   = Math.sin(2 * Math.PI * doy / 365);
  data['DOY_cos']   = Math.cos(2 * Math.PI * doy / 365);

  if (data['Irradiation'] && data['Module_Temperature']) {
    data['Irrad_x_Temp'] = data['Irradiation'] * (1 - 0.004 * data['Module_Temperature']);
  }

  // Show spinner
  box.style.display = 'block';
  box.classList.remove('error');
  spinner.style.display = 'block';
  inner.style.display = 'none';

  try {
    const res  = await fetch('/predict', {
      method : 'POST',
      headers: {'Content-Type':'application/json'},
      body   : JSON.stringify(data)
    });
    const json = await res.json();

    spinner.style.display = 'none';
    inner.style.display = 'block';

    if (json.error) {
      box.classList.add('error');
      valEl.textContent = json.error;
    } else {
      valEl.textContent = json.prediction.toFixed(2);
    }
  } catch(err) {
    spinner.style.display = 'none';
    inner.style.display = 'block';
    box.classList.add('error');
    valEl.textContent = 'Server error';
  }
});
</script>
</body>
</html>
"""

# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run the training notebook first."})

    data = request.get_json()

    try:
        # Build feature vector in the exact column order the model was trained on
        row = np.array([[data.get(col, 0.0) for col in feature_cols]], dtype=float)

        # Scale if scaler is available
        if feat_scaler is not None:
            row = feat_scaler.transform(row)

        pred = float(model.predict(row)[0])
        pred = max(pred, 0.0)          # AC Power can't be negative

        return jsonify({"prediction": round(pred, 4)})

    except Exception as ex:
        return jsonify({"error": str(ex)})


if __name__ == "__main__":
    print("Starting Solar Forecast server at http://localhost:5000")
    app.run(debug=True, port=5000)