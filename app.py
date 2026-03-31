import streamlit as st
import numpy as np
import pandas as pd
import json
import statistics
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(page_title="MachineGuard AI", page_icon="⚙", layout="wide")

DEFAULT_API_KEY = "your_api_key"

# Custom CSS for Bloomberg Terminal style with dark industrial aesthetics
st.markdown("""
<style>
:root {
  --bg-primary: #080c12;
  --bg-secondary: #0d1117;
  --bg-card: #111827;
  --border: #1e2d40;
  --accent-blue: #38bdf8;
}
.stApp { background-color: var(--bg-primary); }
div[data-testid="stSidebar"] { background-color: var(--bg-secondary); }
.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.card:hover { border-color: var(--accent-blue); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.status-dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
.dot-normal { background: #4ade80; animation: pulse 2s infinite; }
.dot-warning { background: #fbbf24; animation: pulse 1.5s infinite; }
.dot-critical { background: #ef4444; animation: pulse 0.8s infinite; }
.dot-emergency { background: #ef4444; animation: pulse 0.4s infinite; box-shadow: 0 0 10px #ef4444; }
.trace-box {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; background: #080c12;
    border: 1px solid #1e2d40; border-left: 3px solid var(--accent-blue);
    border-radius: 6px; padding: 12px 16px; min-height: 100px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# SECTION 1: CORE ARCHITECTURE (TOOLS)
# ==========================================
def analyze_temperature(temp_data: list) -> dict:
    """Tool 1: Analyze temperature data."""
    avg = statistics.mean(temp_data)
    mx, mn = max(temp_data), min(temp_data)
    std = statistics.stdev(temp_data) if len(temp_data) > 1 else 0
    recent, old = temp_data[-5:], temp_data[:5]
    roc = statistics.mean(recent) - statistics.mean(old) if old else 0
    hotspots = sum(1 for t in temp_data if t > 75) / len(temp_data) * 100
    
    if avg > 90: status = "EMERGENCY"
    elif avg > 75: status = "CRITICAL"
    elif avg > 60: status = "WARNING"
    else: status = "NORMAL"
    
    return {"avg": avg, "max": mx, "min": mn, "std_dev": std, "rate_of_change": roc, "hotspot_index": hotspots, "status": status}

def analyze_vibration(vib_data: list) -> dict:
    """Tool 2: Analyze vibration data."""
    mx = max(vib_data)
    avg = statistics.mean(vib_data)
    rms = np.sqrt(np.mean(np.square(vib_data)))
    crest = mx / rms if rms else 0
    
    n = len(vib_data)
    mean_v = np.mean(vib_data)
    kurtosis = sum((x - mean_v)**4 for x in vib_data) / (n * statistics.stdev(vib_data)**4) if n>1 and statistics.stdev(vib_data)>0 else 3.0
    
    if mx > 15: status = "CATASTROPHIC"
    elif mx > 8: status = "CRITICAL"
    elif mx > 5: status = "WARNING"
    else: status = "NORMAL"
    
    return {"peak": mx, "avg": avg, "rms": rms, "crest_factor": crest, "kurtosis": kurtosis, "dominant_frequency_band": "mid", "status": status}

def analyze_oil_pressure(pressure_data: list) -> dict:
    """Tool 3: Analyze oil pressure."""
    avg = statistics.mean(pressure_data)
    mn = min(pressure_data)
    drop_rate = (pressure_data[0] - pressure_data[-1]) / len(pressure_data)
    std = statistics.stdev(pressure_data) if len(pressure_data) > 1 else 0
    stability = max(0, 1 - (std / avg)) if avg else 0
    
    if avg < 20: status = "LOW"
    elif avg > 80: status = "HIGH"
    else: status = "NORMAL"
    
    return {"avg_pressure": avg, "min_pressure": mn, "pressure_drop_rate": drop_rate, "stability_score": stability, "status": status}

def detect_anomalies(temp_data: list, vib_data: list, pressure_data: list) -> dict:
    """Tool 4: Z-score anomaly detection across signals."""
    def get_z(arr):
        if np.std(arr) == 0: return np.zeros(len(arr))
        return np.abs((arr - np.mean(arr)) / np.std(arr))
    
    z_t, z_v, z_p = get_z(temp_data), get_z(vib_data), get_z(pressure_data)
    combined = z_t + z_v + z_p
    anomaly_indices = [i for i, val in enumerate(combined) if val > 5.0]
    
    corr = np.corrcoef(temp_data, vib_data)[0,1] if np.std(temp_data) and np.std(vib_data) else 0
    severity = "SEVERE" if len(anomaly_indices) > 5 else ("MODERATE" if len(anomaly_indices) > 2 else "MILD")
    
    return {"num_anomalies": len(anomaly_indices), "anomaly_indices": anomaly_indices, "anomaly_severity": severity, "cross_correlation_score": corr}

def calculate_rul(temp_avg: float, vib_peak: float, pressure_avg: float) -> dict:
    """Tool 5: Remaining Useful Life estimator."""
    base_rul = 720.0
    temp_factor = max(0.1, 1 - (temp_avg - 60) / 60) if temp_avg > 60 else 1.0
    vib_factor = max(0.1, 1 - (vib_peak - 5) / 15) if vib_peak > 5 else 1.0
    press_factor = max(0.1, 1 - abs(50 - pressure_avg) / 50)
    
    rul_hours = base_rul * temp_factor * vib_factor * press_factor
    health_score = min(100.0, (rul_hours / base_rul) * 100.0)
    
    fail_7d = max(0.0, 100 - (rul_hours / 168) * 100) if rul_hours < 168 else 0.0
    fail_30d = max(0.0, 100 - (rul_hours / 720) * 100) if rul_hours < 720 else 0.0
    
    return {"rul_hours": rul_hours, "rul_days": rul_hours / 24, "health_score": health_score, "failure_probability_7d": fail_7d, "failure_probability_30d": fail_30d}

def generate_maintenance_schedule(rul_hours: float, status: str) -> dict:
    """Tool 6: Recommend maintenance schedule."""
    if status in ["CRITICAL", "CATASTROPHIC", "EMERGENCY"] or rul_hours < 48:
        m_type = "Emergency"
        priority = "P1"
        inspect = 0.0
        parts = ["Bearings", "Cooling System", "Hydraulic Seals"]
        down = 24.0
    elif status == "WARNING" or rul_hours < 168:
        m_type = "Urgent"
        priority = "P2"
        inspect = 24.0
        parts = ["Vibration Dampers", "Oil Filter"]
        down = 8.0
    else:
        m_type = "Routine"
        priority = "P4"
        inspect = rul_hours * 0.5
        parts = ["Visual Inspection", "Lubrication"]
        down = 2.0
        
    return {"next_inspection_hours": inspect, "maintenance_type": m_type, "parts_to_check": parts, "estimated_downtime_hours": down, "priority_level": priority}

TOOL_REGISTRY = {
    "analyze_temperature": analyze_temperature,
    "analyze_vibration": analyze_vibration,
    "analyze_oil_pressure": analyze_oil_pressure,
    "detect_anomalies": detect_anomalies,
    "calculate_rul": calculate_rul,
    "generate_maintenance_schedule": generate_maintenance_schedule
}

tools_schema = [
    {"type": "function", "function": {"name": "analyze_temperature", "description": "Analyze an array of temperature readings.", "parameters": {"type": "object", "properties": {"temp_data": {"type": "array", "items": {"type": "number"}}}, "required": ["temp_data"]}}},
    {"type": "function", "function": {"name": "analyze_vibration", "description": "Analyze an array of vibration readings.", "parameters": {"type": "object", "properties": {"vib_data": {"type": "array", "items": {"type": "number"}}}, "required": ["vib_data"]}}},
    {"type": "function", "function": {"name": "analyze_oil_pressure", "description": "Analyze an array of oil pressure readings.", "parameters": {"type": "object", "properties": {"pressure_data": {"type": "array", "items": {"type": "number"}}}, "required": ["pressure_data"]}}},
    {"type": "function", "function": {"name": "detect_anomalies", "description": "Z-score anomaly detection across all three signals combined.", "parameters": {"type": "object", "properties": {"temp_data": {"type": "array", "items": {"type": "number"}}, "vib_data": {"type": "array", "items": {"type": "number"}}, "pressure_data": {"type": "array", "items": {"type": "number"}}}, "required": ["temp_data", "vib_data", "pressure_data"]}}},
    {"type": "function", "function": {"name": "calculate_rul", "description": "Calculate remaining useful life (RUL) estimator.", "parameters": {"type": "object", "properties": {"temp_avg": {"type": "number"}, "vib_peak": {"type": "number"}, "pressure_avg": {"type": "number"}}, "required": ["temp_avg", "vib_peak", "pressure_avg"]}}},
    {"type": "function", "function": {"name": "generate_maintenance_schedule", "description": "Generate maintenance schedule based on RUL and severity.", "parameters": {"type": "object", "properties": {"rul_hours": {"type": "number"}, "status": {"type": "string"}}, "required": ["rul_hours", "status"]}}}
]

# ==========================================
# SECTION 2: DATA SIMULATION ENGINE
# ==========================================
def generate_sensor_arrays(base_temp, base_vib, base_pressure, machine_mode, num_samples=50):
    t = np.arange(num_samples)
    
    if machine_mode == "Healthy":
        t_arr = np.random.normal(base_temp, base_temp * 0.03, num_samples)
        v_arr = np.random.normal(base_vib, base_vib * 0.03, num_samples)
        p_arr = np.random.normal(base_pressure, base_pressure * 0.03, num_samples)
    elif machine_mode == "Degrading":
        t_arr = np.random.normal(base_temp + (t * 0.1), base_temp * 0.08, num_samples)
        v_arr = np.random.normal(base_vib + (t * 0.05), base_vib * 0.08, num_samples)
        p_arr = np.random.normal(base_pressure, base_pressure * 0.08, num_samples)
    elif machine_mode == "Bearing Fault":
        t_arr = np.random.normal(base_temp, base_temp * 0.03, num_samples)
        v_arr = np.random.normal(base_vib, base_vib * 0.03, num_samples)
        v_arr[::7] += base_vib * 2.0  # Periodic spikes
        p_arr = np.random.normal(base_pressure, base_pressure * 0.03, num_samples)
    elif machine_mode == "Overheating":
        t_arr = np.random.normal(base_temp + (t * 0.5), base_temp * 0.05, num_samples)
        t_arr[::10] += 15.0  # Burst spikes
        v_arr = np.random.normal(base_vib, base_vib * 0.03, num_samples)
        p_arr = np.random.normal(base_pressure, base_pressure * 0.03, num_samples)
    elif machine_mode == "Cavitation":
        t_arr = np.random.normal(base_temp, base_temp * 0.03, num_samples)
        v_arr = np.random.normal(base_vib, base_vib * 0.1, num_samples)
        v_arr += np.sin(t * 10) * 2  # High-freq
        p_arr = np.random.normal(base_pressure, base_pressure * 0.05, num_samples)
        p_arr += np.sin(t * 0.5) * 15 # Oscillation
    
    # Clip negative values where it doesn't make sense
    t_arr, v_arr, p_arr = np.maximum(t_arr, 10), np.maximum(v_arr, 0), np.maximum(p_arr, 0)
    
    now = time.time()
    ts = [(datetime.fromtimestamp(now - (num_samples - i))).strftime("%H:%M:%S") for i in range(num_samples)]
    return t_arr.tolist(), v_arr.tolist(), p_arr.tolist(), ts

# ==========================================
# SECTION 7: SESSION STATE
# ==========================================
if "initialized" not in st.session_state:
    st.session_state.update({
        "initialized": True,
        "machines": {
            "Press A7": {"temp": 65, "vib": 4, "press": 50, "last_status": "NORMAL", "last_run": "Never", "diagnosis": None, "mode": "Healthy"},
            "Pump B3": {"temp": 50, "vib": 2, "press": 65, "last_status": "NORMAL", "last_run": "Never", "diagnosis": None, "mode": "Healthy"},
            "Compressor C1": {"temp": 85, "vib": 9, "press": 35, "last_status": "WARNING", "last_run": "Never", "diagnosis": None, "mode": "Healthy"}
        },
        "diagnosis_history": [],
        "active_machine": "Press A7",
        "api_key": DEFAULT_API_KEY,
        "llm_model": "moonshotai/kimi-k2.5",
        "llm_temperature": 0.2,
        "confidence_threshold": 0.75,
        "fft_enabled": True,
        "bearing_detection_enabled": True,
        "run_count": 0,
        "correction_triggered": False,
        "trace": []
    })

def clear_trace():
    st.session_state["trace"] = []

def append_trace(msg_type, content, text_color):
    st.session_state["trace"].append(f"<span style='color:{text_color}'><b>[{msg_type}]</b> {content}</span>")

# ==========================================
# HEADER & CLOCK (Sections 5 & 9)
# ==========================================
col_title, col_clock = st.columns([3, 1])
with col_title:
    st.markdown("<h2 style='margin-bottom:0'>⚙ MachineGuard AI</h2><p style='color:var(--text-muted);font-family:Orbitron'>AUTONOMOUS PREDICTIVE MAINTENANCE SYSTEM</p>", unsafe_allow_html=True)
    dots = []
    for m, d in st.session_state.machines.items():
        st_color = "dot-normal" if d["last_status"] == "NORMAL" else ("dot-warning" if d["last_status"] == "WARNING" else "dot-critical")
        dots.append(f"<span class='status-dot {st_color}'></span><span style='color:var(--text-muted);margin-right:15px;font-size:0.8rem'>{m}</span>")
    st.markdown(f"<div>{''.join(dots)}</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:0;height:1px;background:linear-gradient(to right, #38bdf8, transparent)'>", unsafe_allow_html=True)

with col_clock:
    st.components.v1.html("""
    <div style="font-family:'IBM Plex Mono';color:#38bdf8;font-size:1.1rem;text-align:right;padding:4px 0">
      <span id="clk"></span>
    </div>
    <script>
      function tick(){
        const n=new Date();
        document.getElementById('clk').textContent =
          n.toLocaleTimeString('en-GB',{hour12:false})+' | '+n.toLocaleDateString('en-GB');
      }
      tick(); setInterval(tick,1000);
    </script>
    """, height=40, scrolling=False)

# ==========================================
# SECTION 4: SIDEBAR
# ==========================================
st.sidebar.markdown("### 🎛️ Control Panel")
machine = st.sidebar.selectbox("Active Machine", list(st.session_state.machines.keys()))
st.session_state.active_machine = machine
active_data = st.session_state.machines[machine]

st.sidebar.markdown("---")
active_data["mode"] = st.sidebar.selectbox("Machine Mode / Condition", ["Healthy", "Degrading", "Bearing Fault", "Overheating", "Cavitation"], index=["Healthy", "Degrading", "Bearing Fault", "Overheating", "Cavitation"].index(active_data["mode"]))
active_data["temp"] = st.sidebar.slider("🌡️ Base Temp (°C)", 20, 120, active_data["temp"])
active_data["vib"] = st.sidebar.slider("📳 Base Vib (mm/s)", 0, 20, active_data["vib"])
active_data["press"] = st.sidebar.slider("🔧 Base Pressure (PSI)", 0, 120, active_data["press"])

st.sidebar.markdown("---")
btn_run = st.sidebar.button("🚀 Run Full AI Diagnostics", use_container_width=True, type="primary")
btn_refresh = st.sidebar.button("🔄 Refresh Sensor Data", use_container_width=True)
btn_quick = st.sidebar.button("📊 Quick Health Check", use_container_width=True)

st.sidebar.markdown(f"""
<div style="font-family:'IBM Plex Mono';font-size:0.75rem;color:#64748b;margin-top:40px">
Last run: {active_data["last_run"]}<br>
Model: {st.session_state.llm_model}<br>
Loop: ReAct v2.0<br>
Status: {active_data["last_status"]}
</div>
""", unsafe_allow_html=True)

# Generate Live Data arrays
temp_arr, vib_arr, press_arr, ts_arr = generate_sensor_arrays(active_data["temp"], active_data["vib"], active_data["press"], active_data["mode"], num_samples=50)

# Quick Check Logic
if btn_quick:
    res_t, res_v, res_p = analyze_temperature(temp_arr), analyze_vibration(vib_arr), analyze_oil_pressure(press_arr)
    all_stat = [res_t["status"], res_v["status"], res_p["status"]]
    if any(s in ["CRITICAL", "CATASTROPHIC", "EMERGENCY"] for s in all_stat): st.sidebar.error("⚠️ Maintenance required")
    elif "WARNING" in all_stat: st.sidebar.warning("🔔 Monitor closely")
    else: st.sidebar.success("✅ Machine operating normally")

# ==========================================
# SECTION 1B & 1C: AI DIAGNOSTICS LOGIC
# ==========================================
if btn_run:
    st.session_state.run_count += 1
    clear_trace()
    append_trace("SENSE", f"Reading 50 samples from {machine}. Mode: {active_data['mode']}", "#22d3ee")
    
    with st.spinner("AI is analyzing signals..."):
        try:
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=st.session_state.api_key, timeout=60.0)
            sensor_input = f"Temperature: avg {np.mean(temp_arr):.1f}, max {np.max(temp_arr):.1f}. Vib: peak {np.max(vib_arr):.1f}. Press: avg {np.mean(press_arr):.1f}."
            
            system_prompt = """You are MachineGuard AI, an expert predictive maintenance engineer.
            1. Use ALL provided tools to analyze the sensor data array summaries.
            2. Output ONLY a valid JSON object with EXACT keys: status, predicted_failure, fault_type, root_cause, affected_components, confidence (as a float between 0.0 and 1.0), severity_score, recommended_action, urgency, estimated_rul_hours, safety_risk.
            3. Do not include markdown formatting like ```json."""
            
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"New sensor data summary: {sensor_input}"}]
            
            append_trace("THINK", "Invoking LLM for tool selection...", "#38bdf8")
            response = client.chat.completions.create(model=st.session_state.llm_model, messages=messages, tools=tools_schema, tool_choice="auto")
            msg = response.choices[0].message
            msg_dict = {"role": msg.role, "content": msg.content or ""}
            if msg.tool_calls:
                msg_dict["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]
            messages.append(msg_dict)
            
            tool_outputs = {}
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    append_trace("ACT", f"Executing {fn_name}", "#fbbf24")
                    
                    if fn_name == "analyze_temperature": res = TOOL_REGISTRY[fn_name](temp_arr)
                    elif fn_name == "analyze_vibration": res = TOOL_REGISTRY[fn_name](vib_arr)
                    elif fn_name == "analyze_oil_pressure": res = TOOL_REGISTRY[fn_name](press_arr)
                    elif fn_name == "detect_anomalies": res = TOOL_REGISTRY[fn_name](temp_arr, vib_arr, press_arr)
                    elif fn_name == "calculate_rul": res = TOOL_REGISTRY[fn_name](np.mean(temp_arr), np.max(vib_arr), np.mean(press_arr))
                    elif fn_name == "generate_maintenance_schedule":
                        # Needs rul from calculate_rul if available, we pass dummies if not
                        res = TOOL_REGISTRY[fn_name](args.get("rul_hours", 720), args.get("status", "NORMAL"))
                    
                    tool_outputs[fn_name] = res
                    messages.append({"tool_call_id": tc.id, "role": "tool", "name": fn_name, "content": json.dumps(res)})
                    append_trace("RESULT", f"{fn_name} returned status/metrics", "#4ade80")
                
                append_trace("THINK", "Synthesizing final diagnostic JSON...", "#38bdf8")
                messages.append({"role": "user", "content": "You have received the tool outputs. Do NOT call any more tools. You MUST now output the final diagnostic JSON exactly matching the requested format."})
                final_res = client.chat.completions.create(model=st.session_state.llm_model, messages=messages)
                raw_text = final_res.choices[0].message.content or ""
                raw_json = raw_text[raw_text.find('{'):raw_text.rfind('}')+1] if '{' in raw_text else raw_text
                
                try:
                    report = json.loads(raw_json)
                    append_trace("VERIFY", f"Valid JSON parsed. Confidence: {report.get('confidence')}", "#f97316")
                    
                    # Self-Correction Loop
                    st.session_state.correction_triggered = False
                    if float(report.get("confidence", 0)) < st.session_state.confidence_threshold:
                        st.session_state.correction_triggered = True
                        append_trace("CORRECT", "Confidence too low. Asking for revision.", "#ef4444")
                        messages.append({"role": "user", "content": "Your confidence is low. Please re-evaluate based on the tool results and provide a revised JSON."})
                        rev_res = client.chat.completions.create(model=st.session_state.llm_model, messages=messages)
                        raw_text = rev_res.choices[0].message.content or ""
                        raw_json = raw_text[raw_text.find('{'):raw_text.rfind('}')+1] if '{' in raw_text else raw_text
                        report = json.loads(raw_json)
                        
                    if report.get("status") in ["CRITICAL", "EMERGENCY"] and "shutdown" not in str(report.get("recommended_action")).lower():
                        st.session_state.correction_triggered = True
                        append_trace("CORRECT", "Missing shutdown instruction in CRITICAL state.", "#ef4444")
                        messages.append({"role": "user", "content": "This is a CRITICAL fault. Your recommended action MUST include immediate shutdown procedures. Replace the output JSON."})
                        rev_res = client.chat.completions.create(model=st.session_state.llm_model, messages=messages)
                        raw_text = rev_res.choices[0].message.content or ""
                        raw_json = raw_text[raw_text.find('{'):raw_text.rfind('}')+1] if '{' in raw_text else raw_text
                        report = json.loads(raw_json)
                    
                    append_trace("DONE", "Analysis Complete.", "#4ade80")
                    
                    # Update State
                    report["tool_outputs"] = tool_outputs
                    report["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    active_data["diagnosis"] = report
                    active_data["last_status"] = report.get("status", "UNKNOWN")
                    active_data["last_run"] = report["timestamp"]
                    
                    # Push to history
                    hist_entry = {
                        "Timestamp": report["timestamp"], "Machine": machine, "Status": report.get("status"), 
                        "Confidence": float(report.get("confidence", 0)), "Predicted Failure": report.get("predicted_failure"), 
                        "RUL (h)": report.get("estimated_rul_hours"), "Action": report.get("recommended_action")
                    }
                    st.session_state.diagnosis_history.insert(0, hist_entry)
                    st.session_state.diagnosis_history = st.session_state.diagnosis_history[:20]
                    
                except json.JSONDecodeError:
                    st.error("LLM did not output valid JSON.")
                    st.code(raw_json)
            else:
                st.warning("LLM didn't call any tools.")
        except Exception as e:
            st.error(f"API Error. Check your API Key. {str(e)}")

# ==========================================
# SECTION 6: PLOTLY CHART THEME
# ==========================================
def p_layout():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Mono', color='#94a3b8', size=11),
        margin=dict(l=40, r=20, t=30, b=30),
        xaxis=dict(gridcolor='#1e2d40', zerolinecolor='#1e2d40'),
        yaxis=dict(gridcolor='#1e2d40', zerolinecolor='#1e2d40'),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0),
        height=280
    )


# ==========================================
# UI TABS
# ==========================================
tabs = st.tabs(["📡 Live Sensors", "📊 Health & RUL", "🔍 Anomaly Analysis", "🤖 AI Diagnostics", "📋 History & Export", "⚙️ Settings"])

with tabs[0]: # Live Sensors
    c1, c2, c3 = st.columns(3)
    
    # Temperature Chart
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=ts_arr, y=temp_arr, mode='lines', fill='tozeroy', name='Temp', line=dict(color='#ef4444')))
    fig_t.add_hline(y=60, line_dash="dash", line_color="#fbbf24")
    fig_t.add_hline(y=75, line_dash="dash", line_color="#ef4444")
    fig_t.update_layout(**p_layout(), title="Temperature (°C)")
    c1.plotly_chart(fig_t, use_container_width=True)
    
    # Vibration Chart
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=ts_arr, y=vib_arr, mode='lines', name='Vib', line=dict(color='#22d3ee')))
    fig_v.add_hline(y=5, line_dash="dash", line_color="#fbbf24")
    fig_v.add_hline(y=8, line_dash="dash", line_color="#ef4444")
    # Mark spikes
    spikes_idx = [i for i, v in enumerate(vib_arr) if v > 8]
    if spikes_idx: fig_v.add_trace(go.Scatter(x=[ts_arr[i] for i in spikes_idx], y=[vib_arr[i] for i in spikes_idx], mode='markers', marker=dict(color='red', size=8), name='Spike'))
    fig_v.update_layout(**p_layout(), title="Vibration (mm/s)")
    c2.plotly_chart(fig_v, use_container_width=True)

    # Pressure Chart
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=ts_arr, y=press_arr, mode='lines', name='Pressure', line=dict(color='#4ade80')))
    fig_p.add_hrect(y0=0, y1=20, fillcolor="red", opacity=0.2, line_width=0)
    fig_p.add_hrect(y0=80, y1=150, fillcolor="red", opacity=0.2, line_width=0)
    fig_p.update_layout(**p_layout(), title="Oil Pressure (PSI)")
    c3.plotly_chart(fig_p, use_container_width=True)
    
    if st.session_state.fft_enabled:
        sp = np.fft.fft(vib_arr)
        freq = np.fft.fftfreq(len(sp), d=0.01) # Assume 100Hz
        pos_f, pos_amp = freq[freq>0], np.abs(sp)[freq>0]
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Bar(x=pos_f, y=pos_amp, marker=dict(color=pos_amp, colorscale='Viridis')))
        l = p_layout()
        l["title"] = "Vibration Frequency Spectrum — FFT Analysis"
        l["xaxis_title"] = "Frequency (Hz)"
        l["yaxis_title"] = "Amplitude"
        fig_fft.update_layout(**l)
        st.plotly_chart(fig_fft, use_container_width=True)

with tabs[1]: # Health & RUL
    if active_data.get("diagnosis"):
        diag = active_data["diagnosis"]
        rul_h = float(diag.get("estimated_rul_hours", 720))
        health_perc = min(100, max(0, (rul_h / 720) * 100))
        
        c1, c2 = st.columns([1, 2])
        fig_g = go.Figure(go.Indicator(
            mode = "gauge+number", value = health_perc, title = {'text': "Machine Health Index"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "white"},
                'steps': [
                    {'range': [0, 30], 'color': "#ef4444"},
                    {'range': [30, 60], 'color': "#f97316"},
                    {'range': [60, 80], 'color': "#fbbf24"},
                    {'range': [80, 100], 'color': "#4ade80"}
                ]
            }
        ))
        l = p_layout()
        l["height"] = 350
        fig_g.update_layout(**l)
        c1.plotly_chart(fig_g, use_container_width=True)
        
        with c2:
            st.markdown("### Remaining Useful Life Forecast")
            met1, met2, met3, met4 = st.columns(4)
            met1.metric("RUL (Hours)", f"{rul_h:.1f}")
            met2.metric("RUL (Days)", f"{rul_h/24:.1f}")
            met3.metric("7-Day Fail Prob", f"{diag.get('tool_outputs',{}).get('calculate_rul',{}).get('failure_probability_7d', 0):.1f}%")
            met4.metric("30-Day Fail Prob", f"{diag.get('tool_outputs',{}).get('calculate_rul',{}).get('failure_probability_30d', 0):.1f}%")
            
            # Progress bar timeline
            st.markdown("#### Degradation Timeline")
            st.progress(max(0, min(100, int(100 - health_perc))))
    else:
        st.info("Run diagnostics to process Health & RUL.")

with tabs[2]: # Anomaly Analysis
    c1, c2 = st.columns(2)
    # Heatmap
    z_t = np.abs((temp_arr - np.mean(temp_arr)) / (np.std(temp_arr) or 1))
    z_v = np.abs((vib_arr - np.mean(vib_arr)) / (np.std(vib_arr) or 1))
    z_p = np.abs((press_arr - np.mean(press_arr)) / (np.std(press_arr) or 1))
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=[z_t, z_v, z_p], y=['Temp', 'Vib', 'Press'], x=list(range(50)),
        colorscale='RdYlGn_r'
    ))
    l = p_layout()
    l["title"] = "Multi-Signal Anomaly Heatmap"
    fig_heat.update_layout(**l)
    c1.plotly_chart(fig_heat, use_container_width=True)
    
    # Correlation Math
    df_corr = pd.DataFrame({"Temp": temp_arr, "Vib": vib_arr, "Press": press_arr}).corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=df_corr.values, x=df_corr.columns, y=df_corr.columns,
        colorscale='RdBu_r', zmin=-1, zmax=1, text=np.round(df_corr.values, 2), texttemplate="%{text}"
    ))
    l2 = p_layout()
    l2["title"] = "Sensor Cross-Correlation Matrix"
    fig_corr.update_layout(**l2)
    c2.plotly_chart(fig_corr, use_container_width=True)

with tabs[3]: # AI Diagnostics
    if st.session_state.trace:
        st.markdown("#### Thinking Trace")
        st.markdown(f"<div class='trace-box'>{'<br>'.join(st.session_state.trace)}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
    if active_data.get("diagnosis"):
        diag = active_data["diagnosis"]
        st.markdown(f"### Diagnostics for {machine}")
        
        # Row 1
        rc1, rc2, rc3, rc4 = st.columns(4)
        stat = diag.get("status", "UNKNOWN")
        bdg_cl = "red" if stat in ["CRITICAL", "EMERGENCY"] else ("orange" if stat == "WARNING" else "green")
        rc1.markdown(f"**Status**<br><span style='color:{bdg_cl};font-size:1.2rem;font-weight:bold'>● {stat}</span>", unsafe_allow_html=True)
        rc2.markdown(f"**Fault Type**<br>`{diag.get('fault_type', 'None')}`", unsafe_allow_html=True)
        rc3.markdown(f"**Urgency**<br>`{diag.get('urgency', 'None')}`", unsafe_allow_html=True)
        rc4.markdown(f"**Safety Risk**<br>`{diag.get('safety_risk', 'None')}`", unsafe_allow_html=True)
        
        # Row 2
        st.markdown("<hr style='border-color:#1e2d40; margin:15px 0'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 2])
        try: conf_val = float(diag.get("confidence", 0))
        except: conf_val = 0.0
        c1.metric("Confidence", f"{conf_val * 100:.1f}%")
        c2.metric("Severity Score", diag.get("severity_score", 0))
        c3.markdown("**Affected Components**")
        c3.markdown(" · " + " · ".join(diag.get("affected_components", [])))
        
        # Row 3
        st.markdown("<br>", unsafe_allow_html=True)
        bx1, bx2 = st.columns(2)
        bx1.markdown(f"<div class='card'><h4>🔍 Root Cause</h4>{diag.get('root_cause', '')}</div>", unsafe_allow_html=True)
        bx2.markdown(f"<div class='card' style='border:1px solid {bdg_cl}'><h4>🛠 Recommended Action</h4>{diag.get('recommended_action', '')}</div>", unsafe_allow_html=True)
        
        with st.expander("Raw JSON Report"):
            st.json(diag)
        with st.expander("Tool Outputs"):
            st.json(diag.get("tool_outputs", {}))
    else:
        st.info("No AI diagnostics run yet. Click 'Run Full AI Diagnostics'.")

with tabs[4]: # History & Export
    st.markdown("### Diagnosis History")
    if st.session_state.diagnosis_history:
        df_hist = pd.DataFrame(st.session_state.diagnosis_history)
        def color_status(val):
            color = "#ef4444" if val in ["CRITICAL","EMERGENCY"] else ("#fbbf24" if val=="WARNING" else "#4ade80")
            return f'color: {color}; font-weight: bold'
        st.dataframe(df_hist.style.applymap(color_status, subset=['Status']), use_container_width=True)
        
        c1, c2 = st.columns(2)
        c1.download_button("📥 Export CSV", df_hist.to_csv(index=False), "history.csv", "text/csv")
        if active_data.get("diagnosis"):
            c2.download_button("📄 Export JSON", json.dumps(active_data["diagnosis"], indent=2), "report.json", "application/json")
    else:
        st.write("No history available.")

with tabs[5]: # Settings
    st.markdown("### System Settings")
    st.session_state.api_key = st.text_input("NVIDIA API Key", value=st.session_state.api_key, type="password")
    
    st.session_state.llm_model = st.selectbox("LLM Model", ["moonshotai/kimi-k2.5", "meta/llama-3.1-70b-instruct"], index=0)
    st.session_state.llm_temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2, 0.1)
    st.session_state.confidence_threshold = st.slider("Confidence Correction Threshold", 0.5, 0.95, 0.75, 0.05)
    
    st.session_state.fft_enabled = st.toggle("Enable FFT Analysis", st.session_state.fft_enabled)
    st.session_state.bearing_detection_enabled = st.toggle("Enable Bearing Fault Detection", st.session_state.bearing_detection_enabled)
    
    if st.button("Reset Session State"):
        st.session_state.clear()
        st.rerun()
