import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------- 1. ASSET LOADING (MEMORY OPTIMIZED) ----------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('patient_risk_model.pkl', mmap_mode='r')
        encoders = joblib.load('feature_encoders.pkl')
        target_le = joblib.load('target_encoder.pkl')
        all_features = joblib.load('feature_names.pkl')
        
        # Clean feature list for UI - Added 'Patient ID' to remove list
        to_remove = [
            'Respiratory Rate', 'Oxygen Saturation', 'Derived_Pulse_Pressure', 
            'Derived_HRV', 'Derived_BMI', 'Derived_MAP',
            'Systolic Blood Pressure', 'Diastolic Blood Pressure',
            'Patient ID'
        ]
        display_features = [f for f in all_features if f not in to_remove]
        
        return model, encoders, target_le, all_features, display_features
    except Exception as e:
        st.error(f"Critical Error: Ensure all .pkl files are in the directory. {e}")
        st.stop()

model, encoders, target_le, model_features, display_features = load_assets()

# ---------- 2. DOCTOR DATABASE ----------
DOCTORS = {
    "General Medicine": {"name": "Dr. Emily White", "specialty": "Internal Medicine", "phone": "+1 555-1234", "email": "e.white@hospital.com"},
    "Cardiology": {"name": "Dr. James Carter", "specialty": "Cardiology", "phone": "+1 555-2345", "email": "j.carter@hospital.com"},
    "Neurology": {"name": "Dr. Sarah Lee", "specialty": "Neurology", "phone": "+1 555-3456", "email": "s.lee@hospital.com"},
    "Infectious Disease": {"name": "Dr. Michael Brown", "specialty": "Infectious Diseases", "phone": "+1 555-4567", "email": "m.brown@hospital.com"},
    "Pulmonology": {"name": "Dr. Lisa Green", "specialty": "Pulmonology", "phone": "+1 555-5678", "email": "l.green@hospital.com"},
    "Emergency": {"name": "Dr. Robert Adams", "specialty": "Emergency Medicine", "phone": "+1 555-6789", "email": "r.adams@hospital.com"}
}

# ---------- 3. SESSION STATE ----------
if "page" not in st.session_state: st.session_state.page = "home"
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "username" not in st.session_state: st.session_state.username = ""
if "history" not in st.session_state: st.session_state.history = []
if "last_result" not in st.session_state: st.session_state.last_result = None
if "next_id" not in st.session_state: st.session_state.next_id = 1
if "show_history" not in st.session_state: st.session_state.show_history = False
if "risk_sort_order" not in st.session_state: st.session_state.risk_sort_order = "descending"  # or "ascending"

def nav(page_name):
    st.session_state.page = page_name
    st.session_state.show_history = False
    st.rerun()

def toggle_history():
    st.session_state.show_history = not st.session_state.show_history
    st.rerun()

def toggle_risk_sort():
    if st.session_state.risk_sort_order == "descending":
        st.session_state.risk_sort_order = "ascending"
    else:
        st.session_state.risk_sort_order = "descending"
    st.rerun()

# Function to get risk score for sorting (convert risk level to numeric)
def get_risk_score(risk_level):
    risk_scores = {
        "High": 3,
        "Medium": 2,
        "Low": 1
    }
    return risk_scores.get(risk_level, 0)

# Function to get sorted history by risk
def get_sorted_history_by_risk(history_list, order="descending"):
    if not history_list:
        return []
    
    # Create a list of tuples (risk_score, original_index, record)
    indexed_records = [(get_risk_score(record.get('risk', 'Low')), i, record) 
                       for i, record in enumerate(history_list)]
    
    # Sort by risk score
    reverse = (order == "descending")
    indexed_records.sort(key=lambda x: x[0], reverse=reverse)
    
    # Return just the records in sorted order
    return [record for _, _, record in indexed_records]

# Function to get risk level counts
def get_risk_counts(history_list):
    if not history_list:
        return {"High": 0, "Medium": 0, "Low": 0}
    
    counts = {"High": 0, "Medium": 0, "Low": 0}
    for record in history_list:
        risk = record.get('risk', 'Low')
        if risk in counts:
            counts[risk] += 1
    return counts

# ---------- 4. PERMANENT NAVIGATION BAR (SIDEBAR) - ALWAYS VISIBLE ----------
def render_permanent_navbar():
    with st.sidebar:
        st.title("🏥 Navigation")
        st.markdown("### Main Menu")
        
        # Home button - always visible
        if st.button("🏠 Home", use_container_width=True):
            nav("home")
        
        # These buttons are always visible but disabled/different when not authenticated
        if not st.session_state.authenticated:
            st.button("📝 New Intake", use_container_width=True, disabled=True)
            st.button("📊 Current Results", use_container_width=True, disabled=True)
            st.caption("🔒 Login to access these features")
        else:
            if st.button("📝 New Intake", use_container_width=True):
                nav("input")
            if st.button("📊 Current Results", use_container_width=True):
                nav("results")
        
        st.divider()
        
        # History section with risk counts - always visible
        st.markdown("### 📜 History")
        
        # Risk level counts summary in sidebar
        if st.session_state.authenticated and st.session_state.history:
            risk_counts = get_risk_counts(st.session_state.history)
            
            # Create a compact risk summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"🔴 {risk_counts['High']}")
            with col2:
                st.markdown(f"🟡 {risk_counts['Medium']}")
            with col3:
                st.markdown(f"🟢 {risk_counts['Low']}")
        
        # Total count
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Total Records**")
        with col2:
            history_count = len(st.session_state.history)
            st.markdown(f"**{history_count}**")
        
        # History toggle button - always visible but disabled when not authenticated
        if not st.session_state.authenticated:
            st.button("👁️ Toggle History Panel", use_container_width=True, disabled=True)
            st.caption("🔒 Login to view history")
        else:
            if st.button("👁️ Toggle History Panel", use_container_width=True):
                toggle_history()
        
        st.divider()
        
        # User info section - changes based on auth state
        if st.session_state.authenticated:
            st.markdown(f"**Logged in as:** {st.session_state.username}")
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.username = ""
                st.session_state.page = "home"
                st.session_state.show_history = False
                st.rerun()
        else:
            st.markdown("**Not logged in**")
            if st.button("🔐 Login", use_container_width=True):
                nav("login")

# ---------- 5. HISTORY PANEL (COLLAPSIBLE) ----------
def render_history_panel():
    if st.session_state.show_history and st.session_state.authenticated:
        with st.sidebar:
            st.divider()
            
            # History header with sort control and risk counts
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 📜 Recent Triages")
            with col2:
                sort_icon = "🔽" if st.session_state.risk_sort_order == "descending" else "🔼"
                if st.button(f"{sort_icon} Risk", key="sort_risk", use_container_width=True):
                    toggle_risk_sort()
            
            # Show risk counts in history panel
            if st.session_state.history:
                risk_counts = get_risk_counts(st.session_state.history)
                
                # Create a visual risk summary with progress bars
                st.markdown("**Risk Breakdown:**")
                col1, col2, col3 = st.columns(3)
                
                total = len(st.session_state.history)
                with col1:
                    high_pct = risk_counts['High']/total if total > 0 else 0
                    st.markdown(f"🔴 **High:** {risk_counts['High']}")
                    st.progress(high_pct, text=f"{high_pct:.0%}")
                
                with col2:
                    medium_pct = risk_counts['Medium']/total if total > 0 else 0
                    st.markdown(f"🟡 **Medium:** {risk_counts['Medium']}")
                    st.progress(medium_pct, text=f"{medium_pct:.0%}")
                
                with col3:
                    low_pct = risk_counts['Low']/total if total > 0 else 0
                    st.markdown(f"🟢 **Low:** {risk_counts['Low']}")
                    st.progress(low_pct, text=f"{low_pct:.0%}")
                
                st.divider()
                
                # Get sorted history by risk
                sorted_history = get_sorted_history_by_risk(
                    st.session_state.history, 
                    st.session_state.risk_sort_order
                )
                
                # Show last 5 from sorted list
                for i, record in enumerate(sorted_history[-5:]):
                    with st.container():
                        cols = st.columns([2, 2, 1])
                        cols[0].markdown(f"**{record['patient_id']}**")
                        
                        # Color code risk level
                        risk = record['risk']
                        confidence = record.get('confidence', 0)
                        
                        if risk == "High":
                            cols[1].markdown(f"🔴 *{risk}*")
                        elif risk == "Medium":
                            cols[1].markdown(f"🟡 *{risk}*")
                        else:
                            cols[1].markdown(f"🟢 *{risk}*")
                            
                        if cols[2].button("View", key=f"view_{i}", use_container_width=True):
                            st.session_state.last_result = record
                            nav("results")
                        
                        # Show confidence as mini progress bar
                        st.progress(confidence, text=f"Confidence: {confidence:.0%}")
                    st.markdown("---")
                
                if len(sorted_history) > 5:
                    st.caption(f"Showing last 5 of {len(sorted_history)} records (sorted by risk {st.session_state.risk_sort_order})")
                    
                if st.button("📋 View Full History", use_container_width=True):
                    st.session_state.page = "full_history"
                    st.session_state.show_history = False
                    st.rerun()
            else:
                st.info("No history yet")

# ---------- 6. PAGE FUNCTIONS ----------

def home_page():
    st.title("🏥 AI‑Powered Smart Patient Triage")
    st.markdown("### Clinical Decision Support System")
    
    if not st.session_state.authenticated:
        st.info("👋 Welcome to the Triage System")
        st.markdown("""
        ### Features:
        - 🔐 Secure login for medical staff
        - 📝 Patient intake and assessment
        - 🤖 AI-powered risk prediction
        - 📊 Historical record keeping
        - 👨‍⚕️ Automated doctor assignment
        
        Please login to access the full features.
        """)
        
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("🔐 Login", type="primary", use_container_width=True):
                nav("login")
    else:
        st.success(f"Welcome back, {st.session_state.username}!")
        
        # Quick stats with risk counts
        if st.session_state.history:
            risk_counts = get_risk_counts(st.session_state.history)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Patients", len(st.session_state.history))
            with col2:
                st.metric("🔴 High Risk", risk_counts['High'])
            with col3:
                st.metric("🟡 Medium Risk", risk_counts['Medium'])
            with col4:
                st.metric("🟢 Low Risk", risk_counts['Low'])
            
            # Risk distribution chart
            st.subheader("📊 Risk Distribution")
            risk_data = pd.DataFrame({
                'Risk Level': ['High', 'Medium', 'Low'],
                'Count': [risk_counts['High'], risk_counts['Medium'], risk_counts['Low']]
            })
            st.bar_chart(risk_data.set_index('Risk Level'))
        else:
            st.info("No triage records yet. Start by creating a new intake.")
        
        if st.button("Start New Triage", type="primary", use_container_width=True):
            nav("input")

def login_page():
    st.title("🔐 Medical Staff Login")
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.markdown("### Enter Credentials")
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Login", type="primary", use_container_width=True):
                    if u.strip() != "" and p.strip() != "":
                        st.session_state.authenticated = True
                        st.session_state.username = u
                        nav("input")
                    else:
                        st.error("Please enter credentials.")
            with col2:
                if st.form_submit_button("Cancel", use_container_width=True):
                    nav("home")

def input_page():
    if not st.session_state.authenticated: 
        st.error("Please login first")
        nav("login")
        return
        
    st.title("📝 Patient Intake Form")

    # ---- Generate next patient ID (e.g., P001) ----
    next_id_str = f"P{st.session_state.next_id:03d}"
    st.info(f"**Patient ID:** {next_id_str} *(auto‑assigned)*")

    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        ui_input_data = {}

        # Blood pressure input
        with col1:
            bp_string = st.text_input("Blood Pressure (e.g., 120/80)", value="120/80")

        # Dynamically add all display features
        for i, col in enumerate(display_features):
            target_col = col1 if i % 2 == 1 else col2
            if col in encoders:
                ui_input_data[col] = target_col.selectbox(f"{col}", encoders[col].classes_)
            elif any(x in col for x in ["Age", "Heart Rate", "Temperature"]):
                ui_input_data[col] = target_col.number_input(f"{col}", value=0, step=1, format="%d")
            else:
                ui_input_data[col] = target_col.number_input(f"{col}", value=0.0, format="%.2f")

        st.divider()
        pre_conditions = st.multiselect(
            "Pre‑Existing Conditions",
            ["Hypertension", "Diabetes", "Asthma", "Heart Disease", "None"]
        )
        pre_conditions_str = ", ".join(pre_conditions) if pre_conditions and "None" not in pre_conditions else "None"

        symptoms = st.text_area("Observations / Symptoms")

        submitted = st.form_submit_button("Run Triage Analysis", type="primary", use_container_width=True)

        if submitted:
            try:
                sbp, dbp = map(float, bp_string.split('/'))

                model_input = ui_input_data.copy()
                model_input['Systolic Blood Pressure'] = sbp
                model_input['Diastolic Blood Pressure'] = dbp

                for f in model_features:
                    if f not in model_input:
                        model_input[f] = 0.0

                df = pd.DataFrame([model_input])
                for col, le in encoders.items():
                    if col in df.columns:
                        df[col] = le.transform(df[col])

                df = df[model_features]

                pred_idx = model.predict(df)[0]
                risk = target_le.inverse_transform([pred_idx])[0]

                dept = "General Medicine"
                if "chest" in symptoms.lower():
                    dept = "Cardiology"
                elif "headache" in symptoms.lower() or "dizzy" in symptoms.lower():
                    dept = "Neurology"
                elif "fever" in symptoms.lower() and "cough" in symptoms.lower():
                    dept = "Infectious Disease"
                elif "breath" in symptoms.lower():
                    dept = "Pulmonology"
                if risk == "High" or "High" in str(risk):
                    dept = "Emergency"

                conf = np.max(model.predict_proba(df)[0]) if hasattr(model, "predict_proba") else 0.85

                result = {
                    "patient_id": next_id_str,
                    "risk": risk,
                    "department": dept,
                    "confidence": conf,
                    "BP": bp_string,
                    "pre_conditions": pre_conditions_str,
                    "symptoms": symptoms,
                    **ui_input_data
                }

                st.session_state.next_id += 1
                st.session_state.last_result = result
                st.session_state.history.append(result)

                nav("results")

            except ValueError:
                st.error("Invalid BP format. Please use e.g., 120/80.")
            except Exception as e:
                st.error(f"Prediction error: {e}")

def results_page():
    if not st.session_state.authenticated: 
        nav("login")
        return
        
    st.title("📋 Triage Assessment")

    res = st.session_state.last_result
    if not res:
        st.info("No triage performed yet.")
        if st.button("New Intake"): nav("input")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Patient ID", res["patient_id"])
    col2.metric("Risk Level", res["risk"])
    col3.metric("BP Recorded", res["BP"])

    if "pre_conditions" in res:
        st.write(f"**Pre‑existing conditions:** {res['pre_conditions']}")

    st.success(f"**Recommended department:** {res['department']}")
    st.progress(res["confidence"])
    st.caption(f"Confidence: {res['confidence']:.0%}")

    if res["department"] in DOCTORS:
        doctor = DOCTORS[res["department"]]
        with st.expander("👨‍⚕️ Assigned Doctor Information"):
            st.write(f"**Name:** {doctor['name']}")
            st.write(f"**Specialty:** {doctor['specialty']}")
            st.write(f"**Phone:** {doctor['phone']}")
            st.write(f"**Email:** {doctor['email']}")

    if res.get("symptoms"):
        with st.expander("📝 Symptoms Summary"):
            st.write(res["symptoms"])

    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New Triage", use_container_width=True):
            nav("input")
    with col2:
        if st.button("📋 View History", use_container_width=True):
            st.session_state.show_history = True
            st.rerun()

def full_history_page():
    if not st.session_state.authenticated: 
        nav("login")
        return
        
    st.title("📜 Complete Triage History")
    
    # Sort control for full history
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### Sorted by Risk ({st.session_state.risk_sort_order})")
    with col2:
        sort_label = "🔽 High to Low" if st.session_state.risk_sort_order == "descending" else "🔼 Low to High"
        if st.button(sort_label, use_container_width=True):
            toggle_risk_sort()
    
    if st.session_state.history:
        # Get risk counts
        risk_counts = get_risk_counts(st.session_state.history)
        total = len(st.session_state.history)
        
        # Display risk level summary with counts and percentages
        st.subheader("📊 Risk Level Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", total)
        with col2:
            high_pct = (risk_counts['High']/total*100) if total > 0 else 0
            st.metric("🔴 High Risk", f"{risk_counts['High']} ({high_pct:.1f}%)")
        with col3:
            medium_pct = (risk_counts['Medium']/total*100) if total > 0 else 0
            st.metric("🟡 Medium Risk", f"{risk_counts['Medium']} ({medium_pct:.1f}%)")
        with col4:
            low_pct = (risk_counts['Low']/total*100) if total > 0 else 0
            st.metric("🟢 Low Risk", f"{risk_counts['Low']} ({low_pct:.1f}%)")
        
        # Visual progress bars for risk distribution
        st.subheader("📈 Risk Distribution")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**High Risk**")
            st.progress(risk_counts['High']/total if total > 0 else 0, 
                       text=f"{risk_counts['High']} patients")
        with col2:
            st.markdown("**Medium Risk**")
            st.progress(risk_counts['Medium']/total if total > 0 else 0,
                       text=f"{risk_counts['Medium']} patients")
        with col3:
            st.markdown("**Low Risk**")
            st.progress(risk_counts['Low']/total if total > 0 else 0,
                       text=f"{risk_counts['Low']} patients")
        
        st.divider()
        
        # Get sorted history by risk
        sorted_history = get_sorted_history_by_risk(
            st.session_state.history, 
            st.session_state.risk_sort_order
        )
        
        # Create DataFrame from sorted history
        history_df = pd.DataFrame(sorted_history)
        
        if 'patient_id' in history_df.columns:
            history_df['patient_id'] = history_df['patient_id'].astype(str)
        
        # Add risk score column for display
        history_df['risk_score'] = history_df['risk'].apply(get_risk_score)
        
        display_cols = ["patient_id", "risk", "confidence", "department", "BP", "pre_conditions", "symptoms"]
        available_cols = [c for c in display_cols if c in history_df.columns]
        
        # Format confidence as percentage
        if 'confidence' in history_df.columns:
            history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            history_df[available_cols], 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "patient_id": "Patient ID",
                "risk": "Risk Level",
                "confidence": "Confidence",
                "department": "Department",
                "BP": "Blood Pressure",
                "pre_conditions": "Pre-existing Conditions",
                "symptoms": "Symptoms"
            }
        )
        
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name="triage_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No triage history available")
    
    if st.button("← Back", use_container_width=True):
        nav("home")

# ---------- 7. MAIN APP LAYOUT ----------
# Render permanent navigation bar (always visible)
render_permanent_navbar()

# Render collapsible history panel (only when toggled and authenticated)
render_history_panel()

# Main content area
if st.session_state.page == "home": 
    home_page()
elif st.session_state.page == "login": 
    login_page()
elif st.session_state.page == "input": 
    input_page()
elif st.session_state.page == "results": 
    results_page()
elif st.session_state.page == "full_history": 
    full_history_page()