import streamlit as st
import pandas as pd

# ---------- Doctor Database ----------
DOCTORS = {
    "General Medicine": {
        "name": "Dr. Emily White",
        "specialty": "Internal Medicine",
        "experience": 12,
        "phone": "+1 555-1234",
        "email": "e.white@hospital.com"
    },
    "Cardiology": {
        "name": "Dr. James Carter",
        "specialty": "Interventional Cardiology",
        "experience": 18,
        "phone": "+1 555-2345",
        "email": "j.carter@hospital.com"
    },
    "Neurology": {
        "name": "Dr. Sarah Lee",
        "specialty": "Neurology",
        "experience": 15,
        "phone": "+1 555-3456",
        "email": "s.lee@hospital.com"
    },
    "Infectious Disease": {
        "name": "Dr. Michael Brown",
        "specialty": "Infectious Diseases",
        "experience": 10,
        "phone": "+1 555-4567",
        "email": "m.brown@hospital.com"
    },
    "Pulmonology": {
        "name": "Dr. Lisa Green",
        "specialty": "Pulmonology",
        "experience": 14,
        "phone": "+1 555-5678",
        "email": "l.green@hospital.com"
    },
    "Emergency": {
        "name": "Dr. Robert Adams",
        "specialty": "Emergency Medicine",
        "experience": 20,
        "phone": "+1 555-6789",
        "email": "r.adams@hospital.com"
    }
}

# ---------- Dummy AI Function (replace with real backend call) ----------
def dummy_triage(age, gender, symptoms, bp, hr, temp, pre_conditions):
    risk = "Low"
    factors = {}
    if hr > 100 or temp > 38.5:
        risk = "High"
        if hr > 100: factors["Heart Rate"] = hr
        if temp > 38.5: factors["Temperature"] = temp
    elif hr > 80 or temp > 37.5:
        risk = "Medium"
        if hr > 80: factors["Heart Rate"] = hr
        if temp > 37.5: factors["Temperature"] = temp

    dept = "General Medicine"
    sym = symptoms.lower()
    if "chest pain" in sym:
        dept = "Cardiology"
    elif "headache" in sym or "dizziness" in sym:
        dept = "Neurology"
    elif "fever" in sym and "cough" in sym:
        dept = "Infectious Disease"
    elif "shortness of breath" in sym:
        dept = "Pulmonology"

    confidence = 0.85
    return risk, dept, factors, confidence

# ---------- Initialize Session State ----------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "next_id" not in st.session_state:
    st.session_state.next_id = 1
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ---------- Sidebar Navigation ----------
with st.sidebar:
    st.title("Navigation")
    
    if st.button("🏠 Home"):
        st.session_state.page = "home"
        st.rerun()
    
    if st.button("🔐 Login"):
        st.session_state.page = "login"
        st.rerun()
    
    # Only show these if authenticated
    if st.session_state.authenticated:
        if st.button("📝 Patient Input"):
            st.session_state.page = "input"
            st.rerun()
        if st.button("📋 Results"):
            st.session_state.page = "results"
            st.rerun()
    else:
        st.button("📝 Patient Input", disabled=True)
        st.button("📋 Results", disabled=True)
    
    st.divider()
    
    if st.session_state.authenticated:
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.page = "home"
            st.session_state.last_result = None  # optional: clear last result
            st.rerun()

# ---------- Page Functions ----------
def home_page():
    st.title("🏥 AI‑Powered Smart Patient Triage")
    st.markdown("""
    Welcome to the AI Triage System.  
    This tool helps classify patient risk levels and recommend the appropriate medical department.
    
    **How it works:**
    1. Login (simple demo)
    2. Enter patient symptoms and vitals
    3. Get instant risk assessment and doctor recommendation
    4. View past triage history
    """)
    if st.button("Get Started"):
        st.session_state.page = "login"
        st.rerun()

def login_page():
    st.title("🔐 Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username and password:  # dummy check
                st.session_state.authenticated = True
                st.session_state.page = "input"
                st.rerun()
            else:
                st.error("Please enter both username and password")

def input_page():
    if not st.session_state.authenticated:
        st.warning("Please login first.")
        st.session_state.page = "login"
        st.rerun()
        return

    st.title("📝 Patient Intake Form")
    
    with st.form("patient_form"):
        st.caption("New patients will be assigned an ID automatically.")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 120, 30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            symptoms = st.text_area("Symptoms (comma separated)", "Fever, cough")
        with col2:
            bp = st.text_input("Blood Pressure", "120/80")
            hr = st.number_input("Heart Rate", 30, 200, 75)
            temp = st.number_input("Temperature (°C)", 35.0, 42.0, 37.0, 0.1)
        pre_conditions = st.multiselect("Pre‑Existing Conditions",
                                         ["Hypertension", "Diabetes", "Asthma", "Heart disease", "None"])
        submitted = st.form_submit_button("Triage")

    if submitted:
        # Generate new ID
        patient_id = f"P{st.session_state.next_id:04d}"
        st.session_state.next_id += 1

        # Call triage function
        risk, dept, factors, conf = dummy_triage(age, gender, symptoms, bp, hr, temp, pre_conditions)

        # Build result dictionary
        result = {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "symptoms": symptoms,
            "bp": bp,
            "hr": hr,
            "temp": temp,
            "pre_conditions": ", ".join(pre_conditions) if pre_conditions else "None",
            "risk": risk,
            "department": dept,
            "confidence": conf,
            "factors": factors
        }

        # Store in session
        st.session_state.last_result = result
        st.session_state.history.append(result)

        # Go to results page
        st.session_state.page = "results"
        st.rerun()

def results_page():
    if not st.session_state.authenticated:
        st.warning("Please login first.")
        st.session_state.page = "login"
        st.rerun()
        return

    st.title("📋 Triage Result")
    
    if st.session_state.last_result is None:
        st.info("No triage performed yet. Please go to the input page.")
        if st.button("Go to Input"):
            st.session_state.page = "input"
            st.rerun()
        return

    res = st.session_state.last_result
    dept = res["department"]
    doctor = DOCTORS.get(dept, DOCTORS["General Medicine"])

    # Main result
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Patient ID", res["patient_id"])
    with col2:
        st.metric("Risk Level", res["risk"])
    with col3:
        st.metric("Recommended Dept", dept)

    st.markdown(f"**Confidence:** {res['confidence']:.0%}")

    # Explainability
    with st.expander("🔍 Why this recommendation?"):
        if res["factors"]:
            st.write("**Contributing factors:**")
            for k, v in res["factors"].items():
                st.write(f"- {k}: {v}")
        else:
            st.write("No abnormal vital signs detected.")
        st.write(f"**Model confidence:** {res['confidence']:.0%}")

    # Doctor info
    st.subheader("👨‍⚕️ Recommended Doctor")
    st.write(f"**Name:** {doctor['name']}")
    st.write(f"**Specialty:** {doctor['specialty']}")
    st.write(f"**Experience:** {doctor['experience']} years")
    st.write(f"**Phone:** {doctor['phone']}")
    st.write(f"**Email:** {doctor['email']}")

    # Action for patient
    if res["risk"] == "High":
        st.warning("🚨 **Immediate action required:** Please contact the doctor or proceed to the Emergency Department.")
    elif res["risk"] == "Medium":
        st.info("⏳ **Schedule an appointment** with the recommended doctor within 24 hours.")
    else:
        st.success("✅ **Low risk.** Monitor symptoms and consult your primary care physician if needed.")

    # History log (sorted by risk: High → Medium → Low)
    st.markdown("---")
    st.subheader("📜 Triage History")
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        # Define risk order
        risk_order = ["High", "Medium", "Low"]
        df_history["risk"] = pd.Categorical(df_history["risk"], categories=risk_order, ordered=True)
        df_history = df_history.sort_values("risk")
        display_cols = ["patient_id", "age", "gender", "risk", "department"]
        st.dataframe(df_history[display_cols], use_container_width=True)
    else:
        st.write("No history yet.")

    # Optional button to start new triage
    if st.button("➕ New Triage"):
        st.session_state.page = "input"
        st.rerun()

# ---------- Router ----------
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "login":
    login_page()
elif st.session_state.page == "input":
    input_page()
elif st.session_state.page == "results":
    results_page()