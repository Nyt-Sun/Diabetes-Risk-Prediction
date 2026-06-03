st.markdown("""
<style>

/* MAIN APP BACKGROUND (BEIGE) */
.stApp {
    background: #f5f0e6;
}

/* HEADER (KEEP DARK FOR CONTRAST) */
.header {
    background-color: #0b1f3a;
    padding: 22px;
    border-radius: 18px;
    text-align: center;
}

/* TITLE */
.title {
    color: white;
    font-size: 46px;
    font-weight: 800;
}

.subtitle {
    color: #d6d6d6;
    font-size: 14px;
}

/* SIDEBAR (DARKER BEIGE) */
section[data-testid="stSidebar"] {
    background-color: #e3d7c3 !important;
}

/* SIDEBAR TEXT TONE */
section[data-testid="stSidebar"] * {
    color: #1f1f1f;
}

/* MAIN CONTENT AREA (LIGHTER REPORT ZONE FEEL) */
.block-container {
    background-color: #f8f3ea;
    padding: 2rem;
    border-radius: 10px;
}

/* BUTTON STYLE (CLINICAL BLUE ACCENT) */
.stButton>button {
    background-color: #0b1f3a;
    color: white;
    border-radius: 8px;
}

.stButton>button:hover {
    background-color: #163a63;
}

</style>
""", unsafe_allow_html=True)