import streamlit as st
from pathlib import Path
from PIL import Image

# =========================================================

# CONFIG

# =========================================================

BASE_DIR = Path(**file**).parent
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "surfacexlab_logo.png"

# =========================================================

# LOGO

# =========================================================

def load_logo():
if LOGO_PATH.exists():
try:
return Image.open(LOGO_PATH)
except:
return None
return None

logo_image = load_logo()

# =========================================================

# PAGE

# =========================================================

st.set_page_config(
page_title="SurfaceXLab",
page_icon=logo_image if logo_image else "🧪",
layout="wide"
)

st.title("SurfaceXLab funcionando 🚀")
st.write("Se você está vendo isso, a indentação está correta ✅")
