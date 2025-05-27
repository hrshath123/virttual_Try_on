import streamlit as st
import pyrebase
import base64
import os
import subprocess
import sys
import time
from io import BytesIO

# ================= Determine base directory =================
# Ensures we always read/write from the repo root (where this file lives).
base_dir = os.path.dirname(os.path.abspath(__file__))

# ================= Firebase Configuration =================
firebase_config = {
    "apiKey": "AIzaSyAm6HtleJzgdBAMM7k0VGaFwfQbe_GNbWY",
    "authDomain": "clothes-wala.firebaseapp.com",
    "databaseURL": "https://clothes-wala-default-rtdb.firebaseio.com",
    "projectId": "clothes-wala",
    "storageBucket": "clothes-wala.appspot.com",
    "messagingSenderId": "48604679015",
    "appId": "1:48604679015:web:f0065e957c1b50eb563dc2",
    "measurementId": "G-HMDVX52K6Q"
}

# ✅ Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
db = firebase.database()

def run_virtual_tryon(cloth_path: str) -> bool:
    """
    Runs `automated.py` (located next to this file) using the same Python interpreter.
    Captures stdout/stderr and displays it in the Streamlit UI. Returns True on success.
    """
    try:
        script_path = os.path.join(base_dir, "automated.py")
        proc = subprocess.run(
            [sys.executable, script_path, cloth_path],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if proc.stdout:
            st.text("=== automated.py STDOUT ===")
            st.code(proc.stdout)
        if proc.stderr:
            st.text("=== automated.py STDERR ===")
            st.code(proc.stderr)
        if proc.returncode != 0:
            st.error(f"❌ automated.py failed (exit code {proc.returncode}).")
            return False
        return True
    except Exception as e:
        st.error(f"⚠️ Error running virtual try-on: {e}")
        return False

def encode_image(image_path: str) -> str | None:
    """Encodes an image to Base64 so we can store it in Firebase."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"⚠️ Error encoding image: {e}")
        return None

def upload_item(
    shop_no: str,
    item_name: str,
    item_price: float,
    item_desc: str,
    uploaded_image: BytesIO,
):
    """
    1) Save uploaded cloth under `datasets/test/cloth/`
    2) Push original_image + metadata to Firebase
    3) Run automated.py → results/_overlayed_{i}.jpg ...
    4) Base64-encode each overlayed_{i}.jpg and push to Firebase
    """
    if uploaded_image is None:
        st.error("⚠️ Please upload an image first.")
        return

    try:
        # 1) Unique item ID
        item_id = f"item_{int(time.time())}"
        item_path = f"Shops/{shop_no}/items/{item_id}/"

        # 2) Convert uploaded image → Base64 string (for Firebase)
        image_bytes = uploaded_image.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        db.child(item_path).set({
            "name": item_name,
            "price": item_price,
            "description": item_desc,
            "original_image": image_base64
        })

        # 3) Save image file locally under datasets/test/cloth/
        cloth_folder = os.path.join(base_dir, "datasets", "test", "cloth")
        os.makedirs(cloth_folder, exist_ok=True)
        cloth_path = os.path.join(cloth_folder, uploaded_image.name)
        with open(cloth_path, "wb") as f:
            f.write(image_bytes)

        # 4) Run the virtual try-on pipeline
        with st.spinner("🧪 Generating virtual try-ons (this may take a few minutes)..."):
            success = run_virtual_tryon(cloth_path)

        if not success:
            st.error("❌ Failed to generate try-on images.")
            return

        # 5) Read each overlayed image under results/ and push to Firebase
        results_folder = os.path.join(base_dir, "results")
        if os.path.exists(results_folder):
            overlayed_images = [
                os.path.join(results_folder, fn)
                for fn in os.listdir(results_folder)
                if fn.lower().endswith((".jpg", ".png"))
            ]
            if overlayed_images:
                updates = {}
                for i, img_path in enumerate(sorted(overlayed_images)):
                    enc = encode_image(img_path)
                    if enc:
                        updates[f"overlayed_{i}"] = enc
                    else:
                        st.warning(f"⚠️ Failed to encode overlay image: {os.path.basename(img_path)}")
                if updates:
                    db.child(item_path).update(updates)
                    st.success("✅ Item uploaded successfully with try-on results!")
                    st.balloons()
                else:
                    st.error("⚠️ All try-on results failed to encode.")
            else:
                st.warning("⚠️ No overlayed images found in results/.")
        else:
            st.warning("⚠️ 'results/' folder does not exist after pipeline.")

    except Exception as e:
        st.error(f"⚠️ Error uploading item: {e}")

def get_shop_items(shop_no: str) -> dict:
    """Fetches all items of a shop from Firebase."""
    try:
        items = db.child(f"Shops/{shop_no}/items").get().val()
        return items or {}
    except Exception as e:
        st.error(f"⚠️ Error fetching items: {e}")
        return {}

def show_dashboard():
    """Displays the Shopkeeper Dashboard (upload form + inventory)."""
    if not st.session_state.get("shopkeeper_logged_in"):
        st.error("🔒 Please login through the main portal first")
        st.rerun()

    # ── Load and embed background (ma.jpg) ────────────────────────────────────
    try:
        with open(os.path.join(base_dir, "ma.jpg"), "rb") as img_file:
            encoded_bg = base64.b64encode(img_file.read()).decode()
        st.markdown(f"""
        <style>
        @keyframes boxFade {{
            0%, 100% {{ 
                background-color: rgba(44, 44, 44, 0.95);
                border-color: rgba(64, 64, 64, 0.95);
            }}
            50% {{ 
                background-color: rgba(44, 44, 44, 0.2);
                border-color: rgba(64, 64, 64, 0.2);
            }}
        }}

        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                        url('data:image/png;base64,{encoded_bg}') no-repeat center/cover !important;
            background-size: cover !important;
        }}

        div[data-testid="stExpander"] {{
            position: relative;
            border-radius: 8px !important;
            padding: 15px !important;
            margin-bottom: 15px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
            background-color: rgba(44, 44, 44, 0.95);
            animation: boxFade 5s ease-in-out infinite;
            transition: all 0.3s ease;
        }}

        div[data-testid="stExpander"] > div:first-child {{
            color: white !important;
            background-color: transparent !important;
            z-index: 2;
            position: relative;
        }}

        div[data-testid="stExpander"] > div:last-child {{
            color: white !important;
            background-color: transparent !important;
            z-index: 2;
            position: relative;
        }}

        div[data-testid="stExpander"] p,
        div[data-testid="stExpander"] span,
        div[data-testid="stExpander"] strong {{
            color: #FFFFFF !important;
            position: relative;
            z-index: 2;
        }}

        div[data-testid="stExpander"] button {{
            background-color: #404040 !important;
            color: white !important;
            border: 1px solid #505050 !important;
            position: relative;
            z-index: 2;
        }}
        </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        # If ma.jpg is not present, skip background styling.
        pass

    st.title(f"🏪 {st.session_state.shop_no} - Shopkeeper Dashboard")

    # ===== Upload Section =====
    st.subheader("Upload New Item")
    item_name = st.text_input("Item Name")
    item_price = st.number_input("Price", min_value=0.0, format="%.2f")
    item_color = st.text_input("Color")
    item_type = st.text_input("Type")
    item_length = st.text_input("Length")
    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if st.button("Upload Item", use_container_width=True):
        if not item_name:
            st.error("⚠️ Please enter item name")
        elif item_price <= 0:
            st.error("⚠️ Please enter a valid price")
        elif not uploaded_image:
            st.error("⚠️ Please upload an image")
        else:
            item_desc = f"Color: {item_color} | Type: {item_type} | Length: {item_length}"
            upload_item(
                st.session_state.shop_no,
                item_name,
                item_price,
                item_desc,
                uploaded_image
            )

    st.markdown("""
    <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <p style='color: #856404; margin: 0;'>
            ⚠️ Note: Please take photos with a white background for best results.
            After uploading, virtual try-on will run automatically (may take a few minutes).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ===== Display Inventory =====
    st.subheader("🛒 Shop Inventory")
    items = get_shop_items(st.session_state.shop_no)

    if items:
        for item_id, item_data in items.items():
            col1, col2 = st.columns([8, 2])
            with col1:
                with st.expander(f"🔹 {item_data.get('name', 'Unnamed Item')}"):
                    desc = item_data.get("description", "")
                    color, type_, length = "", "", ""
                    if desc and "Color:" in desc and "Type:" in desc and "Length:" in desc:
                        try:
                            parts = {
                                part.strip().split(":", 1)[0]: part.strip().split(":", 1)[1]
                                for part in desc.split("|")
                                if ":" in part
                            }
                            color = parts.get("Color", "").strip()
                            type_ = parts.get("Type", "").strip()
                            length = parts.get("Length", "").strip()
                        except:
                            color = type_ = length = ""

                    st.markdown(f"""
                        <span style='font-size:1.5rem; font-weight:bold;'>
                            <span style='margin-right:60px;'>💰 Price: {item_data.get('price', 'N/A')}</span>
                            <span style='margin-right:60px;'>🎨 Color: {color}</span>
                            <span style='margin-right:60px;'>👕 Type: {type_}</span>
                            <span>📏 Length: {length}</span>
                        </span>
                    """, unsafe_allow_html=True)

                    if "original_image" in item_data:
                        img_bytes = base64.b64decode(item_data["original_image"])
                        img = BytesIO(img_bytes)
                        st.image(img, width=250, caption="Original Image")

                    overlay_keys = [k for k in item_data.keys() if k.startswith("overlayed_")]
                    if overlay_keys:
                        st.subheader("👕 Try-On Results")
                        cols = st.columns(3)
                        for i, key in enumerate(sorted(overlay_keys, key=lambda k: int(k.split("_")[1]))):
                            img_bytes = base64.b64decode(item_data[key])
                            img = BytesIO(img_bytes)
                            with cols[i % 3]:
                                st.image(img, width=250, caption=f"Model {i+1}", use_container_width=True)
                    else:
                        st.warning("⚠️ No overlayed images available.")
            with col2:
                if st.button("🗑️", key=f"delete_{item_id}"):
                    try:
                        db.child(f"Shops/{st.session_state.shop_no}/items/{item_id}").remove()
                        st.success(f"✅ {item_data.get('name', 'Item')} deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"⚠️ Error deleting item: {e}")
    else:
        st.warning("⚠️ No items found in the shop.")

    # Extra space before Logout
    st.write("\n" * 5)
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

if __name__ == "__main__":
    show_dashboard()
