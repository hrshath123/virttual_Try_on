import streamlit as st
import os
import subprocess
import time


def run_virtual_tryon(cloth_path):
    """Runs the virtual try-on backend script using the correct virtual environment."""
    venv_python = r"C:/Users/MSI/Desktop/clothes wala/Virtual-Try-On/venv/Scripts/python.exe"
    subprocess.run([venv_python, "automated.py", cloth_path])


def get_result_images(results_folder):
    """Fetches processed images directly from the results folder."""
    if not os.path.exists(results_folder):
        return []
    
    return [
        os.path.join(results_folder, img)
        for img in os.listdir(results_folder)
        if img.endswith(('.jpg', '.png'))
    ]

# Streamlit UI
st.title("👕 Virtual Try-On System")
st.write("Upload a clothing image and see it applied on all models!")

# File uploader
uploaded_file = st.file_uploader("Choose a clothing image", type=["jpg", "png"])

if uploaded_file is not None:
    cloth_folder = "datasets/test/cloth/"
    results_folder = "results/"
    os.makedirs(cloth_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    cloth_path = os.path.join(cloth_folder, uploaded_file.name)
    with open(cloth_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ Image saved: {uploaded_file.name}")

    if st.button("Run Virtual Try-On"):
        with st.spinner("Processing... Please wait ⏳"):
            run_virtual_tryon(cloth_path)
            time.sleep(5)  # Allow script to process files
        st.success("🎉 Processing complete! Check the results below.")

        # Extract cloth name (without extension) for result folder lookup
        cloth_name = os.path.splitext(uploaded_file.name)[0]

        # Display and provide download links for generated images
        # Remove cloth_name argument (not needed anymore)
        result_images = get_result_images(results_folder)

        if result_images:
            for img_path in result_images:
                st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
                with open(img_path, "rb") as file:
                    st.download_button(label="Download", data=file, file_name=os.path.basename(img_path), mime="image/jpeg")
        else:
            st.warning("⚠️ No output images found. Please check if the try-on process completed successfully.")
