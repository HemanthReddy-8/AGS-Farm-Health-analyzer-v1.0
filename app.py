
import streamlit as st
from PIL import Image
import time
import pandas as pd
import cv2 
import numpy as np
import concurrent.futures
import io 

# Import all the functions from backend pipeline files
from pipeline.pest_pipeline import (
    load_pest_model,
    run_pest_detection_batch,
    run_etl_calculation
)
from pipeline.disease_pipeline import (
    load_dino_model,
    run_crop_classification,
    load_clip_classifier,
    run_health_classification,
    run_disease_classification
)
# Worker Functions for Parallel Processing

def run_pest_pipeline(image_batch, model):
    """Runs the pest detection logic and returns the results."""
    try:
        # This function now returns images_by_pest (a dict)
        total_pest_counts, images_by_pest = run_pest_detection_batch(image_batch, model)
        return total_pest_counts, images_by_pest
    except Exception as e:
        return None, f"Error in Pest Pipeline: {e}"

def run_disease_pipeline(image_batch, dino_model, dino_processor, dino_device, clip_classifier):
    """Runs the full crop/health/disease logic and returns the results."""
    try:
        final_sorting_results = {}
        crop_groups = run_crop_classification(image_batch, dino_model, dino_processor, dino_device)

        for crop_name, image_group in crop_groups.items():
            if not image_group: 
                final_sorting_results[crop_name] = {"healthy": [], "unhealthy_by_disease": {}}
                continue
            
            health_results = run_health_classification(crop_name, image_group, clip_classifier)
            healthy_images = health_results["healthy"]
            unhealthy_images = health_results["unhealthy"]

            disease_groups = {}
            if unhealthy_images:
                disease_groups = run_disease_classification(
                    crop_name, unhealthy_images, dino_model, dino_processor, dino_device
                )
            
            final_sorting_results[crop_name] = {
                "healthy": healthy_images,
                "unhealthy_by_disease": disease_groups
            }
        return final_sorting_results
    except Exception as e:
        return None, f"Error in Disease Pipeline: {e}"
def reset_app():
    # Set the file_uploader key to an empty list to clear it
    st.session_state.file_uploader = []

# Streamlit App Configuration
st.set_page_config(layout="wide", page_title="AGS Farmhealth analyser Tool")

col1, col2 = st.columns([1, 6]) 

with col1:
    st.image("assets/logo.jpeg", use_container_width=True) 

with col2:
    st.title("AGS Farm Health Analyser")
    st.write("Upload a batch of images to run Pest/ETL and Crop/Disease analysis.")


# CUSTOM CSS STYLING 
st.markdown("""
    <style>
    
    /* --- Main App Background --- */
    [data-testid="stAppViewContainer"] {
        background-color: #F0F4F0; /* A light gray background */
    }
    

    /* --- Main Theme Colors --- */
    :root {
        --primary-color: #4CAF50; /* Bright, professional green */
        --dark-green: #2E7D32;     /* Deeper green for titles */
        --light-green-bg: #F0F4F0; /* Light green for sidebar/forms */
        --text-color: #333333;
        --light-gray: #F5F5F5;
        --border-color: #E0E0E0;
    }
    /* ... (rest of your CSS is unchanged) ... */
    /* --- General Page --- */
    .stApp {
        color: var(--text-color);
    }
    
    /* --- Title --- */
    h1 {
        color: var(--dark-green);
        font-weight: 600;
    }

    /* --- Headers --- */
    h2 { /* st.header */
        color: var(--dark-green);
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 5px;
    }
    h3 { /* st.subheader */
        color: var(--primary-color);
    }

    /* --- Main Analysis Button --- */
    div.stButton > button[kind="primary"] {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1.1em;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: var(--dark-green);
    }

    /* --- File Uploader --- */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--primary-color);
        background-color: var(--light-green-bg);
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="stFileUploader"] label {
        color: var(--dark-green) !important;
        font-weight: 600;
    }
    
    
    
    /* --- Download Button (make it smaller) --- */
    div.stDownloadButton > button {
        background-color: #F0F0F0;
        color: #333;
        border: 1px solid #CCC;
        border-radius: 5px;
        padding: 2px 8px; /* Smaller padding */
        font-size: 0.9em; /* Smaller font */
        width: 100%; /* Make it fill the column */
    }
    div.stDownloadButton > button:hover {
        background-color: #E0E0E0;
        border-color: #999;
    }

    
    
    </style>
    """, unsafe_allow_html=True)

# File Uploader
uploaded_files = st.file_uploader(
    "Choose images...",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True,
    key="file_uploader"
)

if 'pest_counts' not in st.session_state:
    st.session_state.pest_counts = {}
if 'etl_inputs_ready' not in st.session_state:
    st.session_state.etl_inputs_ready = False
if 'images_by_pest' not in st.session_state:
    st.session_state.images_by_pest = {}
if 'final_sorting_results' not in st.session_state:
    st.session_state.final_sorting_results = {}
if 'image_batch_with_names' not in st.session_state:
    st.session_state.image_batch_with_names = []
if 'processed_filenames' not in st.session_state:
    st.session_state.processed_filenames = []


# Main Processing Logic

if uploaded_files:
    
    current_filenames = [f.name for f in uploaded_files]
    
    if st.session_state.processed_filenames != current_filenames:
        
        st.session_state.pest_counts = {}
        st.session_state.etl_inputs_ready = False
        st.session_state.images_by_pest = {} 
        st.session_state.final_sorting_results = {}
        st.session_state.image_batch_with_names = []
        
        temp_image_batch = []
        with st.spinner(f"Loading {len(uploaded_files)} images..."):
            for uploaded_file in uploaded_files:
                img = Image.open(uploaded_file)
                temp_image_batch.append((uploaded_file.name, img.copy()))
        
        st.session_state.image_batch_with_names = temp_image_batch
        st.session_state.processed_filenames = current_filenames
    
    # --- ALWAYS: Display Thumbnails (fast) ---
    st.subheader(f"Uploaded {len(st.session_state.image_batch_with_names)} Image(s)")
    cols = st.columns(min(len(st.session_state.image_batch_with_names), 10))
    for i, (name, img) in enumerate(st.session_state.image_batch_with_names):
        with cols[i % 10]:
            st.image(img, caption=name, width=100)
    st.markdown("---")


    # --- Analysis Button (COMPUTATION ONLY) ---
    if st.button(f"Run Full Analysis on {len(uploaded_files)} Images", type="primary", use_container_width=True):
        
        st.session_state.pest_counts = {}
        st.session_state.etl_inputs_ready = False
        st.session_state.images_by_pest = {} # --- RESET new key ---
        st.session_state.final_sorting_results = {}
        
        with st.spinner("Loading analysis models... (first time might take a while)"):
            pest_model = load_pest_model()
            dino_model, dino_processor, dino_device = load_dino_model()
            clip_classifier = load_clip_classifier()

        if not all([pest_model, dino_model, clip_classifier]):
            st.error("One or more models failed to load. Cannot proceed.")
        else:
            
            # --- SEQUENTIAL EXECUTION (To save memory) ---
            image_batch = st.session_state.image_batch_with_names
            
            with st.spinner("Step 1/2: Running Pest analysis..."):
                pest_result = run_pest_pipeline(image_batch, pest_model)
            
            with st.spinner("Step 2/2: Running Disease analysis..."):
                disease_result = run_disease_pipeline(image_batch, dino_model, dino_processor, dino_device, clip_classifier)
            
            # --- Unpack and SAVE results to session_state ---
            if pest_result[0] is not None:
                total_pest_counts, images_by_pest = pest_result # --- RENAMED variable ---
                st.session_state.pest_counts = total_pest_counts
                st.session_state.images_by_pest = images_by_pest # --- SAVE new structure ---
                if total_pest_counts:
                    st.session_state.etl_inputs_ready = True
            else:
                st.error(pest_result[1])

            if isinstance(disease_result, dict):
                st.session_state.final_sorting_results = disease_result
            else:
                st.error(disease_result[1])
    
elif not uploaded_files and st.session_state.processed_filenames:
    # --- RESET IF FILES ARE CLEARED ---
    st.session_state.pest_counts = {}
    st.session_state.etl_inputs_ready = False
    st.session_state.images_by_pest = {} # --- RESET new key ---
    st.session_state.final_sorting_results = {}
    st.session_state.image_batch_with_names = []
    st.session_state.processed_filenames = []
    st.rerun()

# -----------------------------------------------------------------
# --- DISPLAY LOGIC (RUNS EVERY TIME) ---
# -----------------------------------------------------------------

# --- UPDATED this condition ---
if st.session_state.images_by_pest or st.session_state.final_sorting_results:
    
    col1, col2 = st.columns(2)

    # ---------------------------------------------------
    # --- BRANCH 1: Display Pest & ETL (REWRITTEN) ---
    # ---------------------------------------------------
    with col1:
        st.header("Pest & ETL Analysis")
        st.subheader("Pest Counts Found:")
        
        total_pest_counts = st.session_state.pest_counts
        images_by_pest = st.session_state.images_by_pest
        
        if not total_pest_counts:
            st.warning("No pests detected in the batch.")
        else:
            st.dataframe(pd.DataFrame(total_pest_counts.items(), columns=['Pest', 'Total Count']), use_container_width=True)
        
        if images_by_pest:
            st.subheader("Annotated Pest Images (Verification)")
            
            # Create tab names like "Aphid (3 images)"
            pest_tab_names = [f"{pest_name} ({len(images)} images)" 
                              for pest_name, images in images_by_pest.items()]
            
            pest_tabs = st.tabs(pest_tab_names)
            
            # Loop through the dictionary and tabs together
            for i, (pest_name, images) in enumerate(images_by_pest.items()):
                with pest_tabs[i]:
                    # Now use the grid logic inside *this* tab
                    num_cols = 5 
                    cols = st.columns(num_cols)
                    for idx, (filename, img) in enumerate(images):
                        with cols[idx % num_cols]:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, caption=filename, use_container_width=True)
                            
                            _ , buf = cv2.imencode(".png", img)
                            data_bytes = buf.tobytes()
                            st.download_button(
                                label="Download",
                                data=data_bytes,
                                file_name=f"annotated_{filename}.png",
                                mime="image/png",
                                key=f"pest_dl_{pest_name}_{idx}" # Make key unique
                            )
        
    # --- BRANCH 2: Display Crop & Disease (Unchanged) ---
    with col2:
        st.header("Crop & Disease Analasys")
        st.subheader("Image Analysis Results:")
        
        final_sorting_results = st.session_state.final_sorting_results
        
        if not final_sorting_results:
            st.warning("Disease classification did not return results.")
        else:
            crop_tab_names = [f"{crop} ({len(data['healthy']) + sum(len(imgs) for imgs in data['unhealthy_by_disease'].values())} images)" for crop, data in final_sorting_results.items()]
            
            if not crop_tab_names:
                st.warning("Crop classification did not return any results.")
            else:
                crop_tabs = st.tabs(crop_tab_names)
                for i, (crop_name, data) in enumerate(final_sorting_results.items()):
                    with crop_tabs[i]:
                        healthy_count = len(data['healthy'])
                        unhealthy_count = sum(len(dis_imgs) for dis_imgs in data['unhealthy_by_disease'].values())
                        health_tabs = st.tabs([f"HEALTHY ({healthy_count})", f"UNHEALTHY ({unhealthy_count})"])
                        
                        with health_tabs[0]:
                            image_list = data['healthy']
                            if image_list:
                                num_cols = 4
                                cols = st.columns(num_cols)
                                for idx, (fname, img) in enumerate(image_list):
                                    with cols[idx % num_cols]:
                                        st.image(img, caption=fname, use_container_width=True)
                                        
                                        buf = io.BytesIO()
                                        img.save(buf, format="PNG")
                                        data_bytes = buf.getvalue()
                                        st.download_button(
                                            label="Download",
                                            data=data_bytes,
                                            file_name=fname,
                                            mime="image/png",
                                            key=f"healthy_{crop_name}_{idx}"
                                        )
                            else:
                                st.write("No healthy images found for this crop.")
                        
                        with health_tabs[1]:
                            if not data['unhealthy_by_disease']:
                                st.write("No unhealthy images found for this crop.")
                            else:
                                disease_tab_names = [f"{disease} ({len(images)} images)" for disease, images in data['unhealthy_by_disease'].items()]
                                disease_tabs = st.tabs(disease_tab_names)
                                
                                for j, (disease_name, images) in enumerate(data['unhealthy_by_disease'].items()):
                                    with disease_tabs[j]:
                                        if images:
                                            num_cols = 4
                                            cols = st.columns(num_cols)
                                            for idx, (fname, img) in enumerate(images):
                                                with cols[idx % num_cols]:
                                                    st.image(img, caption=fname, use_container_width=True)
                                                    
                                                    buf = io.BytesIO()
                                                    img.save(buf, format="PNG")
                                                    data_bytes = buf.getvalue()
                                                    st.download_button(
                                                        label="Download",
                                                        data=data_bytes,
                                                        file_name=fname,
                                                        mime="image/png",
                                                        key=f"unhealthy_{crop_name}_{disease_name}_{idx}"
                                                    )
                                        else:
                                            st.write(f"No images found for {disease_name}.")

# --- ETL Input Form (RUNS EVERY TIME) ---
if st.session_state.get('etl_inputs', False):
    st.markdown("---")
    st.header("Enter ETL Calculation Parameters")
    st.warning("Provide initial damage index (I), control cost (C), market price, and environmental factor (fev_con) for each detected pest.")
    etl_input_rows = []
    pest_counts = st.session_state.pest_counts
    with st.form("etl_input_form"):
        for pest_name, n_count in pest_counts.items():
            st.subheader(f"Parameters for: {pest_name} (Detected Count: {n_count})")
            cols = st.columns(4)
            i_old = cols[0].number_input(f"Initial Damage Index (I) for {pest_name}", min_value=0.0, value=0.1, step=0.01, format="%.3f", key=f"i_{pest_name}")
            c_cost = cols[1].number_input(f"Control Cost (C) for {pest_name}", min_value=0.0, value=10.0, step=0.5, format="%.2f", key=f"c_{pest_name}")
            market_price = cols[2].number_input(f"Market Price/kg for {pest_name}", min_value=0.0, value=5.0, step=0.1, format="%.2f", key=f"mkt_{pest_name}")
            fev_con = cols[3].number_input(f"Environmental Factor (fev_con) for {pest_name}", min_value=0.0, value=20.0, step=0.5, format="%.1f", key=f"fev_{pest_name}")
            etl_input_rows.append((pest_name, n_count, i_old, 0, c_cost, market_price, 0, 0, fev_con))
        
        submitted = st.form_submit_button("Calculate ETL", type="primary", use_container_width=True)
        
        if submitted:
            if not etl_input_rows:
                st.error("No pest data available to calculate ETL.")
            else:
                st.info("Calculating ETL based on provided parameters...")
                df_etl, df_progress, etl_fig = run_etl_calculation(etl_input_rows)
                st.header("📊 ETL Calculation Results")
                if df_etl.empty and df_progress.empty:
                    st.warning("ETL calculation did not produce results.")
                else:
                    if not df_etl.empty:
                        st.subheader("Estimated ETL Days (±10% Range)")
                        st.dataframe(df_etl[["Pest Name", "ETL Range (Days)"]], use_container_width=True)
                    if not df_progress.empty:
                        st.subheader("Full ETL Progression Data")
                        st.dataframe(df_progress, use_container_width=True)
                    if etl_fig:
                        st.subheader("Pest Severity Progression Over Time")
                        st.plotly_chart(etl_fig, use_container_width=True)

# --- NEW: Reset Button ---
st.sidebar.button("Clear All & Reset App", on_click=reset_app, use_container_width=True, type="secondary")
# -------------------------

# Add instructions or footer if needed
st.sidebar.info("Upload multiple images and click 'Run Analysis'. Provide ETL parameters when prompted.")
