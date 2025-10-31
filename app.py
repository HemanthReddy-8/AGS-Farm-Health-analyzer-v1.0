# import streamlit as st
# from PIL import Image
# import time
# import pandas as pd
# import cv2 
# import numpy as np
# import concurrent.futures

# # Import all the functions from backend pipeline files
# from pipeline.pest_pipeline import (
#     load_pest_model,
#     run_pest_detection_batch,
#     run_etl_calculation
# )
# from pipeline.disease_pipeline import (
#     load_dino_model,
#     run_crop_classification,
#     load_clip_classifier,
#     run_health_classification,
#     run_disease_classification
# )

# def run_pest_pipeline(image_batch, model):
#     """Runs the pest detection logic and returns the results."""
#     try:
#         total_pest_counts, annotated_images = run_pest_detection_batch(image_batch, model)
#         return total_pest_counts, annotated_images
#     except Exception as e:
#         return None, f"Error in Pest Pipeline: {e}"

# def run_disease_pipeline(image_batch, dino_model, dino_processor, dino_device, clip_classifier):
#     """Runs the full crop/health/disease logic and returns the results."""
#     try:
#         final_sorting_results = {}
#         crop_groups = run_crop_classification(image_batch, dino_model, dino_processor, dino_device)

#         for crop_name, image_group in crop_groups.items():
#             if not image_group: 
#                 final_sorting_results[crop_name] = {"healthy": [], "unhealthy_by_disease": {}}
#                 continue
            
#             health_results = run_health_classification(crop_name, image_group, clip_classifier)
#             healthy_images = health_results["healthy"]
#             unhealthy_images = health_results["unhealthy"]

#             disease_groups = {}
#             if unhealthy_images:
#                 disease_groups = run_disease_classification(
#                     crop_name, unhealthy_images, dino_model, dino_processor, dino_device
#                 )
            
#             final_sorting_results[crop_name] = {
#                 "healthy": healthy_images,
#                 "unhealthy_by_disease": disease_groups
#             }
#         return final_sorting_results
#     except Exception as e:
#         return None, f"Error in Disease Pipeline: {e}"

# # Streamlit App Configuration
# st.set_page_config(layout="wide", page_title="AGS Farmhealth analyser Tool")
# st.title("AGS Farm Health Analyser")
# st.write("Upload a batch of images to run Pest/ETL and Crop/Disease analysis.")

# st.markdown("""
#     <style>
#     /* --- Main Theme Colors --- */
#     :root {
#         --primary-color: #4CAF50; /* Bright, professional green */
#         --dark-green: #2E7D32;     /* Deeper green for titles */
#         --light-green-bg: #F0F4F0; /* Light green for sidebar/forms */
#         --text-color: #333333;
#         --light-gray: #F5F5F5;
#         --border-color: #E0E0E0;
#     }

#     /* --- General Page --- */
#     .stApp {
#         color: var(--text-color);
#     }
    
#     /* --- Title --- */
#     h1 {
#         color: var(--dark-green);
#         font-weight: 600;
#     }

#     /* --- Headers --- */
#     h2 { /* st.header */
#         color: var(--dark-green);
#         border-bottom: 2px solid var(--border-color);
#         padding-bottom: 5px;
#     }
#     h3 { /* st.subheader */
#         color: var(--primary-color);
#     }

#     /* --- Main Analysis Button --- */
#     div.stButton > button[kind="primary"] {
#         background-color: var(--primary-color);
#         color: white;
#         border: none;
#         border-radius: 8px;
#         padding: 12px 24px;
#         font-size: 1.1em;
#         font-weight: 600;
#         transition: background-color 0.3s ease;
#     }
#     div.stButton > button[kind="primary"]:hover {
#         background-color: var(--dark-green);
#     }

#     /* --- File Uploader --- */
#     [data-testid="stFileUploader"] {
#         border: 2px dashed var(--primary-color);
#         background-color: var(--light-green-bg);
#         border-radius: 10px;
#         padding: 15px;
#     }

    
    
    
    
#     </style>
#     """, unsafe_allow_html=True)

# # File Uploader
# uploaded_files = st.file_uploader(
#     "Choose images...",
#     type=['png', 'jpg', 'jpeg'],
#     accept_multiple_files=True
# )

# # Session state
# if 'pest_counts' not in st.session_state:
#     st.session_state.pest_counts = {}
# if 'etl_inputs_ready' not in st.session_state:
#     st.session_state.etl_inputs_ready = False

# # Main Processing Logic
# if uploaded_files:

#     # --- Display Thumbnails ---
#     st.subheader(f"Uploaded {len(uploaded_files)} Image(s)")
#     cols = st.columns(min(len(uploaded_files), 10))
#     image_batch_pil = []
#     for i, uploaded_file in enumerate(uploaded_files):
#         img = Image.open(uploaded_file)
#         image_batch_pil.append(img)
#         with cols[i % 10]:
#             st.image(img, caption=uploaded_file.name, width=100)
#     st.markdown("---")

#     image_batch_with_names = []
#     for i, uploaded_file in enumerate(uploaded_files):
#         image_batch_with_names.append((uploaded_file.name, image_batch_pil[i]))

#     # --- Analysis Button ---
#     if st.button(f"Run Full Analysis on {len(uploaded_files)} Images", type="primary", use_container_width=True):
#         st.session_state.pest_counts = {}
#         st.session_state.etl_inputs_ready = False
        
#         # --- Clear any old session state keys from previous navigation attempts ---
#         for key in list(st.session_state.keys()):
#             if '_healthy' in key or '_unhealthy' in key or 'pest_viewer' in key:
#                 del st.session_state[key]

#         with st.spinner("Loading analysis models... (first time might take a while)"):
#             pest_model = load_pest_model()
#             dino_model, dino_processor, dino_device = load_dino_model()
#             clip_classifier = load_clip_classifier()

#         if not all([pest_model, dino_model, clip_classifier]):
#             st.error("One or more models failed to load. Cannot proceed.")
#         else:
#             with st.spinner("Running Pest and Disease analysis in parallel... This may take a moment."):
#                 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#                     pest_future = executor.submit(run_pest_pipeline, image_batch_with_names, pest_model)
#                     disease_future = executor.submit(run_disease_pipeline, image_batch_with_names, dino_model, dino_processor, dino_device, clip_classifier)
#                     pest_result = pest_future.result()
#                     disease_result = disease_future.result()
            
#             # Unpack pest results
#             if pest_result[0] is not None:
#                 total_pest_counts, annotated_images = pest_result
#                 st.session_state.pest_counts = total_pest_counts
#                 if total_pest_counts:
#                     st.session_state.etl_inputs_ready = True
#             else:
#                 total_pest_counts, annotated_images = {}, []
#                 st.error(pest_result[1])

#             # Unpack disease results
#             if isinstance(disease_result, dict):
#                 final_sorting_results = disease_result
#             else:
#                 final_sorting_results = {}
#                 st.error(disease_result[1])

#             # --- Create columns for results ---
#             col1, col2 = st.columns(2)

#             # --- BRANCH 1: Display Pest & ETL ---
#             with col1:
#                 st.header("Pest & ETL Analysis")
#                 st.subheader("Pest Counts Found:")
#                 if not total_pest_counts:
#                     st.warning("No pests detected in the batch.")
#                 else:
#                     st.dataframe(pd.DataFrame(total_pest_counts.items(), columns=['Pest', 'Total Count']), use_container_width=True)
                
#                 # --- PEST VIEWER (GRID) ---
#                 if annotated_images:
#                     st.subheader("Annotated Pest Images (Verification)")
#                     with st.expander(f"Click to view {len(annotated_images)} images with detections", expanded=True):
                        
#                         # Define number of columns for the grid
#                         num_cols = 5 # 5 columns in this half of the page
#                         cols = st.columns(num_cols)
#                         for idx, (filename, img) in enumerate(annotated_images):
#                             with cols[idx % num_cols]:
#                                 # Use use_container_width=True to fill the column
#                                 st.image(
#                                     cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
#                                     caption=filename, 
#                                     use_container_width=True 
#                                 )

#             # --- BRANCH 2: Display Crop & Disease ---
#             with col2:
#                 st.header("Crop & Disease Sorting")
#                 st.subheader("Image Sorting Results:")
                
#                 if not final_sorting_results:
#                     st.warning("Disease classification did not return results.")
#                 else:
#                     crop_tab_names = [f"{crop} ({len(data['healthy']) + sum(len(imgs) for imgs in data['unhealthy_by_disease'].values())} images)" for crop, data in final_sorting_results.items()]
                    
#                     if not crop_tab_names:
#                         st.warning("Crop classification did not return any results.")
#                     else:
#                         crop_tabs = st.tabs(crop_tab_names)
#                         for i, (crop_name, data) in enumerate(final_sorting_results.items()):
#                             with crop_tabs[i]:
#                                 healthy_count = len(data['healthy'])
#                                 unhealthy_count = sum(len(dis_imgs) for dis_imgs in data['unhealthy_by_disease'].values())
#                                 health_tabs = st.tabs([f"HEALTHY ({healthy_count})", f"UNHEALTHY ({unhealthy_count})"])
                                
#                                 # --- HEALTHY TAB (GRID) ---
#                                 with health_tabs[0]:
#                                     image_list = data['healthy']
#                                     if image_list:
#                                         # Define number of columns for the grid
#                                         num_cols = 4  # 4 columns in this tab
#                                         cols = st.columns(num_cols)
#                                         for idx, (fname, img) in enumerate(image_list):
#                                             with cols[idx % num_cols]:
#                                                 st.image(img, caption=fname, use_container_width=True)
#                                     else:
#                                         st.write("No healthy images found for this crop.")
                                
#                                 # --- UNHEALTHY TAB ---
#                                 with health_tabs[1]:
#                                     if not data['unhealthy_by_disease']:
#                                         st.write("No unhealthy images found for this crop.")
#                                     else:
#                                         disease_tab_names = [f"{disease} ({len(images)} images)" for disease, images in data['unhealthy_by_disease'].items()]
#                                         disease_tabs = st.tabs(disease_tab_names)
                                        
#                                         for j, (disease_name, images) in enumerate(data['unhealthy_by_disease'].items()):
#                                             with disease_tabs[j]:
#                                                 # --- DISEASE TAB (GRID) ---
#                                                 if images:
#                                                     num_cols = 4 # 4 columns in this tab
#                                                     cols = st.columns(num_cols)
#                                                     for idx, (fname, img) in enumerate(images):
#                                                         with cols[idx % num_cols]:
#                                                             st.image(img, caption=fname, use_container_width=True)
#                                                 else:
#                                                     st.write(f"No images found for {disease_name}.")

#     # --- ETL Input Form (Unchanged) ---
#     if st.session_state.get('etl_inputs_ready', False):
#         st.markdown("---")
#         st.header("Enter ETL Calculation Parameters")
#         st.warning("Provide initial damage index (I), control cost (C), market price, and environmental factor (fev_con) for each detected pest.")
#         etl_input_rows = []
#         pest_counts = st.session_state.pest_counts
#         with st.form("etl_input_form"):
#             for pest_name, n_count in pest_counts.items():
#                 st.subheader(f"Parameters for: {pest_name} (Detected Count: {n_count})")
#                 cols = st.columns(4)
#                 i_old = cols[0].number_input(f"Initial Damage Index (I) for {pest_name}", min_value=0.0, value=0.1, step=0.01, format="%.3f", key=f"i_{pest_name}")
#                 c_cost = cols[1].number_input(f"Control Cost (C) for {pest_name}", min_value=0.0, value=10.0, step=0.5, format="%.2f", key=f"c_{pest_name}")
#                 market_price = cols[2].number_input(f"Market Price/kg for {pest_name}", min_value=0.0, value=5.0, step=0.1, format="%.2f", key=f"mkt_{pest_name}")
#                 fev_con = cols[3].number_input(f"Environmental Factor (fev_con) for {pest_name}", min_value=0.0, value=20.0, step=0.5, format="%.1f", key=f"fev_{pest_name}")
#                 etl_input_rows.append((pest_name, n_count, i_old, 0, c_cost, market_price, 0, 0, fev_con))
#             submitted = st.form_submit_button("Calculate ETL")
#             if submitted:
#                 if not etl_input_rows:
#                     st.error("No pest data available to calculate ETL.")
#                 else:
#                     st.info("Calculating ETL based on provided parameters...")
#                     df_etl, df_progress, etl_fig = run_etl_calculation(etl_input_rows)
#                     st.header("📊 ETL Calculation Results")
#                     if df_etl.empty and df_progress.empty:
#                         st.warning("ETL calculation did not produce results.")
#                     else:
#                         if not df_etl.empty:
#                             st.subheader("Estimated ETL Days (±10% Range)")
#                             st.dataframe(df_etl[["Pest Name", "ETL Range (Days)"]], use_container_width=True)
#                         if not df_progress.empty:
#                             st.subheader("Full ETL Progression Data")
#                             st.dataframe(df_progress, use_container_width=True)
#                         if etl_fig:
#                             st.subheader("Pest Severity Progression Over Time")
#                             st.plotly_chart(etl_fig, use_container_width=True)

# # Add instructions or footer if needed
# st.sidebar.info("Upload multiple images and click 'Run Analysis'. Provide ETL parameters when prompted.")


import streamlit as st
from PIL import Image
import time
import pandas as pd
import cv2 
import numpy as np
import concurrent.futures
import io  # Added for byte conversion

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

# -----------------------------------------------------------------
# --- Worker Functions for Parallel Processing ---
# (No changes here)
# -----------------------------------------------------------------

def run_pest_pipeline(image_batch, model):
    """Runs the pest detection logic and returns the results."""
    try:
        total_pest_counts, annotated_images = run_pest_detection_batch(image_batch, model)
        return total_pest_counts, annotated_images
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

# -----------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------
st.set_page_config(layout="wide", page_title="AGS Internal Tool")
st.title("AGS Farm Health Analyzer")
st.write("Upload a batch of images to run Pest/ETL and Crop/Disease analysis.")

st.markdown("""
    <style>
    /* --- Main Theme Colors --- */
    :root {
        --primary-color: #4CAF50; /* Bright, professional green */
        --dark-green: #2E7D32;     /* Deeper green for titles */
        --light-green-bg: #F0F4F0; /* Light green for sidebar/forms */
        --text-color: #333333;
        --light-gray: #F5F5F5;
        --border-color: #E0E0E0;
    }

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

    
    
    
    
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------
# File Uploader
# -----------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Choose images...",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
)

# -----------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------
if 'pest_counts' not in st.session_state:
    st.session_state.pest_counts = {}
if 'etl_inputs_ready' not in st.session_state:
    st.session_state.etl_inputs_ready = False
if 'annotated_images' not in st.session_state:
    st.session_state.annotated_images = []
if 'final_sorting_results' not in st.session_state:
    st.session_state.final_sorting_results = {}
# --- KEYS TO PREVENT RE-LOADING ---
if 'image_batch_with_names' not in st.session_state:
    st.session_state.image_batch_with_names = []
if 'processed_filenames' not in st.session_state:
    st.session_state.processed_filenames = []


# -----------------------------------------------------------------
# Main Processing Logic
# -----------------------------------------------------------------

if uploaded_files:
    
    current_filenames = [f.name for f in uploaded_files]
    
    # --- ONE-TIME-ONLY: Load images into memory ---
    if st.session_state.processed_filenames != current_filenames:
        
        # --- RESET ALL RESULTS ---
        st.session_state.pest_counts = {}
        st.session_state.etl_inputs_ready = False
        st.session_state.annotated_images = []
        st.session_state.final_sorting_results = {}
        st.session_state.image_batch_with_names = []
        
        # --- Run the slow Image.open() loop ---
        temp_image_batch = []
        with st.spinner(f"Loading {len(uploaded_files)} images..."):
            for uploaded_file in uploaded_files:
                img = Image.open(uploaded_file)
                temp_image_batch.append((uploaded_file.name, img.copy()))
        
        # --- SAVE TO SESSION STATE ---
        st.session_state.image_batch_with_names = temp_image_batch
        st.session_state.processed_filenames = current_filenames
        # No st.rerun() here, let the script continue to display thumbnails
    
    # --- ALWAYS: Display Thumbnails (fast) ---
    st.subheader(f"Uploaded {len(st.session_state.image_batch_with_names)} Image(s)")
    cols = st.columns(min(len(st.session_state.image_batch_with_names), 10))
    # Read from session state, which is now populated
    for i, (name, img) in enumerate(st.session_state.image_batch_with_names):
        with cols[i % 10]:
            st.image(img, caption=name, width=100)
    st.markdown("---")


    # --- Analysis Button (COMPUTATION ONLY) ---
    if st.button(f"Run Full Analysis on {len(uploaded_files)} Images", type="primary", use_container_width=True):
        
        # Reset analysis results, but NOT the loaded images
        st.session_state.pest_counts = {}
        st.session_state.etl_inputs_ready = False
        st.session_state.annotated_images = []
        st.session_state.final_sorting_results = {}
        
        with st.spinner("Loading analysis models... (first time might take a while)"):
            pest_model = load_pest_model()
            dino_model, dino_processor, dino_device = load_dino_model()
            clip_classifier = load_clip_classifier()

        if not all([pest_model, dino_model, clip_classifier]):
            st.error("One or more models failed to load. Cannot proceed.")
        else:
            with st.spinner("Running Pest and Disease analysis in parallel... This may take a moment."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    
                    # --- READ FROM SESSION STATE ---
                    image_batch = st.session_state.image_batch_with_names
                    
                    pest_future = executor.submit(run_pest_pipeline, image_batch, pest_model)
                    disease_future = executor.submit(run_disease_pipeline, image_batch, dino_model, dino_processor, dino_device, clip_classifier)
                    
                    pest_result = pest_future.result()
                    disease_result = disease_future.result()
            
            # --- Unpack and SAVE results to session_state ---
            if pest_result[0] is not None:
                total_pest_counts, annotated_images = pest_result
                st.session_state.pest_counts = total_pest_counts
                st.session_state.annotated_images = annotated_images
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
    st.session_state.annotated_images = []
    st.session_state.final_sorting_results = {}
    st.session_state.image_batch_with_names = []
    st.session_state.processed_filenames = []
    st.rerun()

# -----------------------------------------------------------------
# --- DISPLAY LOGIC (RUNS EVERY TIME) ---
# This block is at the end and reads from session_state
# -----------------------------------------------------------------

if st.session_state.annotated_images or st.session_state.final_sorting_results:
    
    col1, col2 = st.columns(2)

    # --- BRANCH 1: Display Pest & ETL ---
    with col1:
        st.header("🐜 Pest & ETL Analysis")
        st.subheader("Pest Counts Found:")
        
        total_pest_counts = st.session_state.pest_counts
        annotated_images = st.session_state.annotated_images
        
        if not total_pest_counts:
            st.warning("No pests detected in the batch.")
        else:
            st.dataframe(pd.DataFrame(total_pest_counts.items(), columns=['Pest', 'Total Count']), use_container_width=True)
        
        if annotated_images:
            st.subheader("Annotated Pest Images (Verification)")
            with st.expander(f"Click to view {len(annotated_images)} images with detections", expanded=True):
                
                num_cols = 5 
                cols = st.columns(num_cols)
                for idx, (filename, img) in enumerate(annotated_images):
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
                            key=f"pest_dl_{idx}"
                        )

    # --- BRANCH 2: Display Crop & Disease ---
    with col2:
        st.header("🌿 Crop & Disease Sorting")
        st.subheader("Image Sorting Results:")
        
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
if st.session_state.get('etl_inputs_ready', False):
    st.markdown("---")
    st.header("⚙️ Enter ETL Calculation Parameters")
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
        submitted = st.form_submit_button("Calculate ETL")
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

# Add instructions or footer if needed
st.sidebar.info("Upload multiple images and click 'Run Analysis'. Provide ETL parameters when prompted.")
