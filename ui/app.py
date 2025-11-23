"""
Streamlit UI Application

Web interface for real estate information extraction with ML/DL models.
"""

# import streamlit as st
# import pandas as pd
# from pathlib import Path
# import sys

# # Add src to path
# sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# from data_loader.preprocess import TextPreprocessor, Tokenizer
# from utils.post_process import RealEstatePostProcessor

# # Configure page
# st.set_page_config(
#     page_title="Real Estate Information Extraction",
#     page_icon="🏠",
#     layout="wide"
# )

# # Title
# st.title("Real Estate Information Extraction")
# st.markdown("Extract key information from real estate listings using ML/DL models")

# # Sidebar
# with st.sidebar:
#     st.header("Configuration")
    
#     model_type = st.radio(
#         "Select Model Type:",
#         options=["Machine Learning", "Deep Learning", "Ensemble"],
#         help="Choose between traditional ML or deep learning models"
#     )
    
#     if model_type == "Machine Learning":
#         ml_model = st.selectbox(
#             "ML Model:",
#             options=["CRF", "SVM", "Logistic Regression"]
#         )
#     elif model_type == "Deep Learning":
#         dl_model = st.selectbox(
#             "DL Model:",
#             options=["PhoBERT", "mBERT", "XLM-R"]
#         )
    
#     confidence_threshold = st.slider(
#         "Confidence Threshold:",
#         min_value=0.0,
#         max_value=1.0,
#         value=0.5,
#         step=0.05
#     )

# # Main content
# tab1, tab2, tab3 = st.tabs(["Single Text", "Batch Processing", "Analytics"])

# with tab1:
#     st.subheader("Extract Information from Text")
    
#     # Input area
#     input_text = st.text_area(
#         "Enter real estate listing text:",
#         height=200,
#         placeholder="Paste your real estate listing here..."
#     )
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         if st.button("Extract Information", key="extract_single"):
#             if input_text:
#                 st.info("Processing text...")
#                 # Add extraction logic here
                
#                 # Example output
#                 with st.spinner("Extracting information..."):
#                     extracted_info = {
#                         "Price": "$250,000",
#                         "Location": "Downtown Area",
#                         "Area": "2,500 sq ft",
#                         "Bedrooms": "3",
#                         "Property Type": "Apartment"
#                     }
                
#                 st.success("Information extracted successfully!")
                
#                 # Display results
#                 col_results = st.columns(2)
#                 for idx, (key, value) in enumerate(extracted_info.items()):
#                     with col_results[idx % 2]:
#                         st.metric(key, value)
#             else:
#                 st.warning("Please enter some text first.")
    
#     with col2:
#         st.subheader("Preprocessing Options")
        
#         lowercase = st.checkbox("Convert to lowercase", value=False)
#         remove_accents = st.checkbox("Remove accents", value=False)
#         remove_urls = st.checkbox("Remove URLs", value=True)
#         clean_special = st.checkbox("Clean special characters", value=False)

# with tab2:
#     st.subheader("Batch Process Multiple Listings")
    
#     uploaded_file = st.file_uploader(
#         "Upload CSV or JSON file",
#         type=["csv", "json"]
#     )
    
#     if uploaded_file:
#         # Load file
#         if uploaded_file.type == "text/csv":
#             df = pd.read_csv(uploaded_file)
#         else:
#             import json
#             df = pd.DataFrame(json.load(uploaded_file))
        
#         st.dataframe(df.head(), use_container_width=True)
        
#         if st.button("🚀 Process Batch"):
#             st.info(f"Processing {len(df)} listings...")
            
#             # Show progress
#             progress_bar = st.progress(0)
#             for i in range(len(df)):
#                 progress_bar.progress((i + 1) / len(df))
            
#             st.success("Batch processing completed!")
            
#             # Show results
#             results_df = df.copy()
#             results_df['extracted_price'] = ["$250,000"] * len(df)
#             results_df['extracted_location'] = ["Downtown"] * len(df)
            
#             st.dataframe(results_df, use_container_width=True)
            
#             # Download results
#             csv = results_df.to_csv(index=False)
#             st.download_button(
#                 label="📥 Download Results (CSV)",
#                 data=csv,
#                 file_name="extraction_results.csv",
#                 mime="text/csv"
#             )

# with tab3:
#     st.subheader("Model Performance Analytics")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.metric("Accuracy", "92.5%", "+2.5%")
    
#     with col2:
#         st.metric("F1-Score", "0.91", "+0.03")
    
#     with col3:
#         st.metric("Processing Speed", "125ms", "-20%")
    
#     # Performance chart
#     st.subheader("Model Comparison")
    
#     comparison_data = {
#         "Model": ["CRF", "SVM", "LogReg", "PhoBERT"],
#         "F1-Score": [0.84, 0.83, 0.805, 0.91],
#         "Precision": [0.85, 0.82, 0.80, 0.92],
#         "Recall": [0.83, 0.84, 0.81, 0.90]
#     }
    
#     df_comparison = pd.DataFrame(comparison_data)
#     st.line_chart(df_comparison.set_index("Model"))

# # Footer
# st.divider()
# st.markdown("""
# ---
# **Real Estate Information Extraction System** | ML/DL Comparative Study
# """)
