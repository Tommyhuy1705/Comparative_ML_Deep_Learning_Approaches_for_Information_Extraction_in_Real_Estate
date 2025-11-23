"""
Reusable UI Components

Streamlit components for the application.
"""

# import streamlit as st
# import pandas as pd
# from typing import List, Dict, Any


# def render_entity_table(entities: List[Dict[str, Any]]) -> None:
#     """
#     Render extracted entities in table format.
    
#     Args:
#         entities: List of entity dictionaries
#     """
#     if not entities:
#         st.warning("No entities extracted.")
#         return
    
#     df_entities = pd.DataFrame(entities)
#     st.dataframe(df_entities, use_container_width=True)


# def render_entity_visualization(text: str, entities: List[Dict[str, Any]]) -> None:
#     """
#     Render text with highlighted entities.
    
#     Args:
#         text: Original text
#         entities: List of extracted entities
#     """
#     # Create highlighted text
#     highlighted_text = text
    
#     for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
#         start, end = entity['start'], entity['end']
#         entity_text = text[start:end]
        
#         # Color coding for different entity types
#         color_map = {
#             'PRICE': '#FF6B6B',
#             'LOCATION': '#4ECDC4',
#             'AREA': '#45B7D1',
#             'PROPERTY_TYPE': '#FFA07A'
#         }
        
#         color = color_map.get(entity['type'], '#E8E8E8')
#         highlighted = f"<span style='background-color:{color}; padding: 3px 5px; border-radius: 3px;'>{entity_text}</span>"
#         highlighted_text = highlighted_text[:start] + highlighted + highlighted_text[end:]
    
#     st.markdown(highlighted_text, unsafe_allow_html=True)


# def render_metrics_dashboard(metrics: Dict[str, float]) -> None:
#     """
#     Render metrics dashboard.
    
#     Args:
#         metrics: Dictionary of metric names and values
#     """
#     cols = st.columns(len(metrics))
    
#     for col, (metric_name, metric_value) in zip(cols, metrics.items()):
#         with col:
#             st.metric(metric_name, f"{metric_value:.4f}")


# def render_confusion_matrix(cm_data: Any) -> None:
#     """
#     Render confusion matrix heatmap.
    
#     Args:
#         cm_data: Confusion matrix data
#     """
#     try:
#         import plotly.figure_factory as ff
        
#         fig = ff.create_annotated_heatmap(
#             z=cm_data,
#             colorscale='Blues'
#         )
        
#         st.plotly_chart(fig, use_container_width=True)
#     except ImportError:
#         st.warning("Plotly not installed for visualization.")


# def render_settings_panel() -> Dict[str, Any]:
#     """
#     Render settings configuration panel.
    
#     Returns:
#         Dictionary of selected settings
#     """
#     with st.expander("Advanced Settings"):
#         settings = {}
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             settings['batch_size'] = st.number_input(
#                 "Batch Size",
#                 min_value=1,
#                 max_value=512,
#                 value=32
#             )
            
#             settings['max_length'] = st.number_input(
#                 "Max Sequence Length",
#                 min_value=64,
#                 max_value=512,
#                 value=512
#             )
        
#         with col2:
#             settings['confidence_threshold'] = st.slider(
#                 "Confidence Threshold",
#                 min_value=0.0,
#                 max_value=1.0,
#                 value=0.5
#             )
            
#             settings['use_gpu'] = st.checkbox(
#                 "Use GPU (if available)",
#                 value=True
#             )
        
#         settings['output_format'] = st.selectbox(
#             "Output Format",
#             options=["JSON", "CSV", "XML"]
#         )
    
#     return settings


# def render_error_message(error_type: str, message: str) -> None:
#     """
#     Render error message with styling.
    
#     Args:
#         error_type: Type of error (warning, error, info)
#         message: Error message text
#     """
#     if error_type == "error":
#         st.error(f"❌ {message}")
#     elif error_type == "warning":
#         st.warning(f"{message}")
#     else:
#         st.info(f"{message}")


# def render_progress_steps(total_steps: int, current_step: int, 
#                          step_labels: List[str]) -> None:
#     """
#     Render progress indicator.
    
#     Args:
#         total_steps: Total number of steps
#         current_step: Current step number
#         step_labels: List of step labels
#     """
#     progress = current_step / total_steps
#     st.progress(progress)
    
#     # Display step information
#     col1, col2 = st.columns([1, 3])
#     with col1:
#         st.text(f"{current_step}/{total_steps}")
#     with col2:
#         st.text(step_labels[current_step - 1] if current_step <= len(step_labels) else "")
