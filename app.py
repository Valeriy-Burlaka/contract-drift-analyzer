import pandas as pd
import streamlit as st

from analysis import run_analysis

def app():
    st.set_page_config(page_title="Contract Drift Analysis", layout="wide")
    st.title("Contract Drift Analysis")

    col1, col2 = st.columns([1, 3])

    # Left column for controls
    with col1:
        st.header("Controls")

        document_id = st.text_input("Document ID", placeholder="Enter document ID")

        model = st.selectbox(
            "Model",
            ["gemini-1.5-pro-002", "gemini-2.0-flash-001"],
            index=0
        )

        # Parameter inputs
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.01)
        top_p = st.slider("Top P", 0.0, 1.0, 0.95, 0.01)
        seed = st.number_input("Seed", value=10042, min_value=0)

        run_button = st.button("Run", type="primary", use_container_width=True)

    # Right column for results
    with col2:
        st.header("Results")

        if run_button:
            if not document_id:
                st.error("Please enter a Document ID")
            else:
                with st.spinner("Analyzing document..."):
                    results, metadata = run_analysis(document_id, model, seed, temperature, top_p)

                if results:
                    column_mapping = {
                        'contractor_name': 'Supplier',
                        'document_type': 'Document Type',
                        'section_name': 'Section',
                        'body': 'Change Summary',
                        'impact': 'Impact Level'
                    }
                    column_order = ['Supplier', 'Document Type', 'Section', 'Change Summary', 'Impact Level']

                    df = pd.DataFrame(results)
                    df = df.rename(columns=column_mapping)
                    df = df[column_order]

                    st.table(df)

                    st.header("Usage Metadata")
                    st.write(metadata)
                else:
                    st.info("No results found.")

if __name__ == "__main__":
    app()
