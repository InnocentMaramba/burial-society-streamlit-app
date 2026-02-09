import streamlit as st
import data_cleaning_model as dc
import zipfile
import io

# App Title 
st.title("Econet Life Burial Society Data Validation Model")
st.markdown(
    """
    **Upload your :rainbow[Policy File], :rainbow[Revenue File], and :rainbow[Claims File]
    to clean and validate the data. The model will process the files and provide cleaned datasets along with flagged outputs for any discrepancies found.**
    """
)
# Upload Files
policy_file = st.file_uploader("Upload Policy File", type=["csv","xlsx","parquet"])
revenue_file = st.file_uploader("Upload Revenue File", type=["csv","xlsx","parquet"])
claims_file = st.file_uploader("Upload Claims File", type=["csv","xlsx","parquet"])

if policy_file and revenue_file and claims_file:

    st.info("Processing data...")


    # Run Full Workflow 
    results = dc.run_data_cleaning(policy_file, revenue_file, claims_file)

    # Get Cleaned DataFrames 
    policy_df = results["policy_df"]
    rev_df = results["rev_df"]
    claims_df = results["claims_df"]

    st.success("Data cleaning complete!")

    # Preview Cleaned Data 
    st.subheader("Cleaned Policy Data")
    st.dataframe(policy_df.head())
    col1, col2 = st.columns(2)
    col1.metric("Total rows in the Cleaned Policy Data:", policy_df.shape[0])
    col2.metric("Total columns in the Cleaned Policy Data:", policy_df.shape[1])

    st.subheader("Cleaned Revenue Data")
    st.dataframe(rev_df.head())
    col1, col2 = st.columns(2)
    col1.metric("Total rows in the Cleaned Revenue Data:", rev_df.shape[0])
    col2.metric("Total columns in the Cleaned Revenue Data:", rev_df.shape[1])

    st.subheader("Cleaned Claims Data")
    st.dataframe(claims_df.head())
    col1, col2 = st.columns(2)
    col1.metric("Total rows in the Cleaned Claims Data:", claims_df.shape[0])
    col2.metric("Total columns in the Cleaned Claims Data:", claims_df.shape[1])

    #  Download Cleaned Data
    st.subheader("Download Cleaned Data")
    for name, df in {
        "policy_cleaned": policy_df,
        "revenue_cleaned": rev_df,
        "claims_cleaned": claims_df
    }.items():
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {name}",
            data=csv,
            file_name=f"{name}.csv",
            mime='text/csv'
        )

    # Download Flagged Outputs
    st.subheader("Download Flagged Outputs")

    # Revenue flagged zipped outputs
    for name, df_flag in results["rev_outputs"].items():
        csv = df_flag.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download Revenue Flagged: {name}",
            data=csv,
            file_name=f"revenue_flagged_{name}.csv",
            mime='text/csv'
        )

    # Claims flagged outputs
    for name, df_flag in results["claims_outputs"].items():
        csv = df_flag.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download Claims Flagged: {name}",
            data=csv,
            file_name=f"claims_flagged_{name}.csv",
            mime='text/csv'
        )

    # Mapping flagged outputs
    for name, df_flag in results["mapping_outputs"].items():
        csv = df_flag.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download Mapping Flagged: {name}",
            data=csv,
            file_name=f"mapping_flagged_{name}.csv",
            mime='text/csv'
        )
    # Policy flagged outputs
    for name, df_flag in results["policy_outputs"].items():
        csv = df_flag.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download Policy Flagged: {name}",
            data=csv,
            file_name=f"policy_flagged_{name}.csv",
            mime='text/csv'
        )

    for name, df_flag in results["mapping_outputs"].items():
        if "policy" in name or "revenue" in name:
            csv = df_flag.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download Policy to Revenue Mapping Flagged: {name}",
                data=csv,
                file_name=f"policy_to_revenue_mapping_flagged_{name}.csv",
                mime="text/csv"
            )
