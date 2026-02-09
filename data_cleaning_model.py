import pandas as pd
from datetime import datetime

today = datetime.today().date()

# Policy Data Function to read the policy file
def read_policy_data(uploaded_file) -> pd.DataFrame:
   
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith((".xlsx", ".xls")):
        # Read all sheets
        sheets = pd.read_excel(uploaded_file, sheet_name=None)
        if isinstance(sheets, dict):
            df = pd.concat(
                [df.assign(source_sheet=sheet_name) for sheet_name, sheets_df in sheets.items()],
                ignore_index=True)
        else:
            df = next(iter(sheets.values()))

    elif filename.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    else:
        raise ValueError(
            "Unsupported file type. Please upload CSV, Excel, or Parquet."
        )
    return df

# Revenue Function for reading the revenue data
def read_revenue_data(uploaded_file) -> pd.DataFrame:

    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        df_rev = pd.read_csv(uploaded_file)

    elif filename.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(uploaded_file, sheet_name=None)
        # Consolidate multiple sheets if present
        if isinstance(sheets, dict):
            df_rev = pd.concat(
                [df.assign(source_sheet=sheet_name) for sheet_name, df in sheets.items()],
                 ignore_index=True
                 )
        else:
            df_rev = sheets
    elif filename.endswith(".parquet"):
        df_rev = pd.read_parquet(uploaded_file)
    else:
        raise ValueError(
            "Unsupported file type. Please upload CSV, Excel, or Parquet."
        )
    return df_rev

# Function to read Claims File
def read_claims_data(uploaded_file) -> pd.DataFrame:
    """
    Read claims data from a Streamlit UploadedFile.
    Supports CSV, Excel (single or multiple sheets), and Parquet.
    Always returns a single pandas DataFrame.
    """

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        claims = pd.read_csv(uploaded_file)

    elif filename.endswith((".xlsx", ".xls")):
        claims = pd.read_excel(uploaded_file)

    elif filename.endswith(".parquet"):
        claims = pd.read_parquet(uploaded_file)

    else:
        raise ValueError(
            "Unsupported file type. Please upload CSV, Excel, or Parquet."
        )
    return claims

# Policy Data Function to standardize columns
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "", regex=True)
        .str.lower()
    )
    df = df[
        [
            "nationalid", "firstname", "surname", "date(pm.dateofbirth)",
            "packagename", "gender", "membertype", "relationshipid",
            "groupcreationdate", "date(pm.creationdate)", "stewardname",
            "merchantcode", "status", "stewardid", "policynumber",
            "memberstatus", "currencyname", "source_sheet",
        ]
    ]
    # clean national id column
    df["nationalid"] = (
        df["nationalid"]
        .astype(str)
        .str.strip()
        .str.replace(r"[-/\s]", "", regex=True)
        .str.upper()
    )
    # clean policynumber column
    df["policynumber"] = (
        df["policynumber"]
        .astype(str)
        .str.strip()
        .str.replace(r"[-/\s]", "", regex=True)
        .str.upper()
    )
    # create a full name column and insert it next to the surname column
    df["firstname"] = df["firstname"].astype("string").str.lower()
    df["surname"] = df["surname"].astype("string").str.lower()
    df["full_name"] = (
        df["firstname"].str.strip() + " " + df["surname"].str.strip()
    )
    cols = df.columns.tolist()
    cols.remove("full_name")
    idx = df.columns.get_loc("surname")
    cols.insert(idx + 1, "full_name")
    df = df[cols]

    # convert date columns to datetime
    date_cols = [
        "date(pm.dateofbirth)",
        "groupcreationdate",
        "date(pm.creationdate)",
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# Policy Data Function to Clean the Policy Data
def clean_policy_data(df: pd.DataFrame):
    outputs = {}
    df.rename(columns={"source_sheet": "month"}, inplace=True)
    # Remove inactive members
    df = df[
        (~df["memberstatus"].isin(["DELETED", "SUSPENDED", "DECEASED", "POLICY CLOSED"])) &
        (~df["status"].isin(["CLOSED"]))
    ]
    # Complete duplicates where all entries under every column are repeated
    duplicates = df.duplicated(keep=False)
    if duplicates.any():
        outputs["complete_duplicates"] = df[duplicates]
        df = df[~duplicates]
    # Date invalidness
    cond1 = df["date(pm.dateofbirth)"] < pd.Timestamp("1900-01-01")
    cond2 = df["date(pm.dateofbirth)"] > pd.to_datetime(today)
    cond3 = df["groupcreationdate"] < df["date(pm.creationdate)"]
    condition = cond1 | cond2
    if condition.any():
        outputs["invalid_dates_of_birth"] = df[condition]
        df = df[~condition]
    if cond3.any():
        outputs["group_creation_date_less_than_policy_creation_date"] = df[cond3]
        df = df[~cond3]
    # Names with numbers
    mask = df["full_name"].astype(str).str.contains(r"\d", na=False)
    if mask.any():
        outputs["names_with_numerics"] = df[mask]
        df = df[~mask]
    # Insert the Premium column
    package_premiums = pd.DataFrame(
        {"packagename": df["packagename"].unique()}
    )
    product_mapping = {
        "Ecosure Lite": 0.75,
        "Ecosure Basic": 1.5,
        "Ecosure Standard": 3,
        "Ecosure Premium": 7.5,
        "Ecosure Africa": 7.5,
        "Ecosure Global": 15,
    }
    package_premiums["premium"] = (
        package_premiums["packagename"]
        .map(product_mapping)
        .fillna(3)
    )
    df["premium"] = df["packagename"].map(
        package_premiums.set_index("packagename")["premium"]
    )
    df.reset_index(drop=True, inplace=True)
    return df, outputs

# Revenue function to wrangle the revenue file
def wrangle_revenue(revenue: pd.DataFrame):
    outputs = {}
    # Drop and rename columns
    revenue.drop(columns=[' TOTAL BILLED ', ' TOTAL COLLECTED '], inplace=True)
    revenue.rename(columns={"source_sheet": "month"}, inplace=True)
    revenue.columns = revenue.columns.str.strip().str.replace(" ", "_", regex=True).str.lower()    
    # Clean region and steward_id
    revenue["region"] = revenue["region"].astype(str).str.strip()
    revenue["region"] = revenue["region"].str.replace("Mash West", "Mashonaland West", regex=True)
    revenue["steward_id"] = revenue["steward_id"].astype(int)
    # Fill missing totals
    revenue[["total_billed", "total_collected"]] = revenue[["total_billed", "total_collected"]].fillna(0)
    # Flag and remove active members with decimals / non-numeric values
    cols_to_check = ["total_billed", "total_collected", "active_members"]
    for col in cols_to_check:
        revenue[col] = pd.to_numeric(revenue[col], errors="coerce")
    non_numeric_mask = revenue[cols_to_check].isna().any(axis=1)
    if non_numeric_mask.any():
        outputs["non_numeric_entries"] = revenue[non_numeric_mask]
        revenue = revenue[~non_numeric_mask]
    decimal_mask = revenue["active_members"] % 1 != 0
    if decimal_mask.any():
        outputs["decimal_active_members"] = revenue[decimal_mask]
        revenue = revenue[~decimal_mask]
    # Complete duplicates
    complete_duplicate_mask = revenue.duplicated(keep=False)
    if complete_duplicate_mask.any():
        outputs["complete_duplicates"] = revenue[complete_duplicate_mask]
        revenue = revenue[~complete_duplicate_mask]
    return revenue.reset_index(drop=True), outputs

# Claims Data Function to Wrangle the Claims Data
def wrangle_claims(claims: pd.DataFrame):
    outputs = {}
    # Standardize column names
    claims.columns = claims.columns.str.strip().str.replace(" ", "_", regex=True).str.lower()
    # Keep only relevant columns
    claims = claims[[
        'date_submitted', 'date_of_reg', 'date_of_birth', 'date_of_death',
        'age_at_death', 'policy_type', 'policy_id',
        'deceased_no;_group_name;_ssr_no', 'name_of_deceased', 'id_number',
        'gender', 'package', 'sum_assured', 'cause_of_death',
        'cause_of_death_classification', 'place_of_death', 'place_burial',
        'submitting_agent', 'region', 'fsp_used', 'claimant_mobile_number',
        'witness_cell_number_1', 'documents_provided', 'email_date',
        'beneficiary_name','beneficiary_number',
        'account_number', 'amount_paid', 'date_of_payout', 'bank'
    ]]
    # Clean identifiers and names
    claims["id_number"] = (
        claims["id_number"].astype(str)
        .str.strip()
        .str.replace(r"[-/\s]", "", regex=True)
        .str.upper()
    )
    claims["policy_id"] = (
        claims["policy_id"].astype(str)
        .str.strip()
        .str.replace(r"[-/\s]", "", regex=True)
        .str.upper()
    )
    claims["name_of_deceased"] = (
        claims["name_of_deceased"].astype(str)
        .str.strip()
        .str.lower()
    )
    # Convert date columns to datetime
    date_columns = ['date_submitted', 'date_of_reg', 'date_of_birth', 'date_of_death', 'date_of_payout']
    for col in date_columns:
        claims[col] = pd.to_datetime(claims[col], errors="coerce")
    # Complete duplicate entries
    dup_entries = claims.duplicated(keep=False)
    if dup_entries.any():
        outputs["complete_duplicated_claims"] = claims[dup_entries]
        claims = claims[~dup_entries]
    # Names with numbers
    num_names_mask = claims["name_of_deceased"].astype(str).str.contains(r"\d", na=False)
    if num_names_mask.any():
        outputs["claim_names_with_numbers"] = claims[num_names_mask]
        claims = claims[~num_names_mask]
    # Age at Death mismatches
    expected_age = ((claims["date_of_death"] - claims["date_of_birth"]).dt.days // 365)
    age_at_death_mask = claims["age_at_death"] != expected_age
    if age_at_death_mask.any():
        outputs["age_at_death_mismatches"] = claims[age_at_death_mask]
        claims = claims[~age_at_death_mask]
    # Incorrect order of dates
    cond_1 = claims["date_of_birth"] < pd.Timestamp("1900-01-01")
    cond_2 = claims["date_of_birth"] > pd.to_datetime(today)
    cond_3 = claims["date_of_birth"] > claims["date_of_reg"]
    cond_4 = claims["date_of_death"] > claims["date_submitted"]
    cond_5 = claims["date_of_reg"] > claims["date_of_death"]
    cond_6 = claims["date_submitted"] > claims["date_of_payout"]
    birth_cond = cond_1 | cond_2
    if birth_cond.any():
        outputs["invalid_dates_of_births"] = claims[birth_cond]
        claims = claims[~birth_cond]
    conditions = [
        (cond_3, "date_of_birth_greater_than_date_of_reg"),
        (cond_4, "date_of_death_greater_than_date_submitted"),
        (cond_5, "date_of_reg_greater_than_date_of_death"),
        (cond_6, "date_submitted_greater_than_date_of_payout")
    ]
    for cond, name in conditions:
        if cond.any():
            outputs[name] = claims[cond]
            claims = claims[~cond]
    # Overpaid / underpaid claims
    over_paid_claims = claims["amount_paid"] > pd.to_numeric(claims["sum_assured"])
    under_paid_claims = claims["amount_paid"] < pd.to_numeric(claims["sum_assured"])
    payments = under_paid_claims | over_paid_claims
    if payments.any():
        outputs["inconsistent_payments"] = claims[payments]
        claims = claims[~payments]
    return claims.reset_index(drop=True), outputs

# Revenue File function to map policy data
def map_policy_to_rev(df: pd.DataFrame, rev_df: pd.DataFrame):
    outputs = {}
    rev_stewards = set(rev_df["steward_id"])
    pol_in_rev = df[df["stewardid"].isin(rev_stewards)].reset_index(drop=True)

    pol_not_in_rev = df[~df["stewardid"].isin(rev_stewards)].reset_index(drop=True)
    # Collect the mapped policies for web app handling instead of saving
    if not pol_not_in_rev.empty:
        outputs["policies_not_available_in_revenue"] = pol_not_in_rev
    return pol_in_rev, outputs

# Map 'region' from revenue data to policy data based on steward_id
def map_region_and_group(df: pd.DataFrame, rev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map region and group onto policy data using stewardid.
    Returns ONLY a DataFrame.
    """

    df = df.copy()

    # Build lookup from revenue data
    steward_lookup = (
        rev_df
        .dropna(subset=["steward_id"])
        .drop_duplicates("steward_id")
        .set_index("steward_id")[["region", "name_of_group"]]
    )

    # Map region and group
    df["region"] = df["stewardid"].map(steward_lookup["region"])
    df["name_of_group"] = df["stewardid"].map(steward_lookup["name_of_group"])

    return df

# Claims Data Function to map Claims to the POlicy data
def map_claims(claims: pd.DataFrame, df: pd.DataFrame):
    outputs = {}
    # Mask for IDs in claims that are in df["nationalid"]
    avail_in_pol_using_nat_id_mask = claims["id_number"].isin(df["nationalid"])
    avail_in_pol_using_nat_id = pd.DataFrame()
    if avail_in_pol_using_nat_id_mask.any():
        avail_in_pol_using_nat_id = claims[avail_in_pol_using_nat_id_mask].reset_index(drop=True)
        outputs["claims_avail_in_policy_using_nat_id"] = avail_in_pol_using_nat_id
    # Mask for IDs in claims_df that are NOT in df["nationalid"]
    not_avail_in_pol_using_nat_id_mask = ~claims["id_number"].isin(df["nationalid"])
    not_avail_in_pol_using_nat_id = pd.DataFrame()
    if not_avail_in_pol_using_nat_id_mask.any():
        not_avail_in_pol_using_nat_id = claims[not_avail_in_pol_using_nat_id_mask].reset_index(drop=True)
        outputs["claims_not_avail_in_policy_using_nat_id"] = not_avail_in_pol_using_nat_id
    # Mask for Policy IDs in not_avail_in_pol_using_nat_id that are in df["policynumber"]
    avail_in_pol_using_pol_id_mask = not_avail_in_pol_using_nat_id["policy_id"].isin(df["policynumber"])
    avail_in_pol_using_pol_id = pd.DataFrame()
    if avail_in_pol_using_pol_id_mask.any():
        avail_in_pol_using_pol_id = not_avail_in_pol_using_nat_id[avail_in_pol_using_pol_id_mask].reset_index(drop=True)
        outputs["claims_avail_in_policy_using_pol_id"] = avail_in_pol_using_pol_id
    # Mask for Policy IDs in not_avail_in_pol_using_nat_id that are NOT in df["policynumber"]
    not_avail_in_pol_using_pol_id_mask = ~not_avail_in_pol_using_nat_id["policy_id"].isin(df["policynumber"])
    not_avail_in_pol_using_pol_id = pd.DataFrame()
    if not_avail_in_pol_using_pol_id_mask.any():
        not_avail_in_pol_using_pol_id = not_avail_in_pol_using_nat_id[not_avail_in_pol_using_pol_id_mask].reset_index(drop=True)
        outputs["claims_not_avail_in_policy_using_pol_id"] = not_avail_in_pol_using_pol_id
    # Mask for Name of Deceased in not_avail_in_pol_using_pol_id that are in df["full_name"]
    avail_in_pol_using_name_mask = not_avail_in_pol_using_pol_id["name_of_deceased"].isin(df["full_name"])
    avail_in_pol_using_name = pd.DataFrame()
    if avail_in_pol_using_name_mask.any():
        avail_in_pol_using_name = not_avail_in_pol_using_pol_id[avail_in_pol_using_name_mask].reset_index(drop=True)
        outputs["claims_avail_in_policy_using_name"] = avail_in_pol_using_name
    # Mask for Name of Deceased in not_avail_in_pol_using_pol_id that are NOT in df["full_name"]
    not_avail_in_pol_using_name_mask = ~not_avail_in_pol_using_pol_id["name_of_deceased"].isin(df["full_name"])
    not_avail_in_pol_using_name = pd.DataFrame()
    if not_avail_in_pol_using_name_mask.any():
        not_avail_in_pol_using_name = not_avail_in_pol_using_pol_id[not_avail_in_pol_using_name_mask].reset_index(drop=True)
        outputs["claims_not_avail_in_policy_using_name"] = not_avail_in_pol_using_name
    # Remove completely unmatched names from claims_df
    claims = claims[
        ~claims["name_of_deceased"].isin(not_avail_in_pol_using_name["name_of_deceased"])
    ].reset_index(drop=True)
    return (
        claims,
        outputs,
        avail_in_pol_using_nat_id,
        not_avail_in_pol_using_nat_id,
        avail_in_pol_using_pol_id,
        not_avail_in_pol_using_pol_id,
        avail_in_pol_using_name,
        not_avail_in_pol_using_name
    )

def map_policy_info_to_claims(claims: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing stewardid, name_of_group, and region in claims
    using policy data via nationalid, policynumber, and full_name.
    """

    claims_df = claims.copy()
    lookup_cols = ["stewardid", "name_of_group", "region"]
    for col in lookup_cols:
        if col not in claims_df.columns:
            claims_df[col] = pd.NA
    
    # First Lookup by nationalid 
    nat_lookup = (
        policy_df
        .drop_duplicates("nationalid")
        .set_index("nationalid")[lookup_cols]
    )

    mask = claims_df["stewardid"].isna()
    claims_df.loc[mask, lookup_cols] = (
        claims_df.loc[mask, "id_number"]
        .map(nat_lookup.to_dict(orient="index"))
        .apply(pd.Series)
    )

    # Those not found, Lookup by policynumber 
    policy_lookup = (
        policy_df
        .drop_duplicates("policynumber")
        .set_index("policynumber")[lookup_cols]
    )

    mask = claims_df["stewardid"].isna()
    claims_df.loc[mask, lookup_cols] = (
        claims_df.loc[mask, "policy_id"]
        .map(policy_lookup.to_dict(orient="index"))
        .apply(pd.Series)
    )

    # Ultimately Lookup by full_name
    name_lookup = (
        policy_df
        .drop_duplicates("full_name")
        .set_index("full_name")[lookup_cols]
    )

    mask = claims_df["stewardid"].isna()
    claims_df.loc[mask, lookup_cols] = (
        claims_df.loc[mask, "name_of_deceased"]
        .map(name_lookup.to_dict(orient="index"))
        .apply(pd.Series)
    )

    return claims_df

def run_data_cleaning(policy_file: str, revenue_file: str, claims_file: str) -> dict:

    # Read files
    policy_df = read_policy_data(policy_file)
    rev_df = read_revenue_data(revenue_file)
    claims_df = read_claims_data(claims_file)

    # Clean and wrangle 
    policy_df = standardize_columns(policy_df)
    policy_df, policy_outputs = clean_policy_data(policy_df)

    rev_df, rev_outputs = wrangle_revenue(rev_df)
    claims_df, claims_outputs = wrangle_claims(claims_df)
    # Map to revenue
    policy_df, policy_mapping_outputs = map_policy_to_rev(policy_df, rev_df)

    # Map region and group
    policy_df = map_region_and_group(policy_df, rev_df)

    # Map claims to policy 
    claims_df, mapping_outputs, *_ = map_claims(claims_df, policy_df)

    # Fill missing info in claims from policy
    claims_df = map_policy_info_to_claims(claims_df, policy_df)

    # Return all results 
    return {
        "policy_df": policy_df,
        "rev_df": rev_df,
        "claims_df": claims_df,
        "policy_outputs": policy_outputs,
        "rev_outputs": rev_outputs,
        "claims_outputs": claims_outputs,
        "mapping_outputs": {
            **mapping_outputs,
            **policy_mapping_outputs
        }
    }


















