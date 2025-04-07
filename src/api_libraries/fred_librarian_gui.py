import streamlit as st
import pandas as pd
import json
import requests
import os
import numpy as np
import hashlib
import faiss
from sentence_transformers import SentenceTransformer

# CONFIGURATION
FRED_API_KEY = "dbe0b0309bf7116c11f595d3e1dffd53"  # Replace if you’ve got your own key
CATALOG_FILE = "fred_catalog.json"
EMBEDDINGS_CACHE = "fred_title_embeddings.npy"
TITLES_CACHE = "fred_titles.npy"
TITLES_HASH_FILE = "fred_title_hash.txt"

# If the catalog doesn't exist (like on Streamlit Cloud), let the user upload it
if not os.path.exists(CATALOG_FILE):
    st.warning("Please upload your `fred_catalog.json` file.")
    uploaded_file = st.file_uploader("Upload fred_catalog.json", type="json")
    if uploaded_file is not None:
        with open(CATALOG_FILE, "wb") as f:
            f.write(uploaded_file.read())
        st.experimental_rerun()
    st.stop()
# Load the catalog
with open(CATALOG_FILE, 'r') as f:
    catalog = json.load(f)
series_data = catalog["series"]
series_list = list(series_data.values())
titles = [s['title'] for s in series_list]
series_ids = [s['id'] for s in series_list]

# Hash titles for caching
def hash_titles(titles):
    return hashlib.sha256("".join(titles).encode('utf-8')).hexdigest()

# Load or generate embeddings
@st.cache_resource(show_spinner=True)
def get_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(TITLES_CACHE):
        embeddings = np.load(EMBEDDINGS_CACHE)
        titles_loaded = np.load(TITLES_CACHE, allow_pickle=True)
        if list(titles_loaded) == titles:
            return model, embeddings
    
    st.warning("Generating new embeddings... Hang tight, this takes a sec.")
    embeddings = model.encode(titles, show_progress_bar=True, batch_size=128)
    np.save(EMBEDDINGS_CACHE, embeddings)
    np.save(TITLES_CACHE, np.array(titles))
    return model, embeddings

model, embeddings = get_embeddings()

# Build FAISS index for searching
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Cached function to fetch series data for downloads
@st.cache_data
def fetch_series_data(sid):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": sid,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": "1950-01-01"
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        obs = r.json().get("observations", [])
        if obs:
            data_df = pd.DataFrame(obs)
            data_df['date'] = pd.to_datetime(data_df['date'])
            data_df.set_index('date', inplace=True)
            data_df[sid] = pd.to_numeric(data_df['value'], errors='coerce')
            return data_df[[sid]].to_csv(index=True).encode('utf-8')
        else:
            return b"No data found for this series."
    else:
        return f"Failed to fetch data. HTTP {r.status_code}".encode('utf-8')

# UI Shit
st.title("📚 FRED Series Librarian")
st.write("Search the whole damn FRED catalog with natural language. Preview or download whatever you want.")

# Sidebar for filters and settings
with st.sidebar:
    st.header("⚙️ Settings & Filters")
    
    num_results = st.slider("Number of results", 5, 50, 10)
    
    with st.expander("Advanced Filters"):
        frequency_filter = st.multiselect(
            "Data Frequency",
            options=["Daily", "Weekly", "Monthly", "Quarterly", "Annual", "Semiannual"],
            default=[]
        )
        units_filter = st.multiselect(
            "Units",
            options=["Percent", "Index", "Billions of Dollars", "Millions", "Thousands"],
            default=[]
        )
        popularity_threshold = st.slider("Minimum Popularity", 0, 100, 0)
    
    with st.expander("Recent Searches", expanded=False):
        if "recent_searches" not in st.session_state:
            st.session_state.recent_searches = []
        
        for prev_query in st.session_state.recent_searches:
            if st.button(f"🔍 {prev_query}", key=f"prev_{prev_query}"):
                st.session_state.query = prev_query
                st.experimental_rerun()

# Main layout
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("🔎 Enter your query:", "natural rate of unemployment", key="query")
with col2:
    st.write("")
    st.write("")
    if st.button("🧹 Clear", key="clear_search"):
        st.session_state.query = ""
        st.experimental_rerun()

if query:
    # Save recent searches
    if "recent_searches" not in st.session_state:
        st.session_state.recent_searches = []
    
    if query not in st.session_state.recent_searches:
        st.session_state.recent_searches.insert(0, query)
        st.session_state.recent_searches = st.session_state.recent_searches[:5]
    
    # Search the catalog
    with st.spinner("Searching FRED catalog..."):
        query_embedding = model.encode([query])
        D, I = index.search(np.array(query_embedding), min(100, len(titles)))
        
        results = []
        for idx in I[0]:
            match = series_list[idx]
            
            frequency = match.get("frequency", "")
            units = match.get("units", "")
            popularity = match.get("popularity", 0)
            
            if (frequency_filter and frequency not in frequency_filter) or \
               (units_filter and units not in units_filter) or \
               (popularity < popularity_threshold):
                continue
            
            results.append({
                "Series ID": match["id"],
                "Title": match["title"],
                "Frequency": frequency,
                "Units": units,
                "Start": match.get("start_date", ""),
                "End": match.get("end_date", ""),
                "Popularity": popularity
            })
            
            if len(results) >= num_results:
                break
    
    # Show results
    if results:
        df = pd.DataFrame(results)
        
        with st.expander("🔍 Search Info", expanded=False):
            st.write(f"Found {len(results)} matching series for '{query}'")
            st.write(f"Applied Filters: {', '.join(frequency_filter + units_filter) or 'None'}")
            st.write(f"Minimum Popularity: {popularity_threshold}")
            if st.button("📋 Copy Query to Clipboard"):
                st.code(query)
                st.success("Query copied to clipboard! (Use Ctrl+C)")
        
        st.dataframe(df, use_container_width=True, height=400)
        
        # Buttons for each series
        for i, row in df.iterrows():
            sid = row["Series ID"]
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button(f"👁️ Preview {sid}", key=f"preview_{sid}"):
                    with st.expander("Preview Options", expanded=True):
                        col_start, col_end = st.columns(2)
                        with col_start:
                            start_date = st.date_input("Start Date", 
                                                      value=pd.to_datetime("1950-01-01"),
                                                      key=f"start_{sid}")
                        with col_end:
                            end_date = st.date_input("End Date", 
                                                    value=pd.Timestamp.today(),
                                                    key=f"end_{sid}")
                        
                        transform = st.selectbox(
                            "Data Transformation",
                            options=["None", "Percent Change", "Percent Change from Year Ago", 
                                    "12-Month Moving Average", "Natural Log"],
                            key=f"transform_{sid}"
                        )
                    
                    url = "https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        "series_id": sid,
                        "api_key": FRED_API_KEY,
                        "file_type": "json",
                        "observation_start": start_date.strftime("%Y-%m-%d"),
                        "observation_end": end_date.strftime("%Y-%m-%d")
                    }
                    
                    if transform == "Percent Change":
                        params["units"] = "pc1"
                    elif transform == "Percent Change from Year Ago":
                        params["units"] = "pc12"
                    
                    with st.spinner(f"Loading data for {sid}..."):
                        r = requests.get(url, params=params)
                        if r.status_code == 200:
                            obs = r.json().get("observations", [])
                            if obs:
                                data_df = pd.DataFrame(obs)
                                data_df['date'] = pd.to_datetime(data_df['date'])
                                data_df.set_index('date', inplace=True)
                                data_df[sid] = pd.to_numeric(data_df['value'], errors='coerce')
                                
                                if transform == "12-Month Moving Average":
                                    data_df[f"{sid}_raw"] = data_df[sid].copy()
                                    data_df[sid] = data_df[sid].rolling(window=12).mean()
                                    data_df.dropna(subset=[sid], inplace=True)
                                    transform_note = "Data shown as 12-month moving average"
                                elif transform == "Natural Log":
                                    data_df[f"{sid}_raw"] = data_df[sid].copy()
                                    data_df[sid] = np.log(data_df[sid])
                                    data_df.dropna(subset=[sid], inplace=True)
                                    transform_note = "Data shown in natural log form"
                                else:
                                    transform_note = f"Transformation: {transform}" if transform != "None" else "No transformation applied"
                                
                                st.session_state[f"data_{sid}"] = data_df
                                st.session_state[f"transform_note_{sid}"] = transform_note
                                
                                st.subheader(f"{row['Title']} ({sid})")
                                st.caption(st.session_state[f"transform_note_{sid}"])
                                
                                if data_df[sid].isnull().all():
                                    st.warning("No valid data points found for this range and transformation.")
                                else:
                                    st.line_chart(data_df[[sid]])
                                    
                                    col_stats1, col_stats2 = st.columns(2)
                                    with col_stats1:
                                        st.write("**Basic Statistics:**")
                                        st.dataframe(data_df[[sid]].describe().round(2), use_container_width=True)
                                    with col_stats2:
                                        st.write("**Most Recent Values:**")
                                        st.dataframe(data_df[[sid]].tail(), use_container_width=True)
                                    
                                    st.write("**Export Options:**")
                                    export_col1, export_col2, export_col3 = st.columns(3)
                                    with export_col1:
                                        if st.button("📊 Export to CSV", key=f"csv_{sid}"):
                                            data_df[[sid]].to_csv(f"{sid}.csv")
                                            st.success(f"✅ Saved {sid}.csv")
                                    with export_col2:
                                        if st.button("📑 Export to Excel", key=f"excel_{sid}"):
                                            data_df[[sid]].to_excel(f"{sid}.xlsx")
                                            st.success(f"✅ Saved {sid}.xlsx")
                                    with export_col3:
                                        if st.button("📋 Copy Data", key=f"copy_{sid}"):
                                            st.code(data_df[[sid]].to_csv())
                                            st.success("Data copied to clipboard! (Use Ctrl+C)")
                            else:
                                st.error("No data found for this series.")
                        else:
                            st.error(f"Failed to fetch data for {sid}. HTTP {r.status_code}")
            
            with col2:
                csv_data = fetch_series_data(sid)
                st.download_button(
                    label=f"📥 Download {sid}",
                    data=csv_data,
                    file_name=f"{sid}.csv",
                    mime="text/csv",
                    key=f"btn_{sid}"
                )
    else:
        st.warning("No results found. Try a different query or fuck with the filters.")