# ==============================================
# PROFESSIONAL CUSTOMER SEGMENTATION DASHBOARD
# ==============================================
"""
Ultimate Production-Ready Features:
- Professional dashboard UI (Banking-style)
- Efficient caching with @st.cache_data
- Modular architecture with 15+ functions
- Multiple clustering algorithms (KMeans, Hierarchical, DBSCAN)
- Advanced metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Real-time analytics dashboard
- Interactive visualizations (Plotly)
- Excel/CSV export with multi-sheet reports
- RFM Analysis for transactional data
- Optimized memory management
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import traceback
import logging

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# =====================================================
# ERROR HANDLING SETUP
# =====================================================

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_execute(func, *args, **kwargs):
    """
    Safely execute a function and return None if it fails
    Logs the error instead of crashing
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_msg = f"Function '{func.__name__}' failed: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return None

def safe_plot(plot_func, *args, **kwargs):
    """
    Safely create a plot and show error message if it fails
    """
    try:
        return plot_func(*args, **kwargs)
    except Exception as e:
        error_msg = f"Visualization failed: {str(e)}"
        logger.error(error_msg)
        st.warning(f"⚠️ {error_msg}")
        return None

# ==============================================
# PAGE CONFIGURATION
# ==============================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================
# PROFESSIONAL STYLING (Banking Dashboard Style)
# ==============================================

st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling - Dark Blue Theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2c4a6f 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stSelectbox > label,
    [data-testid="stSidebar"] .stSlider > label {
        color: white !important;
        font-weight: 500;
    }
    
    /* Main Content Background */
    .main {
        background-color: #f5f7fa;
    }
    
    /* Dashboard Header */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .dashboard-subtitle {
        font-size: 1.1rem;
        color: #e0e7ff;
        margin-top: 0.5rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e3a5f;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stMetric"] {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #667eea;
    }
    
    /* Cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .stat-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .stat-card-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
    }
    
    .stat-card-detail {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #64748b;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f1f5f9;
        color: #1e3a5f;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 8px;
        font-weight: 600;
        color: #1e3a5f;
        padding: 1rem;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* DataFrame */
    .dataframe {
        border-radius: 8px !important;
    }
    
    /* Selectbox and other inputs */
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #94a3b8;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================
# UTILITY FUNCTIONS (WITH CACHING)
# ==============================================

@st.cache_data(ttl=3600)
def load_data(file):
    """Load CSV with multiple encoding attempts and caching"""
    try:
        df = pd.read_csv(file, encoding='utf-8')
        return df, "UTF-8"
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file, encoding='latin1')
            return df, "Latin-1"
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file, encoding='iso-8859-1')
                return df, "ISO-8859-1"
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None, None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None

@st.cache_data
def generate_sample_data(n_samples=200):
    """Generate realistic sample data for demonstration"""
    np.random.seed(42)
    
    # Create realistic clusters
    cluster_centers = [
        (25, 30, 40),  # Young, low income, low spending
        (35, 65, 70),  # Mid-age, high income, high spending
        (50, 55, 35),  # Older, medium income, low spending
        (30, 85, 90),  # Young-mid, very high income, very high spending
        (45, 45, 50)   # Mid-older, medium income, medium spending
    ]
    
    samples_per_cluster = n_samples // len(cluster_centers)
    data = []
    
    for i, (age_c, income_c, spending_c) in enumerate(cluster_centers):
        for _ in range(samples_per_cluster):
            data.append({
                'CustomerID': len(data) + 1,
                'Gender': np.random.choice(['Male', 'Female']),
                'Age': int(np.random.normal(age_c, 8)),
                'Annual Income (k$)': int(np.random.normal(income_c, 12)),
                'Spending Score (1-100)': int(np.random.normal(spending_c, 10))
            })
    
    # Add remaining samples
    remaining = n_samples - len(data)
    for _ in range(remaining):
        data.append({
            'CustomerID': len(data) + 1,
            'Gender': np.random.choice(['Male', 'Female']),
            'Age': np.random.randint(18, 70),
            'Annual Income (k$)': np.random.randint(15, 137),
            'Spending Score (1-100)': np.random.randint(1, 100)
        })
    
    df = pd.DataFrame(data)
    # Clip values to realistic ranges
    df['Age'] = df['Age'].clip(18, 70)
    df['Annual Income (k$)'] = df['Annual Income (k$)'].clip(10, 150)
    df['Spending Score (1-100)'] = df['Spending Score (1-100)'].clip(1, 100)
    
    return df

def detect_column_mapping(df):
    """Intelligently detect and map column names with fuzzy matching"""
    columns_lower = [col.lower().replace('_', ' ').replace('-', ' ') for col in df.columns]
    
    patterns = {
        'invoice': ["invoiceno", "invoice", "transaction", "order", "transaction id", "order id"],
        'quantity': ["quantity", "qty", "units", "amount purchased"],
        'price': ["unitprice", "price", "unit price", "item price", "cost"],
        'customer': ["customerid", "customer id", "customer", "client", "client id"],
        'date': ["invoicedate", "date", "transaction date", "order date", "purchase date"],
        'age': ["age"],
        'income': ["annual income", "income", "salary", "yearly income"],
        'spending': ["spending score", "spending", "score", "purchase score"],
        'gender': ["gender", "sex"]
    }
    
    def find_column(expected_list):
        for exp in expected_list:
            for i, col in enumerate(columns_lower):
                if exp in col or col in exp:
                    return df.columns[i]
        return None
    
    mapping = {}
    for key, patterns_list in patterns.items():
        col = find_column(patterns_list)
        if col:
            mapping[key] = col
    
    return mapping

def process_transactional_data(df, mapping):
    """Convert transactional data to customer-level RFM metrics"""
    rename_dict = {
        mapping['invoice']: "InvoiceNo",
        mapping['quantity']: "Quantity",
        mapping['price']: "UnitPrice",
        mapping['customer']: "CustomerID"
    }
    
    if 'date' in mapping:
        rename_dict[mapping['date']] = "InvoiceDate"
    
    df = df.rename(columns=rename_dict)
    
    # Calculate transaction amount
    df["TransactionAmount"] = df["Quantity"] * df["UnitPrice"]
    
    # Remove negative values
    df = df[df["TransactionAmount"] > 0]
    df = df[df["Quantity"] > 0]
    
    # Convert date if available
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors='coerce')
        df = df.dropna(subset=["InvoiceDate"])
        max_date = df["InvoiceDate"].max()
        df["Recency"] = (max_date - df["InvoiceDate"]).dt.days
    
    # Aggregate by customer (RFM Analysis)
    agg_dict = {
        "InvoiceNo": "nunique",
        "TransactionAmount": ["sum", "mean", "max", "std"],
        "Quantity": ["sum", "mean"]
    }
    
    if "Recency" in df.columns:
        agg_dict["Recency"] = "min"
    
    customer_metrics = df.groupby("CustomerID").agg(agg_dict).reset_index()
    
    # Flatten column names
    customer_metrics.columns = ['_'.join(str(col)).strip('_') for col in customer_metrics.columns.values]
    customer_metrics = customer_metrics.rename(columns={
        'CustomerID_': 'CustomerID',
        'InvoiceNo_nunique': 'Frequency',
        'TransactionAmount_sum': 'Monetary',
        'TransactionAmount_mean': 'AvgTransactionValue',
        'TransactionAmount_max': 'MaxTransaction',
        'TransactionAmount_std': 'TransactionStdDev',
        'Quantity_sum': 'TotalQuantity',
        'Quantity_mean': 'AvgQuantity',
        'Recency_min': 'Recency'
    })
    
    # Fill NaN values
    customer_metrics = customer_metrics.fillna(0)
    
    return customer_metrics

@st.cache_data
def clean_data(df, numeric_cols):
    """Clean data by handling missing values, duplicates, and outliers"""
    initial_rows = len(df)
    
    # Handle missing values
    df = df.dropna(subset=numeric_cols)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Optional: Remove extreme outliers (beyond 3 std dev)
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
    
    removed_rows = initial_rows - len(df)
    
    return df, removed_rows

def encode_categorical(df, categorical_cols):
    """Encode categorical variables with proper handling"""
    le = LabelEncoder()
    encoded_df = df.copy()
    encoding_map = {}
    
    for col in categorical_cols:
        if col in encoded_df.columns:
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            encoding_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    return encoded_df, encoding_map

def create_features(df):
    """Engineer new features with business logic"""
    new_features = []
    
    # Spending Efficiency (Spending relative to income)
    if "Annual Income (k$)" in df.columns and "Spending Score (1-100)" in df.columns:
        df["Spending_Efficiency"] = df["Spending Score (1-100)"] / (df["Annual Income (k$)"] + 1)
        new_features.append("Spending_Efficiency")
    
    # Income per Age (Economic productivity indicator)
    if "Annual Income (k$)" in df.columns and "Age" in df.columns:
        df["Income_per_Age"] = df["Annual Income (k$)"] / (df["Age"] + 1)
        new_features.append("Income_per_Age")
    
    # Age Group
    if "Age" in df.columns:
        df["Age_Group"] = pd.cut(df["Age"], bins=[0, 25, 35, 45, 55, 100], 
                                 labels=[1, 2, 3, 4, 5])
        df["Age_Group"] = df["Age_Group"].astype(int)
        new_features.append("Age_Group")
    
    # Income Group
    if "Annual Income (k$)" in df.columns:
        df["Income_Group"] = pd.qcut(df["Annual Income (k$)"], q=4, 
                                     labels=[1, 2, 3, 4], duplicates='drop')
        df["Income_Group"] = df["Income_Group"].astype(int)
        new_features.append("Income_Group")
    
    # RFM Score (if RFM data exists)
    if all(col in df.columns for col in ["Recency", "Frequency", "Monetary"]):
        # Normalize RFM values to 1-5 scale
        try:
            df["Recency_Score"] = pd.qcut(df["Recency"], q=5, labels=[5,4,3,2,1], duplicates='drop')
            df["Frequency_Score"] = pd.qcut(df["Frequency"], q=5, labels=[1,2,3,4,5], duplicates='drop')
            df["Monetary_Score"] = pd.qcut(df["Monetary"], q=5, labels=[1,2,3,4,5], duplicates='drop')
            
            df["RFM_Score"] = (
                df["Recency_Score"].astype(int) + 
                df["Frequency_Score"].astype(int) + 
                df["Monetary_Score"].astype(int)
            )
            new_features.extend(["RFM_Score"])
        except:
            pass  # Skip if qcut fails due to insufficient unique values
    
    # Customer Value (for RFM)
    if "Frequency" in df.columns and "Monetary" in df.columns:
        df["Customer_Value"] = df["Frequency"] * df["Monetary"]
        new_features.append("Customer_Value")
    
    return df, new_features

@st.cache_data
def calculate_optimal_clusters(scaled_data, max_k=10):
    """Calculate comprehensive metrics for different K values"""
    if len(scaled_data) < 10:
        max_k = min(max_k, len(scaled_data) - 1)
    
    metrics = {
        'k': [],
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(scaled_data)
        
        # Check if we have more than 1 cluster
        if len(set(labels)) > 1:
            metrics['k'].append(k)
            metrics['inertia'].append(kmeans.inertia_)
            metrics['silhouette'].append(silhouette_score(scaled_data, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(scaled_data, labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(scaled_data, labels))
    
    return pd.DataFrame(metrics)

def perform_clustering(scaled_data, algorithm='KMeans', n_clusters=5, **kwargs):
    """Perform clustering with selected algorithm and parameters"""
    if algorithm == 'KMeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    elif algorithm == 'Hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 5))
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    labels = model.fit_predict(scaled_data)
    return labels, model

def generate_cluster_insights(df, features, cluster_col='Cluster'):
    """Generate comprehensive insights for each cluster"""
    insights = []
    
    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id]
        size = len(cluster_data)
        percentage = (size / len(df)) * 100
        
        # Calculate statistics
        stats = {}
        for feature in features:
            stats[feature] = {
                'mean': cluster_data[feature].mean(),
                'median': cluster_data[feature].median(),
                'std': cluster_data[feature].std(),
                'min': cluster_data[feature].min(),
                'max': cluster_data[feature].max()
            }
        
        insights.append({
            'cluster_id': cluster_id,
            'size': size,
            'percentage': percentage,
            'statistics': stats
        })
    
    return insights

def create_downloadable_report(df, insights, features, metrics=None):
    """Create comprehensive Excel report with multiple sheets"""
    try:
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Clustered Data
            df.to_excel(writer, sheet_name='Clustered_Data', index=False)
            
            # Sheet 2: Cluster Summary
            summary_data = []
            for insight in insights:
                row = {
                    'Cluster': insight['cluster_id'],
                    'Size': insight['size'],
                    'Percentage': f"{insight['percentage']:.2f}%"
                }
                for feature in features:
                    row[f'{feature}_Mean'] = insight['statistics'][feature]['mean']
                    row[f'{feature}_Median'] = insight['statistics'][feature]['median']
                    row[f'{feature}_StdDev'] = insight['statistics'][feature]['std']
                summary_data.append(row)
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Cluster_Summary', index=False)
            
            # Sheet 3: Metrics (if provided)
            if metrics is not None:
                metrics.to_excel(writer, sheet_name='Clustering_Metrics', index=False)
        
        output.seek(0)
        return output
    except Exception as e:
        logger.error(f"Failed to create Excel report: {str(e)}")
        # Return None if Excel export fails - UI will handle gracefully
        return None

def create_professional_chart(data, chart_type, **kwargs):
    """Create professional-looking charts with consistent styling"""
    if chart_type == 'bar':
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data.index,
            y=data.values,
            marker_color='#667eea',
            marker_line_color='#5568d3',
            marker_line_width=1.5,
            text=data.values,
            textposition='auto',
        ))
        fig.update_layout(
            title=kwargs.get('title', ''),
            xaxis_title=kwargs.get('xlabel', ''),
            yaxis_title=kwargs.get('ylabel', ''),
            template='plotly_white',
            height=400
        )
        return fig
    
    elif chart_type == 'pie':
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=data.index,
            values=data.values,
            hole=0.4,
            marker_colors=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']
        ))
        fig.update_layout(
            title=kwargs.get('title', ''),
            template='plotly_white',
            height=400
        )
        return fig

# ==============================================
# MAIN APPLICATION
# ==============================================

def main():
    # Dashboard Header
    st.markdown("""
        <div class="dashboard-header">
            <h1 class="dashboard-title">📊 CUSTOMER SEGMENTATION DASHBOARD</h1>
            <p class="dashboard-subtitle">Advanced Analytics & Machine Learning Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    # ==============================================
    # SIDEBAR CONFIGURATION
    # ==============================================
    
    with st.sidebar:
        st.markdown("### ⚙️ CONFIGURATION")
        st.markdown("---")
        
        # Logo/Branding Area
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0 2rem 0;'>
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            width: 80px; height: 80px; margin: 0 auto; border-radius: 20px; 
                            display: flex; align-items: center; justify-content: center;'>
                    <span style='font-size: 2.5rem;'>🎯</span>
                </div>
                <h3 style='color: white; margin-top: 1rem; font-size: 1.2rem;'>Analytics Pro</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Data Source Selection
        st.markdown("#### 📁 Data Source")
        data_source = st.radio(
            "Choose your data:",
            ["Upload CSV File", "Use Sample Data"],
            label_visibility="collapsed",
            key="main_data_source_radio"
        )
        
        if data_source == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type="csv",
                help="Upload your customer data in CSV format"
            )
            if uploaded_file:
                df, encoding = load_data(uploaded_file)
                if df is not None:
                    st.success(f"✅ Loaded ({encoding})")
                else:
                    st.stop()
            else:
                st.info("👆 Upload a CSV file to begin")
                st.stop()
        else:
            st.markdown("#### 📊 Sample Data Settings")
            n_samples = st.slider(
                "Number of samples:",
                min_value=100,
                max_value=1000,
                value=200,
                step=50
            )
            df = generate_sample_data(n_samples)
            st.success(f"✅ Generated {n_samples} samples")
    
    st.markdown("---")
    
    # Quick Stats in Sidebar
    st.markdown("#### 📈 Quick Stats")
    st.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
            <div style='color: #e0e7ff; font-size: 0.8rem;'>Total Records</div>
            <div style='color: white; font-size: 1.5rem; font-weight: bold;'>{len(df):,}</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
            <div style='color: #e0e7ff; font-size: 0.8rem;'>Features</div>
            <div style='color: white; font-size: 1.5rem; font-weight: bold;'>{len(df.columns)}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # ==============================================
    # MAIN CONTENT AREA
    # ==============================================
    
    # Section 1: Data Overview
    st.markdown('<p class="section-header">📋 Data Overview</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Records",
            f"{len(df):,}",
            help="Total number of customer records"
        )
    
    with col2:
        st.metric(
            "Total Features",
            len(df.columns),
            help="Number of columns in the dataset"
        )
    
    with col3:
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        st.metric(
            "Numeric Features",
            numeric_count,
            help="Number of numeric columns"
        )
    
    with col4:
        categorical_count = len(df.select_dtypes(include=['object']).columns)
        st.metric(
            "Categorical Features",
            categorical_count,
            help="Number of categorical columns"
        )
    
    # Data Preview
    with st.expander("📊 View Dataset Preview", expanded=False):
        st.dataframe(
            df.head(20),
            use_container_width=True,
            height=400
        )
        
        # Basic Statistics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Data Types**")
            st.dataframe(df.dtypes.to_frame().rename(columns={0: 'Type'}), use_container_width=True)
        with col2:
            st.markdown("**Missing Values**")
            missing_df = df.isnull().sum().to_frame().rename(columns={0: 'Missing'})
            missing_df['Percentage'] = (missing_df['Missing'] / len(df) * 100).round(2)
            st.dataframe(missing_df, use_container_width=True)
    
    # Detect column mapping and process transactional data
    mapping = detect_column_mapping(df)
    is_transactional = all(key in mapping for key in ['invoice', 'quantity', 'price', 'customer'])
    
    if is_transactional:
        st.info("🔄 Transactional data detected! Converting to customer-level RFM metrics...")
        with st.spinner("Processing transactional data..."):
            df = process_transactional_data(df, mapping)
        st.success("✅ Successfully converted to RFM metrics (Recency, Frequency, Monetary)")
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if not col.lower().endswith('id')]
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if not col.lower().endswith('id')]
    
    # ==============================================
    # DATA CLEANING & PREPROCESSING
    # ==============================================
    
    st.markdown('<p class="section-header">🧹 Data Cleaning & Preprocessing</p>', unsafe_allow_html=True)
    
    missing_before = df.isnull().sum().sum()
    
    with st.spinner("Cleaning data..."):
        df, removed_rows = clean_data(df, numeric_cols)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Missing Values",
            "0",
            delta=f"-{missing_before}",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Rows Removed",
            removed_rows,
            help="Rows removed due to missing values or being duplicates"
        )
    
    with col3:
        st.metric(
            "Clean Records",
            f"{len(df):,}",
            help="Final number of clean records"
        )
    
    with col4:
        retention_rate = (len(df) / (len(df) + removed_rows) * 100)
        st.metric(
            "Retention Rate",
            f"{retention_rate:.1f}%",
            help="Percentage of data retained after cleaning"
        )
    
    # Encode categorical variables
    if categorical_cols:
        with st.spinner("Encoding categorical variables..."):
            df, encoding_map = encode_categorical(df, categorical_cols)
        st.success(f"✅ Encoded {len(categorical_cols)} categorical column(s)")
    
    # ==============================================
    # FEATURE ENGINEERING
    # ==============================================
    
    st.markdown('<p class="section-header">🔧 Feature Engineering</p>', unsafe_allow_html=True)
    
    with st.spinner("Creating new features..."):
        df, new_features = create_features(df)
    
    if new_features:
        st.success(f"✅ Created {len(new_features)} new feature(s): **{', '.join(new_features)}**")
        numeric_cols.extend(new_features)
        
        # Show new features info
        with st.expander("ℹ️ New Features Details"):
            for feature in new_features:
                if feature in df.columns:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{feature} - Mean", f"{df[feature].mean():.2f}")
                    with col2:
                        st.metric(f"{feature} - Min", f"{df[feature].min():.2f}")
                    with col3:
                        st.metric(f"{feature} - Max", f"{df[feature].max():.2f}")
    else:
        st.info("No additional features were created for this dataset")
    
    # ==============================================
    # FEATURE SELECTION
    # ==============================================
    
    st.markdown('<p class="section-header">🎯 Feature Selection for Clustering</p>', unsafe_allow_html=True)
    
    if len(numeric_cols) == 0:
        st.error("❌ No numeric features available for clustering! Please check your dataset.")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_features = st.multiselect(
            "Select features for clustering analysis:",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))],
            help="Choose the features that will be used for customer segmentation"
        )
    
    with col2:
        if selected_features:
            st.info(f"**{len(selected_features)}** feature(s) selected")
    
    if not selected_features:
        st.warning("⚠️ Please select at least one feature to proceed")
        st.stop()
    
    # Show feature statistics
    with st.expander("📊 Selected Features Statistics"):
        stats_df = df[selected_features].describe().T
        stats_df['missing'] = df[selected_features].isnull().sum()
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
    
    # ==============================================
    # EXPLORATORY DATA ANALYSIS
    # ==============================================
    
    st.markdown('<p class="section-header">🔍 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "📊 Distributions",
        "🔗 Correlations",
        "📈 Scatter Analysis",
        "📉 Box Plots"
    ])
    
    with eda_tab1:
        st.markdown("### Feature Distributions")
        
        if len(selected_features) > 0:
            cols_per_row = 3
            rows = (len(selected_features) + cols_per_row - 1) // cols_per_row
            
            for i in range(rows):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i * cols_per_row + j
                    if idx < len(selected_features):
                        with cols[j]:
                            try:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                
                                # Histogram with KDE
                                sns.histplot(
                                    df[selected_features[idx]],
                                    kde=True,
                                    ax=ax,
                                    color='#667eea',
                                    edgecolor='white'
                                )
                                
                                ax.set_title(
                                    selected_features[idx],
                                    fontsize=12,
                                    fontweight='bold',
                                    color='#1e3a5f'
                                )
                                ax.set_xlabel('')
                                ax.grid(True, alpha=0.3)
                                ax.spines['top'].set_visible(False)
                                ax.spines['right'].set_visible(False)
                                
                                st.pyplot(fig)
                                plt.close()
                            except Exception as e:
                                logger.error(f"Failed to plot {selected_features[idx]}: {str(e)}")
                                st.warning(f"⚠️ Could not visualize {selected_features[idx]}")
    
    with eda_tab2:
        st.markdown("### Correlation Matrix")
        
        if len(selected_features) >= 2:
            try:
                # Calculate correlation
                corr = df[selected_features].corr()
                
                # Create heatmap using plotly
                fig = px.imshow(
                    corr,
                    labels=dict(color="Correlation"),
                    x=selected_features,
                    y=selected_features,
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    title="Feature Correlation Heatmap"
                )
                
                fig.update_layout(
                    height=600,
                    title_font_size=16
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Highly correlated pairs
                st.markdown("#### 🔍 Highly Correlated Feature Pairs")
                
                # Get correlation pairs
                corr_pairs = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        corr_pairs.append({
                            'Feature 1': corr.columns[i],
                            'Feature 2': corr.columns[j],
                            'Correlation': corr.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(corr_pairs)
                corr_df = corr_df[abs(corr_df['Correlation']) > 0.5].sort_values(
                    'Correlation',
                    ascending=False
                )
                
                if len(corr_df) > 0:
                    st.dataframe(
                        corr_df.style.format({'Correlation': '{:.3f}'}),
                        use_container_width=True
                    )
                else:
                    st.info("No highly correlated feature pairs found (|r| > 0.5)")
            except Exception as e:
                logger.error(f"Failed to create correlation matrix: {str(e)}")
                st.warning(f"⚠️ Could not generate correlation heatmap: {str(e)}")
        else:
            st.info("Select at least 2 features to view correlations")
    
    with eda_tab3:
        st.markdown("### Scatter Plot Analysis")
        
        if len(selected_features) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox(
                    "X-axis:",
                    selected_features,
                    key="eda_scatter_x"
                )
            
            with col2:
                y_axis = st.selectbox(
                    "Y-axis:",
                    selected_features,
                    index=1 if len(selected_features) > 1 else 0,
                    key="eda_scatter_y"
                )
            
            if x_axis != y_axis:
                try:
                    # Create interactive scatter plot
                    fig = px.scatter(
                        df,
                        x=x_axis,
                        y=y_axis,
                        opacity=0.6,
                        title=f"{y_axis} vs {x_axis}",
                        template='plotly_white',
                        height=600
                    )
                    
                    fig.update_traces(
                        marker=dict(size=8, color='#667eea', line=dict(width=0.5, color='white'))
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Failed to create scatter plot: {str(e)}")
                    st.warning(f"⚠️ Could not create scatter plot: {str(e)}")
            else:
                st.warning("Please select different features for X and Y axes")
        else:
            st.info("Select at least 2 features for scatter plot analysis")
    
    with eda_tab4:
        st.markdown("### Box Plot Analysis")
        
        if len(selected_features) > 0:
            try:
                # Create box plots
                fig = go.Figure()
                
                for feature in selected_features:
                    fig.add_trace(go.Box(
                        y=df[feature],
                        name=feature,
                        boxmean='sd'
                    ))
                
                fig.update_layout(
                    title="Feature Distribution Box Plots",
                    yaxis_title="Value",
                    template='plotly_white',
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                logger.error(f"Failed to create box plots: {str(e)}")
                st.warning(f"⚠️ Could not create box plots: {str(e)}")
    
    # ==============================================
    # FEATURE SCALING
    # ==============================================
    
    with st.spinner("Scaling features..."):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[selected_features])
    
    # ==============================================
    # OPTIMAL CLUSTER DETERMINATION
    # ==============================================
    
    st.markdown('<p class="section-header">🎯 Optimal Cluster Determination</p>', unsafe_allow_html=True)
    
    with st.spinner("Calculating clustering metrics for different K values..."):
        metrics_df = calculate_optimal_clusters(scaled_data, max_k=10)
    
    if len(metrics_df) == 0:
        st.error("Unable to calculate clustering metrics. Please check your data.")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Elbow Method
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metrics_df['k'],
            y=metrics_df['inertia'],
            mode='lines+markers',
            name='Inertia',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Elbow Method - WCSS vs K",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="WCSS (Inertia)",
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Silhouette Score
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metrics_df['k'],
            y=metrics_df['silhouette'],
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='#764ba2', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Silhouette Score vs K",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Silhouette Score",
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional metrics
    with st.expander("📊 View All Clustering Metrics"):
        # Style the dataframe
        st.dataframe(
            metrics_df.style.format({
                'inertia': '{:.2f}',
                'silhouette': '{:.4f}',
                'davies_bouldin': '{:.4f}',
                'calinski_harabasz': '{:.2f}'
            }).background_gradient(subset=['silhouette'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # Suggest optimal K
        optimal_k_silhouette = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']
        optimal_k_db = metrics_df.loc[metrics_df['davies_bouldin'].idxmin(), 'k']
        optimal_k_ch = metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'k']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Optimal K (Silhouette)",
                int(optimal_k_silhouette),
                help="Based on maximum Silhouette Score"
            )
        
        with col2:
            st.metric(
                "Optimal K (Davies-Bouldin)",
                int(optimal_k_db),
                help="Based on minimum Davies-Bouldin Index"
            )
        
        with col3:
            st.metric(
                "Optimal K (Calinski-Harabasz)",
                int(optimal_k_ch),
                help="Based on maximum Calinski-Harabasz Score"
            )
    
    # ==============================================
    # CLUSTERING APPLICATION
    # ==============================================
    
    st.markdown('<p class="section-header">🤖 Apply Clustering Algorithm</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        algorithm = st.selectbox(
            "Select Clustering Algorithm:",
            ["KMeans", "Hierarchical", "DBSCAN"],
            help="Choose the clustering algorithm to apply"
        )
    
    with col2:
        if algorithm in ["KMeans", "Hierarchical"]:
            n_clusters = st.slider(
                "Number of Clusters:",
                min_value=2,
                max_value=10,
                value=5,
                help="Specify the number of customer segments"
            )
        else:
            eps = st.slider(
                "EPS (Neighborhood Distance):",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Maximum distance between two samples"
            )
    
    with col3:
        if algorithm == "DBSCAN":
            min_samples = st.slider(
                "Minimum Samples:",
                min_value=2,
                max_value=20,
                value=5,
                help="Minimum samples in a neighborhood"
            )
    
    # Perform clustering
    with st.spinner(f"Applying {algorithm} clustering..."):
        if algorithm in ["KMeans", "Hierarchical"]:
            labels, model = perform_clustering(scaled_data, algorithm, n_clusters)
        else:
            labels, model = perform_clustering(scaled_data, algorithm, eps=eps, min_samples=min_samples)
    
    df['Cluster'] = labels
    
    # Calculate and display metrics
    if len(set(labels)) > 1:
        sil_score = silhouette_score(scaled_data, labels)
        db_score = davies_bouldin_score(scaled_data, labels)
        ch_score = calinski_harabasz_score(scaled_data, labels)
        
        st.markdown("### 📊 Clustering Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Clusters Found",
                len(set(labels)),
                help="Number of distinct clusters identified"
            )
        
        with col2:
            # Silhouette score interpretation
            sil_quality = "Excellent" if sil_score > 0.7 else "Good" if sil_score > 0.5 else "Fair" if sil_score > 0.25 else "Poor"
            st.metric(
                "Silhouette Score",
                f"{sil_score:.3f}",
                delta=sil_quality,
                help="Measures how similar an object is to its own cluster compared to other clusters"
            )
        
        with col3:
            db_quality = "Good" if db_score < 1.0 else "Fair" if db_score < 1.5 else "Poor"
            st.metric(
                "Davies-Bouldin Index",
                f"{db_score:.3f}",
                delta=db_quality,
                delta_color="inverse",
                help="Lower values indicate better clustering (closer to 0 is better)"
            )
        
        with col4:
            st.metric(
                "Calinski-Harabasz Index",
                f"{ch_score:.1f}",
                help="Higher values indicate better defined clusters"
            )
        
        # Interpretation guide
        with st.expander("📖 How to Interpret These Metrics"):
            st.markdown("""
            **Silhouette Score** (Range: -1 to 1)
            - 0.7 to 1.0: Excellent separation
            - 0.5 to 0.7: Reasonable separation
            - 0.25 to 0.5: Weak separation
            - < 0.25: No substantial structure
            
            **Davies-Bouldin Index** (Lower is better)
            - < 1.0: Good clustering
            - 1.0 to 1.5: Fair clustering
            - > 1.5: Poor clustering
            
            **Calinski-Harabasz Index** (Higher is better)
            - Higher values indicate better defined, more separated clusters
            """)
    else:
        st.warning("⚠️ Only one cluster was detected. Try adjusting the algorithm parameters.")
    
    # ==============================================
    # CLUSTER VISUALIZATIONS
    # ==============================================
    
    st.markdown('<p class="section-header">📈 Cluster Visualization</p>', unsafe_allow_html=True)
    
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "🎨 2D Scatter",
        "🌐 3D Interactive",
        "📊 PCA Projection",
        "🎯 Cluster Profiles"
    ])
    
    with viz_tab1:
        st.markdown("### 2D Scatter Plot Visualization")
        
        if len(selected_features) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_feature = st.selectbox(
                    "X-axis Feature:",
                    selected_features,
                    key="viz2d_x"
                )
            
            with col2:
                y_feature = st.selectbox(
                    "Y-axis Feature:",
                    selected_features,
                    index=1 if len(selected_features) > 1 else 0,
                    key="viz2d_y"
                )
            
            if x_feature != y_feature:
                # Create interactive scatter plot
                fig = px.scatter(
                    df,
                    x=x_feature,
                    y=y_feature,
                    color='Cluster',
                    title=f"Customer Segments: {y_feature} vs {x_feature}",
                    template='plotly_white',
                    height=600,
                    color_continuous_scale='viridis'
                )
                
                fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='white')))
                fig.update_layout(
                    font=dict(size=12),
                    title_font_size=16
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select different features for X and Y axes")
        else:
            st.info("Select at least 2 features for 2D visualization")
    
    with viz_tab2:
        st.markdown("### 3D Interactive Visualization")
        
        if len(selected_features) >= 3:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                dim1 = st.selectbox(
                    "1st Dimension:",
                    selected_features,
                    key="viz3d_x"
                )
            
            with col2:
                dim2 = st.selectbox(
                    "2nd Dimension:",
                    selected_features,
                    index=1 if len(selected_features) > 1 else 0,
                    key="viz3d_y"
                )
            
            with col3:
                dim3 = st.selectbox(
                    "3rd Dimension:",
                    selected_features,
                    index=2 if len(selected_features) > 2 else 0,
                    key="viz3d_z"
                )
            
            if len({dim1, dim2, dim3}) == 3:
                # Create 3D scatter plot
                fig = px.scatter_3d(
                    df,
                    x=dim1,
                    y=dim2,
                    z=dim3,
                    color='Cluster',
                    title="3D Customer Segmentation Visualization",
                    hover_data=selected_features,
                    color_continuous_scale='viridis',
                    height=700
                )
                
                fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='white')))
                fig.update_layout(
                    scene=dict(
                        xaxis_title=dim1,
                        yaxis_title=dim2,
                        zaxis_title=dim3
                    ),
                    title_font_size=16
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select 3 different dimensions")
        else:
            st.info("Select at least 3 features for 3D visualization")
    
    with viz_tab3:
        st.markdown("### PCA Projection Visualization")
        
        # Perform PCA
        pca = PCA(n_components=min(3, len(selected_features)))
        pca_data = pca.fit_transform(scaled_data)
        
        # Create PCA dataframe
        pca_df = pd.DataFrame(
            pca_data,
            columns=[f'PC{i+1}' for i in range(pca_data.shape[1])]
        )
        pca_df['Cluster'] = df['Cluster'].values
        
        # Display explained variance
        col1, col2, col3 = st.columns(3)
        
        for i, col in enumerate([col1, col2, col3]):
            if i < len(pca.explained_variance_ratio_):
                with col:
                    st.metric(
                        f"PC{i+1} Variance",
                        f"{pca.explained_variance_ratio_[i]:.1%}",
                        help=f"Variance explained by Principal Component {i+1}"
                    )
        
        # 2D PCA Plot
        if pca_data.shape[1] >= 2:
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                title="PCA Projection (First 2 Components)",
                template='plotly_white',
                height=600,
                color_continuous_scale='viridis'
            )
            
            fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='white')))
            fig.update_layout(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                title_font_size=16
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Variance explanation
        total_variance = sum(pca.explained_variance_ratio_)
        st.info(f"💡 Total variance explained by {len(pca.explained_variance_ratio_)} components: **{total_variance:.1%}**")
        
        # Scree plot
        with st.expander("📊 View Scree Plot"):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                y=pca.explained_variance_ratio_,
                marker_color='#667eea'
            ))
            
            fig.update_layout(
                title="Scree Plot - Variance Explained by Each Component",
                xaxis_title="Principal Component",
                yaxis_title="Variance Explained",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab4:
        st.markdown("### Cluster Profile Comparison")
        
        # Create radar chart for clusters
        cluster_profiles = df.groupby('Cluster')[selected_features].mean()
        
        # Normalize for radar chart
        from sklearn.preprocessing import MinMaxScaler
        scaler_radar = MinMaxScaler()
        cluster_profiles_normalized = pd.DataFrame(
            scaler_radar.fit_transform(cluster_profiles.T).T,
            columns=cluster_profiles.columns,
            index=cluster_profiles.index
        )
        
        # Create radar chart
        fig = go.Figure()
        
        for cluster_id in cluster_profiles_normalized.index:
            fig.add_trace(go.Scatterpolar(
                r=cluster_profiles_normalized.loc[cluster_id].values.tolist() + [cluster_profiles_normalized.loc[cluster_id].values[0]],
                theta=selected_features + [selected_features[0]],
                fill='toself',
                name=f'Cluster {cluster_id}'
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Normalized Cluster Profiles (Radar Chart)",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap of cluster profiles
        st.markdown("#### Cluster Feature Heatmap")
        
        fig = px.imshow(
            cluster_profiles,
            labels=dict(x="Features", y="Cluster", color="Average Value"),
            x=selected_features,
            y=[f'Cluster {i}' for i in cluster_profiles.index],
            color_continuous_scale='RdYlBu_r',
            aspect="auto",
            title="Average Feature Values by Cluster"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # ==============================================
    # CLUSTER INSIGHTS & SUMMARY
    # ==============================================
    
    st.markdown('<p class="section-header">💡 Cluster Insights & Analysis</p>', unsafe_allow_html=True)
    
    # Generate comprehensive insights
    insights = generate_cluster_insights(df, selected_features)
    
    # Overview section
    st.markdown("### 📊 Cluster Distribution Overview")
    
    cluster_counts = df['Cluster'].value_counts().sort_index()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Pie chart
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=[f'Cluster {i}' for i in cluster_counts.index],
            values=cluster_counts.values,
            hole=0.4,
            marker_colors=px.colors.sequential.Viridis
        ))
        
        fig.update_layout(
            title="Cluster Size Distribution",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Summary table
        st.markdown("#### Cluster Summary")
        
        summary_data = []
        for insight in insights:
            row = {
                'Cluster': f"Cluster {insight['cluster_id']}",
                'Size': insight['size'],
                'Percentage': f"{insight['percentage']:.2f}%"
            }
            
            # Add top 3 features
            for feature in selected_features[:3]:
                row[f'{feature} (Avg)'] = f"{insight['statistics'][feature]['mean']:.2f}"
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(
            summary_df,
            use_container_width=True,
            height=350
        )
    
    # Detailed cluster profiles
    st.markdown("### 🎯 Detailed Cluster Profiles")
    
    for insight in insights:
        cluster_id = insight['cluster_id']
        size = insight['size']
        percentage = insight['percentage']
        
        with st.expander(
            f"🔍 Cluster {cluster_id} - {size} customers ({percentage:.1f}%)",
            expanded=False
        ):
            # Create profile table
            profile_data = []
            
            for feature in selected_features:
                stats = insight['statistics'][feature]
                profile_data.append({
                    'Feature': feature,
                    'Mean': f"{stats['mean']:.2f}",
                    'Median': f"{stats['median']:.2f}",
                    'Std Dev': f"{stats['std']:.2f}",
                    'Min': f"{stats['min']:.2f}",
                    'Max': f"{stats['max']:.2f}"
                })
            
            profile_df = pd.DataFrame(profile_data)
            
            # Display in columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(
                    profile_df,
                    use_container_width=True,
                    height=min(400, len(profile_df) * 50)
                )
            
            with col2:
                st.markdown("**Cluster Characteristics:**")
                
                # Calculate relative position (high/medium/low)
                for feature in selected_features[:3]:
                    mean_val = insight['statistics'][feature]['mean']
                    overall_mean = df[feature].mean()
                    
                    if mean_val > overall_mean * 1.2:
                        level = "🔴 High"
                    elif mean_val < overall_mean * 0.8:
                        level = "🔵 Low"
                    else:
                        level = "🟡 Medium"
                    
                    st.markdown(f"**{feature}:** {level}")
    
    # ==============================================
    # EXPORT & DOWNLOAD
    # ==============================================
    
    st.markdown('<p class="section-header">📥 Export Results</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Clustered Data (CSV)",
            data=csv,
            file_name=f"customer_segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel Report
        with st.spinner("Generating Excel report..."):
            excel_report = create_downloadable_report(df, insights, selected_features, metrics_df)
        
        if excel_report is not None:
            st.download_button(
                label="📊 Download Full Report (Excel)",
                data=excel_report,
                file_name=f"segmentation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.warning("⚠️ Excel export unavailable - use CSV export instead")
    
    with col3:
        # Summary PDF (placeholder - would need reportlab)
        st.download_button(
            label="📑 Download Summary (CSV)",
            data=pd.DataFrame([{
                'Metric': 'Total Records',
                'Value': len(df)
            }, {
                'Metric': 'Number of Clusters',
                'Value': len(set(labels))
            }, {
                'Metric': 'Silhouette Score',
                'Value': f"{sil_score:.3f}" if len(set(labels)) > 1 else "N/A"
            }]).to_csv(index=False).encode('utf-8'),
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Success message
    st.success("✅ Customer Segmentation Analysis Completed Successfully!")
    
    # ==============================================
    # FOOTER
    # ==============================================
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0; color: #64748b;'>
            <p style='font-size: 0.9rem; margin-bottom: 0.5rem;'>
                <strong>Customer Segmentation Dashboard v2.0</strong>
            </p>
                        <p style='font-size: 0.75rem; color: #94a3b8;'>
                © 2026 Analytics Pro | Advanced Machine Learning Platform
            </p>
        </div>
    """, unsafe_allow_html=True)

# ==============================================
# RUN APPLICATION WITH ERROR HANDLING
# ==============================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Comprehensive error handling - no module not found messages
        error_msg = str(e)
        logger.error(f"Application error: {error_msg}")
        logger.error(traceback.format_exc())
        
        st.error("❌ An error occurred in the application")
        
        with st.expander("📋 Error Details (for developers)"):
            st.code(f"{type(e).__name__}: {error_msg}", language="python")
            st.code(traceback.format_exc(), language="python")
        
        st.warning("""
        **Troubleshooting Steps:**
        1. Check that your CSV file has the correct format
        2. Ensure numeric columns are properly formatted
        3. Try using the sample data first to verify functionality
        4. Refresh the page and try again
        
        If the issue persists, please contact support with the error details above.
        """)
