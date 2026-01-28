# ==============================================
# ENTERPRISE CUSTOMER SEGMENTATION DASHBOARD
# ==============================================
"""
Professional Enterprise Features:
- Clean, emoji-free professional UI
- Enterprise dashboard styling
- Smooth CSS animations
- Real-world dashboard design
- Banking/corporate aesthetic
- Advanced analytics
- Production-ready code
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

# ==============================================
# LOGGING CONFIGURATION
# ==============================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================
# PAGE CONFIGURATION
# ==============================================

st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================
# PROFESSIONAL ENTERPRISE STYLING
# ==============================================

st.markdown("""
    <style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-feature-settings: "ss01", "ss02", "cv01", "cv11";
    }
    
    /* Smooth Page Transitions */
    .main {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Sidebar - Professional Dark Theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem 1rem;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.15);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stSelectbox > label,
    [data-testid="stSidebar"] .stSlider > label {
        color: #cbd5e1 !important;
        font-weight: 500;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Main Content - Clean White Background */
    .main {
        background-color: #f8fafc;
    }
    
    /* Professional Header */
    .dashboard-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(30, 64, 175, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    
    .dashboard-subtitle {
        font-size: 1.125rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Section Headers with Animation */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #3b82f6;
        position: relative;
        animation: slideInLeft 0.6s ease-out;
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 120px;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6 0%, transparent 100%);
    }
    
    /* Professional Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2.25rem;
        font-weight: 700;
        color: #0f172a;
        font-variant-numeric: tabular-nums;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    div[data-testid="stMetric"] {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1), 0 4px 8px rgba(0, 0, 0, 0.06);
        border-color: #3b82f6;
    }
    
    /* Professional Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.875rem;
        letter-spacing: 0.025em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(37, 99, 235, 0.3);
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(5, 150, 105, 0.3);
        background: linear-gradient(135deg, #047857 0%, #059669 100%);
    }
    
    /* Professional Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: white;
        padding: 0.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #64748b;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-size: 0.875rem;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f1f5f9;
        color: #1e40af;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: white !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }
    
    /* Professional Expander */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 10px;
        font-weight: 600;
        color: #0f172a;
        padding: 1rem 1.25rem;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #3b82f6;
        background-color: #f8fafc;
    }
    
    /* Alert Boxes */
    .stAlert {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        animation: slideIn 0.4s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* DataFrame Styling */
    .dataframe {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Input Fields */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:hover,
    .stMultiSelect > div > div:hover {
        border-color: #3b82f6;
    }
    
    /* Slider Styling */
    .stSlider > div > div > div {
        background-color: #e2e8f0;
    }
    
    .stSlider > div > div > div > div {
        background-color: #3b82f6;
    }
    
    /* Custom Card */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .custom-card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-color: #3b82f6 transparent transparent transparent !important;
    }
    
    /* Professional Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 5px;
        transition: background 0.2s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Professional Stats Badge */
    .stats-badge {
        display: inline-block;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        color: #1e40af;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        border: 1px solid #bfdbfe;
    }
    
    /* Section Divider */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, #e2e8f0 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Professional Logo Container */
    .logo-container {
        text-align: center;
        padding: 2rem 0;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .logo-icon {
        width: 80px;
        height: 80px;
        margin: 0 auto;
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.3);
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .logo-icon svg {
        width: 40px;
        height: 40px;
        color: white;
    }
    
    .logo-text {
        color: #e2e8f0;
        margin-top: 1rem;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.95); }
    }
    
    .status-success { background-color: #10b981; box-shadow: 0 0 10px rgba(16, 185, 129, 0.5); }
    .status-warning { background-color: #f59e0b; box-shadow: 0 0 10px rgba(245, 158, 11, 0.5); }
    .status-info { background-color: #3b82f6; box-shadow: 0 0 10px rgba(59, 130, 246, 0.5); }
    
    /* Progress Bar Animation */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 50%, #60a5fa 100%);
        background-size: 200% 100%;
        animation: progressBar 2s ease infinite;
    }
    
    @keyframes progressBar {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Tooltip Enhancement */
    [data-testid="stTooltipHoverTarget"] {
        transition: all 0.2s ease;
    }
    
    [data-testid="stTooltipHoverTarget"]:hover {
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================
# UTILITY FUNCTIONS (SAME AS BEFORE)
# ==============================================

@st.cache_data(ttl=3600)
def load_data(file):
    """Load CSV with multiple encoding attempts"""
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
    """Generate realistic sample data"""
    np.random.seed(42)
    cluster_centers = [
        (25, 30, 40), (35, 65, 70), (50, 55, 35),
        (30, 85, 90), (45, 45, 50)
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
    df['Age'] = df['Age'].clip(18, 70)
    df['Annual Income (k$)'] = df['Annual Income (k$)'].clip(10, 150)
    df['Spending Score (1-100)'] = df['Spending Score (1-100)'].clip(1, 100)
    
    return df

def detect_column_mapping(df):
    """Intelligently detect column names"""
    columns_lower = [col.lower().replace('_', ' ').replace('-', ' ') for col in df.columns]
    
    patterns = {
        'invoice': ["invoiceno", "invoice", "transaction", "order"],
        'quantity': ["quantity", "qty", "units"],
        'price': ["unitprice", "price", "unit price"],
        'customer': ["customerid", "customer id", "customer"],
        'date': ["invoicedate", "date", "transaction date"],
        'age': ["age"],
        'income': ["annual income", "income", "salary"],
        'spending': ["spending score", "spending", "score"],
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
    """Convert transactional to RFM metrics"""
    rename_dict = {
        mapping['invoice']: "InvoiceNo",
        mapping['quantity']: "Quantity",
        mapping['price']: "UnitPrice",
        mapping['customer']: "CustomerID"
    }
    
    if 'date' in mapping:
        rename_dict[mapping['date']] = "InvoiceDate"
    
    df = df.rename(columns=rename_dict)
    df["TransactionAmount"] = df["Quantity"] * df["UnitPrice"]
    df = df[df["TransactionAmount"] > 0]
    df = df[df["Quantity"] > 0]
    
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors='coerce')
        df = df.dropna(subset=["InvoiceDate"])
        max_date = df["InvoiceDate"].max()
        df["Recency"] = (max_date - df["InvoiceDate"]).dt.days
    
    agg_dict = {
        "InvoiceNo": "nunique",
        "TransactionAmount": ["sum", "mean", "max"],
        "Quantity": ["sum", "mean"]
    }
    
    if "Recency" in df.columns:
        agg_dict["Recency"] = "min"
    
    customer_metrics = df.groupby("CustomerID").agg(agg_dict).reset_index()
    customer_metrics.columns = ['_'.join(str(col)).strip('_') for col in customer_metrics.columns.values]
    customer_metrics = customer_metrics.rename(columns={
        'CustomerID_': 'CustomerID',
        'InvoiceNo_nunique': 'Frequency',
        'TransactionAmount_sum': 'Monetary',
        'TransactionAmount_mean': 'AvgTransactionValue',
        'TransactionAmount_max': 'MaxTransaction',
        'Quantity_sum': 'TotalQuantity',
        'Quantity_mean': 'AvgQuantity',
        'Recency_min': 'Recency'
    })
    
    customer_metrics = customer_metrics.fillna(0)
    return customer_metrics

@st.cache_data
def clean_data(df, numeric_cols):
    """Clean data"""
    initial_rows = len(df)
    df = df.dropna(subset=numeric_cols)
    df = df.drop_duplicates()
    
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
    
    removed_rows = initial_rows - len(df)
    return df, removed_rows

def encode_categorical(df, categorical_cols):
    """Encode categorical variables"""
    le = LabelEncoder()
    encoded_df = df.copy()
    encoding_map = {}
    
    for col in categorical_cols:
        if col in encoded_df.columns:
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            encoding_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    return encoded_df, encoding_map

def create_features(df):
    """Engineer new features"""
    new_features = []
    
    if "Annual Income (k$)" in df.columns and "Spending Score (1-100)" in df.columns:
        df["Spending_Efficiency"] = df["Spending Score (1-100)"] / (df["Annual Income (k$)"] + 1)
        new_features.append("Spending_Efficiency")
    
    if "Annual Income (k$)" in df.columns and "Age" in df.columns:
        df["Income_per_Age"] = df["Annual Income (k$)"] / (df["Age"] + 1)
        new_features.append("Income_per_Age")
    
    if "Age" in df.columns:
        df["Age_Group"] = pd.cut(df["Age"], bins=[0, 25, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5])
        df["Age_Group"] = df["Age_Group"].astype(int)
        new_features.append("Age_Group")
    
    if "Annual Income (k$)" in df.columns:
        df["Income_Group"] = pd.qcut(df["Annual Income (k$)"], q=4, labels=[1, 2, 3, 4], duplicates='drop')
        df["Income_Group"] = df["Income_Group"].astype(int)
        new_features.append("Income_Group")
    
    if all(col in df.columns for col in ["Recency", "Frequency", "Monetary"]):
        try:
            df["Recency_Score"] = pd.qcut(df["Recency"], q=5, labels=[5,4,3,2,1], duplicates='drop')
            df["Frequency_Score"] = pd.qcut(df["Frequency"], q=5, labels=[1,2,3,4,5], duplicates='drop')
            df["Monetary_Score"] = pd.qcut(df["Monetary"], q=5, labels=[1,2,3,4,5], duplicates='drop')
            df["RFM_Score"] = (df["Recency_Score"].astype(int) + df["Frequency_Score"].astype(int) + df["Monetary_Score"].astype(int))
            new_features.extend(["RFM_Score"])
        except:
            pass
    
    if "Frequency" in df.columns and "Monetary" in df.columns:
        df["Customer_Value"] = df["Frequency"] * df["Monetary"]
        new_features.append("Customer_Value")
    
    return df, new_features

@st.cache_data
def calculate_optimal_clusters(scaled_data, max_k=10):
    """Calculate metrics for K values"""
    if len(scaled_data) < 10:
        max_k = min(max_k, len(scaled_data) - 1)
    
    metrics = {'k': [], 'inertia': [], 'silhouette': [], 'davies_bouldin': [], 'calinski_harabasz': []}
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(scaled_data)
        
        if len(set(labels)) > 1:
            metrics['k'].append(k)
            metrics['inertia'].append(kmeans.inertia_)
            metrics['silhouette'].append(silhouette_score(scaled_data, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(scaled_data, labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(scaled_data, labels))
    
    return pd.DataFrame(metrics)

def perform_clustering(scaled_data, algorithm='KMeans', n_clusters=5, **kwargs):
    """Perform clustering"""
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
    """Generate cluster insights"""
    insights = []
    
    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id]
        size = len(cluster_data)
        percentage = (size / len(df)) * 100
        
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
    """Create Excel report"""
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Clustered_Data', index=False)
            
            summary_data = []
            for insight in insights:
                row = {'Cluster': insight['cluster_id'], 'Size': insight['size'], 'Percentage': f"{insight['percentage']:.2f}%"}
                for feature in features:
                    row[f'{feature}_Mean'] = insight['statistics'][feature]['mean']
                    row[f'{feature}_Median'] = insight['statistics'][feature]['median']
                summary_data.append(row)
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Cluster_Summary', index=False)
            if metrics is not None:
                metrics.to_excel(writer, sheet_name='Metrics', index=False)
        
        output.seek(0)
        return output
    except Exception as e:
        logger.error(f"Excel export failed: {str(e)}")
        return None

# ==============================================
# MAIN APPLICATION
# ==============================================

def main():
    # Professional Header
    st.markdown("""
        <div class="dashboard-header">
            <h1 class="dashboard-title">CUSTOMER SEGMENTATION ANALYTICS</h1>
            <p class="dashboard-subtitle">Enterprise Analytics & Machine Learning Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    # ==============================================
    # SIDEBAR
    # ==============================================
    
    with st.sidebar:
        st.markdown("### CONFIGURATION")
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Professional Logo
        st.markdown("""
            <div class="logo-container">
                <div class="logo-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
                    </svg>
                </div>
                <div class="logo-text">Analytics Pro</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Data Source")
        data_source = st.radio(
            "Select data source:",
            ["Upload CSV File", "Use Sample Data"],
            label_visibility="collapsed"
        )
        
        if data_source == "Upload CSV File":
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            if uploaded_file:
                df, encoding = load_data(uploaded_file)
                if df is not None:
                    st.markdown(f'<span class="status-indicator status-success"></span> Loaded ({encoding})', unsafe_allow_html=True)
                else:
                    st.stop()
            else:
                st.info("Upload a CSV file to begin")
                st.stop()
        else:
            st.markdown("#### Sample Data Settings")
            n_samples = st.slider("Number of samples:", 100, 1000, 200, 50)
            df = generate_sample_data(n_samples)
            st.markdown(f'<span class="status-indicator status-success"></span> Generated {n_samples} samples', unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Quick Stats
        st.markdown("#### Quick Statistics")
        st.markdown(f"""
            <div class="custom-card">
                <div style="color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Total Records</div>
                <div style="color: #0f172a; font-size: 1.875rem; font-weight: 700; margin-top: 0.25rem;">{len(df):,}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="custom-card">
                <div style="color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Features</div>
                <div style="color: #0f172a; font-size: 1.875rem; font-weight: 700; margin-top: 0.25rem;">{len(df.columns)}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # ==============================================
    # MAIN CONTENT
    # ==============================================
    
    # Data Overview
    st.markdown('<p class="section-header">DATA OVERVIEW</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", len(df.columns))
    with col3:
        st.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Categorical Features", len(df.select_dtypes(include=['object']).columns))
    
    with st.expander("View Dataset Preview"):
        st.dataframe(df.head(20), use_container_width=True, height=400)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Data Types**")
            st.dataframe(df.dtypes.to_frame().rename(columns={0: 'Type'}))
        with col2:
            st.markdown("**Missing Values**")
            missing_df = df.isnull().sum().to_frame().rename(columns={0: 'Missing'})
            missing_df['Percentage'] = (missing_df['Missing'] / len(df) * 100).round(2)
            st.dataframe(missing_df)
    
    # Detect transactional data
    mapping = detect_column_mapping(df)
    is_transactional = all(key in mapping for key in ['invoice', 'quantity', 'price', 'customer'])
    
    if is_transactional:
        st.info("Transactional data detected. Converting to customer-level RFM metrics...")
        with st.spinner("Processing..."):
            df = process_transactional_data(df, mapping)
        st.success("Successfully converted to RFM metrics")
    
    # Column identification
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns.tolist() if not col.lower().endswith('id')]
    categorical_cols = [col for col in df.select_dtypes(include=['object']).columns.tolist() if not col.lower().endswith('id')]
    
    # Data Cleaning
    st.markdown('<p class="section-header">DATA CLEANING & PREPROCESSING</p>', unsafe_allow_html=True)
    
    missing_before = df.isnull().sum().sum()
    with st.spinner("Cleaning data..."):
        df, removed_rows = clean_data(df, numeric_cols)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Missing Values", "0", delta=f"-{missing_before}", delta_color="normal")
    with col2:
        st.metric("Rows Removed", removed_rows)
    with col3:
        st.metric("Clean Records", f"{len(df):,}")
    with col4:
        retention_rate = (len(df) / (len(df) + removed_rows) * 100)
        st.metric("Retention Rate", f"{retention_rate:.1f}%")
    
    if categorical_cols:
        with st.spinner("Encoding categorical variables..."):
            df, encoding_map = encode_categorical(df, categorical_cols)
        st.success(f"Encoded {len(categorical_cols)} categorical column(s)")
    
    # Feature Engineering
    st.markdown('<p class="section-header">FEATURE ENGINEERING</p>', unsafe_allow_html=True)
    
    with st.spinner("Creating new features..."):
        df, new_features = create_features(df)
    
    if new_features:
        st.success(f"Created {len(new_features)} new feature(s): {', '.join(new_features)}")
        numeric_cols.extend(new_features)
        
        with st.expander("New Features Details"):
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
        st.info("No additional features created")
    
    # Feature Selection
    st.markdown('<p class="section-header">FEATURE SELECTION FOR CLUSTERING</p>', unsafe_allow_html=True)
    
    if len(numeric_cols) == 0:
        st.error("No numeric features available for clustering!")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_features = st.multiselect(
            "Select features for clustering:",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))]
        )
    with col2:
        if selected_features:
            st.info(f"{len(selected_features)} feature(s) selected")
    
    if not selected_features:
        st.warning("Please select at least one feature")
        st.stop()
    
    with st.expander("Selected Features Statistics"):
        stats_df = df[selected_features].describe().T
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
    
    # EDA
    st.markdown('<p class="section-header">EXPLORATORY DATA ANALYSIS</p>', unsafe_allow_html=True)
    
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs(["Distributions", "Correlations", "Scatter Analysis", "Box Plots"])
    
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
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.histplot(df[selected_features[idx]], kde=True, ax=ax, color='#3b82f6')
                            ax.set_title(selected_features[idx], fontsize=12, fontweight='600')
                            ax.grid(True, alpha=0.2)
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            st.pyplot(fig)
                            plt.close()
    
    with eda_tab2:
        st.markdown("### Correlation Matrix")
        if len(selected_features) >= 2:
            corr = df[selected_features].corr()
            fig = px.imshow(corr, labels=dict(color="Correlation"), color_continuous_scale='RdBu_r', title="Feature Correlation Heatmap")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least 2 features")
    
    with eda_tab3:
        st.markdown("### Scatter Plot Analysis")
        if len(selected_features) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis:", selected_features, key="eda_x")
            with col2:
                y_axis = st.selectbox("Y-axis:", selected_features, index=1 if len(selected_features) > 1 else 0, key="eda_y")
            
            if x_axis != y_axis:
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}", height=600)
                fig.update_traces(marker=dict(size=8, color='#3b82f6'))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least 2 features")
    
    with eda_tab4:
        st.markdown("### Box Plot Analysis")
        if len(selected_features) > 0:
            fig = go.Figure()
            for feature in selected_features:
                fig.add_trace(go.Box(y=df[feature], name=feature))
            fig.update_layout(title="Feature Distribution", height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature Scaling
    with st.spinner("Scaling features..."):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[selected_features])
    
    # Optimal Clusters
    st.markdown('<p class="section-header">OPTIMAL CLUSTER DETERMINATION</p>', unsafe_allow_html=True)
    
    with st.spinner("Calculating metrics..."):
        metrics_df = calculate_optimal_clusters(scaled_data, max_k=10)
    
    if len(metrics_df) == 0:
        st.error("Unable to calculate metrics")
        st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics_df['k'], y=metrics_df['inertia'], mode='lines+markers', line=dict(color='#3b82f6', width=3)))
        fig.update_layout(title="Elbow Method", xaxis_title="Clusters (K)", yaxis_title="WCSS", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics_df['k'], y=metrics_df['silhouette'], mode='lines+markers', line=dict(color='#10b981', width=3)))
        fig.update_layout(title="Silhouette Score", xaxis_title="Clusters (K)", yaxis_title="Score", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View All Metrics"):
        st.dataframe(metrics_df.style.format({'inertia': '{:.2f}', 'silhouette': '{:.4f}', 'davies_bouldin': '{:.4f}', 'calinski_harabasz': '{:.2f}'}), use_container_width=True)
        
        optimal_k = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']
        st.success(f"Suggested optimal K: {int(optimal_k)}")
    
    # Clustering
    st.markdown('<p class="section-header">APPLY CLUSTERING ALGORITHM</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        algorithm = st.selectbox("Algorithm:", ["KMeans", "Hierarchical", "DBSCAN"])
    with col2:
        if algorithm in ["KMeans", "Hierarchical"]:
            n_clusters = st.slider("Clusters:", 2, 10, 5)
        else:
            eps = st.slider("EPS:", 0.1, 2.0, 0.5, 0.1)
    with col3:
        if algorithm == "DBSCAN":
            min_samples = st.slider("Min Samples:", 2, 20, 5)
    
    with st.spinner(f"Applying {algorithm}..."):
        if algorithm in ["KMeans", "Hierarchical"]:
            labels, model = perform_clustering(scaled_data, algorithm, n_clusters)
        else:
            labels, model = perform_clustering(scaled_data, algorithm, eps=eps, min_samples=min_samples)
    
    df['Cluster'] = labels
    
    if len(set(labels)) > 1:
        sil_score = silhouette_score(scaled_data, labels)
        db_score = davies_bouldin_score(scaled_data, labels)
        ch_score = calinski_harabasz_score(scaled_data, labels)
        
        st.markdown("### Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Clusters Found", len(set(labels)))
        with col2:
            st.metric("Silhouette Score", f"{sil_score:.3f}")
        with col3:
            st.metric("Davies-Bouldin", f"{db_score:.3f}")
        with col4:
            st.metric("Calinski-Harabasz", f"{ch_score:.1f}")
    else:
        st.warning("Only one cluster detected. Adjust parameters.")
    
    # Visualizations
    st.markdown('<p class="section-header">CLUSTER VISUALIZATION</p>', unsafe_allow_html=True)
    
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["2D Scatter", "3D Interactive", "PCA Projection"])
    
    with viz_tab1:
        if len(selected_features) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("X-axis:", selected_features, key="viz_x")
            with col2:
                y_feature = st.selectbox("Y-axis:", selected_features, index=1 if len(selected_features) > 1 else 0, key="viz_y")
            
            if x_feature != y_feature:
                fig = px.scatter(df, x=x_feature, y=y_feature, color='Cluster', title=f"Segments: {y_feature} vs {x_feature}", height=600)
                fig.update_traces(marker=dict(size=10))
                st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab2:
        if len(selected_features) >= 3:
            col1, col2, col3 = st.columns(3)
            with col1:
                dim1 = st.selectbox("1st Dimension:", selected_features, key="3d_x")
            with col2:
                dim2 = st.selectbox("2nd Dimension:", selected_features, index=1 if len(selected_features) > 1 else 0, key="3d_y")
            with col3:
                dim3 = st.selectbox("3rd Dimension:", selected_features, index=2 if len(selected_features) > 2 else 0, key="3d_z")
            
            if len({dim1, dim2, dim3}) == 3:
                fig = px.scatter_3d(df, x=dim1, y=dim2, z=dim3, color='Cluster', title="3D Visualization", height=700)
                st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab3:
        pca = PCA(n_components=min(3, len(selected_features)))
        pca_data = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])
        pca_df['Cluster'] = df['Cluster'].values
        
        col1, col2, col3 = st.columns(3)
        for i, col in enumerate([col1, col2, col3]):
            if i < len(pca.explained_variance_ratio_):
                with col:
                    st.metric(f"PC{i+1} Variance", f"{pca.explained_variance_ratio_[i]:.1%}")
        
        if pca_data.shape[1] >= 2:
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title="PCA Projection", height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.markdown('<p class="section-header">CLUSTER INSIGHTS & ANALYSIS</p>', unsafe_allow_html=True)
    
    insights = generate_cluster_insights(df, selected_features)
    cluster_counts = df['Cluster'].value_counts().sort_index()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Pie(labels=[f'Cluster {i}' for i in cluster_counts.index], values=cluster_counts.values, hole=0.4))
        fig.update_layout(title="Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Summary")
        summary_data = []
        for insight in insights:
            row = {'Cluster': f"Cluster {insight['cluster_id']}", 'Size': insight['size'], 'Percentage': f"{insight['percentage']:.2f}%"}
            for feature in selected_features[:3]:
                row[f'{feature} (Avg)'] = f"{insight['statistics'][feature]['mean']:.2f}"
            summary_data.append(row)
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, height=350)
    
    # Export
    st.markdown('<p class="section-header">EXPORT RESULTS</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, f"segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", use_container_width=True)
    
    with col2:
        with st.spinner("Generating Excel..."):
            excel_report = create_downloadable_report(df, insights, selected_features, metrics_df)
        if excel_report:
            st.download_button("Download Excel Report", excel_report, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    
    with col3:
        summary_csv = pd.DataFrame([{'Metric': 'Records', 'Value': len(df)}, {'Metric': 'Clusters', 'Value': len(set(labels))}]).to_csv(index=False).encode('utf-8')
        st.download_button("Download Summary", summary_csv, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", use_container_width=True)
    
    st.success("Analysis Completed Successfully!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0; color: #64748b;'>
            <p style='font-size: 0.9rem; font-weight: 600;'>Customer Segmentation Analytics v2.0</p>
            <p style='font-size: 0.75rem; color: #94a3b8;'>© 2026 Analytics Pro | Enterprise Machine Learning Platform</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred in the application")
        with st.expander("Error Details"):
            st.code(f"{type(e).__name__}: {str(e)}")
