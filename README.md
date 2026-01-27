# Customer Segmentation Project

A Streamlit-based application for customer segmentation using clustering algorithms.

## Features

- Data upload via CSV file
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Clustering with K-Means
- Visualizations and dashboards

## Installation (Local)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage (Local)

Run the Streamlit app:
```bash
streamlit run task1_inten.py
```

Open the provided URL in your browser and upload a CSV file with customer data (expected columns: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)).

## Deployment on Streamlit Cloud

1. Push this repository to GitHub (public repo).
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Sign in with GitHub.
4. Click "New app" and select this repository.
5. Set the main file path to `task1_inten.py`.
6. Deploy!

The app will be hosted online and accessible via a public URL.

## Dataset

If no file is uploaded, the app uses sample data for demonstration.

For real data, download from Kaggle: "Customer Segmentation Dataset" or Google Dataset Search.

## Troubleshooting

- Ensure all dependencies are installed.
- If Streamlit doesn't start, check your Python environment.
- For deployment issues, check Streamlit Cloud logs.