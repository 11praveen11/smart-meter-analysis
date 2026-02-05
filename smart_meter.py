import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os


# --- 1. Data Loading & Cleaning ---
def load_and_clean_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Parse dates
    df['READING_DATETIME'] = pd.to_datetime(df['READING_DATETIME'], dayfirst=True)

    return df


# --- 2. Data Quality Checks ---
def check_data_quality(df):
    print("\n--- Data Quality Report ---")
    print(f"Total Rows: {len(df)}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")

    neg_values = df[df['GENERAL_SUPPLY_KWH'] < 0].shape[0]
    zeros = df[df['GENERAL_SUPPLY_KWH'] == 0].shape[0]
    print(f"Negative Values: {neg_values}")
    print(f"Zero Values: {zeros} ({zeros / len(df) * 100:.2f}%)")
    print("---------------------------")


# --- 3. Analysis Functions ---
def analyze_temporal_patterns(df):
    # Feature Engineering
    df['Hour'] = df['READING_DATETIME'].dt.hour
    df['DayOfWeek'] = df['READING_DATETIME'].dt.day_name()
    df['MonthNo'] = df['READING_DATETIME'].dt.month

    # Aggregations
    daily_profile = df.groupby('Hour')['GENERAL_SUPPLY_KWH'].mean()

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_profile = df.groupby('DayOfWeek')['GENERAL_SUPPLY_KWH'].mean().reindex(days_order)

    monthly_profile = df.groupby('MonthNo')['GENERAL_SUPPLY_KWH'].mean()

    return daily_profile, weekly_profile, monthly_profile


def perform_clustering(df, n_clusters=3):
    print("\nPerforming Customer Clustering...")
    # Pivot: Customer vs Hour
    pivot_df = df.pivot_table(index='CUSTOMER_ID', columns='Hour', values='GENERAL_SUPPLY_KWH', aggfunc='mean')
    pivot_df = pivot_df.fillna(0)

    # Normalize
    scaler = MinMaxScaler()
    pivot_normalized = pd.DataFrame(scaler.fit_transform(pivot_df.T).T,
                                    index=pivot_df.index,
                                    columns=pivot_df.columns)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pivot_normalized)

    pivot_normalized['Cluster'] = clusters
    return pivot_normalized


# --- 4. Visualization & Saving Functions ---
def save_temporal_patterns(daily, weekly, monthly):
    """Generates and saves Figure 1"""
    print("Generating temporal_patterns.png...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Daily
    daily.plot(ax=axes[0], marker='o', color='royalblue')
    axes[0].set_title('Average Daily Consumption Pattern')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Avg KWH')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Weekly
    weekly.plot(kind='bar', ax=axes[1], color='skyblue')
    axes[1].set_title('Average Weekly Consumption Pattern')
    axes[1].set_ylabel('Avg KWH')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Seasonal
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly.index = months
    monthly.plot(kind='bar', ax=axes[2], color='lightgreen')
    axes[2].set_title('Average Seasonal Consumption Pattern')
    axes[2].set_ylabel('Avg KWH')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    # Save to file
    plt.savefig('temporal_patterns.png', dpi=300)
    plt.close()  # Close to free memory
    print("Saved temporal_patterns.png")


def save_customer_clusters(cluster_df):
    """Generates and saves Figure 2"""
    print("Generating customer_clusters.png...")
    n_clusters = cluster_df['Cluster'].nunique()
    cluster_counts = cluster_df['Cluster'].value_counts()

    fig, axes = plt.subplots(1, n_clusters, figsize=(18, 5), sharey=True)

    for i in range(n_clusters):
        cluster_data = cluster_df[cluster_df['Cluster'] == i].drop('Cluster', axis=1)

        # Plot individual customer lines (lightly)
        for customer in cluster_data.index:
            axes[i].plot(cluster_data.loc[customer], color='gray', alpha=0.1)

        # Plot mean line (bold)
        axes[i].plot(cluster_data.mean(), color='red', linewidth=2, label='Cluster Mean')

        axes[i].set_title(f'Cluster {i} (n={cluster_counts.get(i, 0)})')
        axes[i].set_xlabel('Hour')
        if i == 0:
            axes[i].set_ylabel('Normalized Consumption')
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend()

    plt.tight_layout()
    # Save to file
    plt.savefig('customer_clusters.png', dpi=300)
    plt.close()
    print("Saved customer_clusters.png")


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the file name matches your upload
    file_path = 'Smart Meter Interval Reading.csv'

    if os.path.exists(file_path):
        # 1. Load
        df = load_and_clean_data(file_path)

        # 2. Quality
        check_data_quality(df)

        # 3. Analyze
        daily_profile, weekly_profile, monthly_profile = analyze_temporal_patterns(df)
        clustered_data = perform_clustering(df)

        # 4. Save Figures
        save_temporal_patterns(daily_profile, weekly_profile, monthly_profile)
        save_customer_clusters(clustered_data)

        print("\nAll tasks completed successfully. Images saved.")
    else:
        print(f"Error: File '{file_path}' not found.")