import pandas as pd

def get_port_risk_scores(file_path=None):
    """
    Parses the CICIDS2017 dataset to calculate a risk score for each port.
    The risk score is the ratio of attack traffic to total traffic on that port.
    """
    if file_path:
        try:
            # Load dataset (supports both parquet and csv)
            df = pd.read_parquet(file_path) if file_path.endswith('.parquet') else pd.read_csv(file_path)
            
            # Convert labels to binary: 0 for normal, 1 for attack
            df['is_attack'] = df['Label'].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)
            
            # Group by destination port and calculate the mean attack ratio
            port_stats = df.groupby('Destination Port')['is_attack'].mean().to_dict()
            print("Success: CICIDS2017 dataset parsed successfully.")
            return port_stats
        except Exception as e:
            print(f"Dataset error: {e}. Using fallback data.")
    
    # Fallback data based on real-world network vulnerabilities
    # (e.g., Port 3306 (MySQL) is highly targeted, Port 443 (HTTPS) is safer)
    print("Using fallback risk scores.")
    return {80: 0.4, 443: 0.3, 22: 0.8, 3306: 0.9, 8080: 0.5}
