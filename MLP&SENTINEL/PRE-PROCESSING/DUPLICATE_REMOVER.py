import pandas as pd

# Load the CSV file
file_path = '/Users/matteo_crespinjouan/Desktop/NHM/SENTINEL_MLP_ENDEAVOUR/SENTINEL_FINAL/vegaABC.csv'
df = pd.read_csv(file_path)

# Print the total number of rows
total_rows = len(df)
print(f"Total number of rows: {total_rows}")

# Count duplicates
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# Remove duplicates
df_cleaned = df.drop_duplicates()

# Save the cleaned DataFrame to a new CSV
output_path = '/Users/matteo_crespinjouan/Desktop/NHM/SENTINEL_MLP_ENDEAVOUR/SENTINEL_FINAL/vegaABC_cleaned.csv'
df_cleaned.to_csv(output_path, index=False)

print(f"Cleaned CSV saved to {output_path}")