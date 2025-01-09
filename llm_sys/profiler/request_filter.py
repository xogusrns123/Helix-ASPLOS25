import pandas as pd

# Read the merged_sorted.csv file
merged_df = pd.read_csv("merged_sorted.csv")

# Filter rows where request_id is 9
filtered_df = merged_df[merged_df['request_id'] == 9]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv("filtered_request_id_9.csv", index=False)

print("Filtered data has been saved to 'filtered_request_id_9.csv'")
