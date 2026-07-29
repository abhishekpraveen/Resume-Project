import pandas as pd

# Load the original dataset
df = pd.read_csv("Resume.csv")

# Select 5000 random rows
df_small = df.sample(n=5000, random_state=42)

# Save the smaller dataset
df_small.to_csv("Resume_small.csv", index=False)

print("Small dataset created successfully!")
print(df_small.shape)