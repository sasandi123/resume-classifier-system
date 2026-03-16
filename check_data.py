import pandas as pd

# Load the dataset
df = pd.read_csv('data/raw/Resume.csv')

print("="*70)
print(" "*20 + "DATASET ANALYSIS")
print("="*70)

print(f"\n Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

print(f"\n Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i}. {col}")

print(f"\n Data Types:")
print(df.dtypes)

print(f"\n Missing Values:")
print(df.isnull().sum())

print(f"\n Category Distribution:")
print(df['Category'].value_counts())

print(f"\nSample Resume (first 500 characters):")
print("-" * 70)
print(df['Resume_str'].iloc[0][:500] + "...")
print("-" * 70)

print(f"\n Statistics:")
print(f"   - Total Resumes: {len(df)}")
print(f"   - Unique Categories: {df['Category'].nunique()}")
print(f"   - Average Resume Length: {df['Resume_str'].str.len().mean():.0f} characters")
print(f"   - Min Resume Length: {df['Resume_str'].str.len().min()} characters")
print(f"   - Max Resume Length: {df['Resume_str'].str.len().max()} characters")

print("\n" + "="*70)
print(" Dataset loaded successfully!")
print("="*70)

# Check if we have enough data for each category
print(f"\n⚠  Categories with few samples (< 20):")
low_count = df['Category'].value_counts()[df['Category'].value_counts() < 20]
if len(low_count) > 0:
    print(low_count)
else:
    print("   All categories have sufficient samples!")