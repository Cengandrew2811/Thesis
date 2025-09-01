import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from patient import Patient

# Load Excel file
df = pd.read_excel("Copy of mmc1[66].xlsx")
exclude_ids = ["580.1", "613.3", "614.1"]
df["ID"] = df["ID"].astype(str)
df = df[~df["ID"].isin(exclude_ids)]
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
df = df[df["Age"] >= 18]

# Check required columns exist
required_columns = [
    'ID',
    'Body weight',
    'Height',
    'Gender: 1, male; 2, female',
    'Age',
    'hematocrit',
    'serum creatinine'
]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in Excel. Please check your file headers.")

# Lists to collect final patient-level results
patient_ids = []
avg_gfr_values = []
gfr_classes = []
weights = []

# Group by patient ID
for patient_id, group in df.groupby('ID'):
    gfrs = []
    weight = None 
    
    # Loop over all rows for this patient
    for _, row in group.iterrows():
        try:
            # Convert serum creatinine safely
            serum_creatinine = float(row['serum creatinine']) * 0.0113  # µmol/L → mg/dL

            # Convert hematocrit to fraction
            hematocrit_fraction = float(row['hematocrit']) / 100.0

            # Convert weight to float
            weight = float(row['Body weight'])

            # Create Patient instance
            patient = Patient(
                row['Gender: 1, male; 2, female'],
                weight,
                row['Height'],
                row['Age'],
                hematocrit_fraction,
                serum_creatinine
            )

            # Calculate GFR for this measurement
            gfr = patient.compute_gfr_ckd_epi()
            gfrs.append(gfr)

        except (ValueError, TypeError):
            print(f"Invalid data for patient {patient_id}. Skipping this measurement.")
            continue
    
    if len(gfrs) == 0:
        print(f"No valid GFRs for patient {patient_id}. Skipping patient.")
        continue

    # Calculate average GFR for patient
    avg_gfr = np.median(gfrs)
    
    # Classify based on average GFR
    if avg_gfr >= 90:
        gfr_class = 'Normal'
    elif avg_gfr >= 60:
        gfr_class = 'Mild impairment'
    elif avg_gfr >= 30:
        gfr_class = 'Moderate to severe impairment'
    else:
        gfr_class = 'Severe impairment'
    
    # Save results
    patient_ids.append(patient_id)
    avg_gfr_values.append(avg_gfr)
    gfr_classes.append(gfr_class)
    weights.append(weight)

# Create summary DataFrame
classification_df = pd.DataFrame({
    'PatientID': patient_ids,
    'Average_GFR': avg_gfr_values,
    'GFR_Classification': gfr_classes,
    'Weight': weights
})

#Calculate tertiles from weight
low_cutoff, high_cutoff = classification_df['Weight'].quantile([1/3, 2/3])

# Define weight category
def classify_weight(w):
    if w <= low_cutoff:
        return 'Low'
    elif w<= high_cutoff:
        return 'Medium'
    else:
        return 'High'

classification_df['Weight_Group'] = classification_df['Weight'].apply(classify_weight)
#Define final group classification
def assign_group(row):
    gfr = row['GFR_Classification']
    wt = row['Weight_Group']
    if gfr == 'Normal':
        return {'Low':'A', 'Medium':'B', 'High':'C'}[wt]
    else:
        return 'D'

classification_df['Final_Group'] = classification_df.apply(assign_group, axis=1)

# Display summary
print("Patient Classification Summary:")
print(classification_df)

# Save to CSV
classification_df.to_csv("patient_7group_classification.csv", index=False)
print("\nClassification saved to 'patient_7group_classification.csv'.")


# Count patients in each classification category
group_counts = classification_df['Final_Group'].value_counts().sort_index()
total = group_counts.sum()

# Create pie chart
# Mapping of groups to descriptions
group_descriptions = {
    'A': 'Normal GFR + Low weight',
    'B': 'Normal GFR + Medium weight',
    'C': 'Normal GFR + High weight',
    'D': 'Impaired GFR (any weight)'
}

# Pie chart
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    group_counts,
    labels=group_counts.index,
    autopct=lambda p: f"{int(round(p * total / 100))} patients\n({p:.1f}%)",
    startangle=140,
    wedgeprops={'edgecolor': 'black'}
)

# Add legend with full group descriptions
plt.legend(
    wedges,
    [f"Group {k}: {v}" for k, v in group_descriptions.items()],
    title="Group Definitions",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.01)
)

plt.title('Patient Distribution by GFR and Weight Group (4 Groups: A–D)')
plt.tight_layout()
plt.show()
