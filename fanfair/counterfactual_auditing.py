import pandas as pd
import numpy as np
import warnings
import os
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Silence technical warnings
warnings.filterwarnings("ignore")

def run_counterfactual_shift_audit_FanFAIR(
    X, 
    y, 
    sensitive_cols, 
    dataset_name="Generic_Dataset", 
    feature_mappings=None, 
    target_mapping=None,
    model=None
):
    """
    Runs a counterfactual audit on a given dataset to find prediction shifts caused by changing sensitive attributes.

    Parameters:
        - X: pd.DataFrame of features (must include sensitive_cols)
        - y: pd.Series or np.array of target labels
        - sensitive_cols: List of column names in X that are considered sensitive (e.g., ["Race", "Gender"])
        - dataset_name: String name for the dataset (used in report titles and file paths)
        - feature_mappings: Optional dict of dicts to map raw feature values to readable labels (e.g., {"Race": {0: "White", 1: "Black"}})
        - target_mapping: Optional dict to map raw target values to readable labels
        - model: Pre-trained model (if None, a default RandomForestClassifier will be trained)
    
    Returns:
        - shifts_df: DataFrame containing all detected counterfactual shifts with details
    """
    
    if feature_mappings is None: feature_mappings = {}
    if target_mapping is None: target_mapping = {}
    
    def get_val_label(col, val):
        if col in feature_mappings and val in feature_mappings[col]:
            return str(feature_mappings[col][val])
        return str(val)

    def get_target_label(val):
        return str(target_mapping.get(val, val))

    df_audit_X = X.copy()
    
    if 'Record_ID' not in df_audit_X.columns:
        df_audit_X = df_audit_X.reset_index(names=['Record_ID'])
    
    domains = {col: sorted(X[col].unique()) for col in sensitive_cols}
    
    if model is None:
        print("No model provided. Training a default RandomForestClassifier...")
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
    
    orig_preds = model.predict(X)
    accuracy = accuracy_score(y, orig_preds)
    print(f"Model Accuracy on Audit Data: {accuracy:.2%}")

    all_shifts = []
    trials = 0
    keys = sensitive_cols
    
    print(f"Starting Generic counterfactual Audit on {len(df_audit_X)} samples auditing {keys}...")
    
    value_lists = [domains[col] for col in keys]
    all_possible_combos = list(itertools.product(*value_lists))

    for idx in tqdm(df_audit_X.index, desc="Searching shifts"):
        row = df_audit_X.loc[idx]
        
        feat_row = row.drop('Record_ID').to_frame().T if 'Record_ID' in row else row.to_frame().T
        
        original_pred = orig_preds[idx]
        
        orig_sens_vals = {col: row[col] for col in keys}
        orig_profile_str = " ".join([get_val_label(col, orig_sens_vals[col]) for col in keys])
        p_lab_orig = get_target_label(original_pred)

        # Test all possible counterfactual combinations
        for combo in all_possible_combos:
            
            # Determine which specific columns are changing in this iteration
            changed_cols = [keys[i] for i in range(len(keys)) if combo[i] != orig_sens_vals[keys[i]]]
            
            if not changed_cols:
                continue # Skip if nothing changed (this is the original record)
                
            trials += 1
            test_row = feat_row.copy()
            for i, col in enumerate(keys):
                test_row[col] = combo[i]
                
            new_pred = model.predict(test_row)[0]
            
            if new_pred != original_pred:
                p_lab_new = get_target_label(new_pred)
                
                if len(changed_cols) == 1:
                    shift_type = f"Pure {changed_cols[0]} shift"
                elif len(changed_cols) == len(keys) and len(keys) > 1:
                    shift_type = f"Combined {'-'.join(keys)} shift"
                else:
                    shift_type = f"Combined {'-'.join(changed_cols)} shift"
                
                shiftped_profile_str = " ".join([get_val_label(keys[i], combo[i]) for i in range(len(keys))])
                
                # Create a readable string of what changed (e.g., "White → Black")
                change_desc_list = [f"{get_val_label(col, orig_sens_vals[col])} → {get_val_label(col, combo[i])}" for i, col in enumerate(keys) if col in changed_cols]
                
                all_shifts.append({
                    "Record_ID": row.get('Record_ID', idx),
                    "Type": shift_type,
                    "Original_Profile": orig_profile_str,
                    "shifted_Profile": shiftped_profile_str,
                    "Changed_Cols": ", ".join(changed_cols),
                    "Change_Details": " | ".join(change_desc_list),
                    "Result": f"{p_lab_orig} → {p_lab_new}"
                })

    output_dir = f"audits/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving audit report and shifts to '{output_dir}/' ...")
    
    shifts_df = pd.DataFrame(all_shifts)
    if not shifts_df.empty:
        shifts_df.sort_values(by=['Type', 'Record_ID'], inplace=True)
        shifts_df.to_csv(f"{output_dir}/counterfactual_shifts.csv", index=False)

    with open(f"{output_dir}/counterfactual_audit.md", "w", encoding="utf-8") as f:
        f.write(f"# counterfactual Audit Report: {dataset_name}\n\n")
        f.write(f"- **Sensitive Attributes Audited:** `{', '.join(sensitive_cols)}`\n")
        f.write(f"- **Audit Samples:** `{len(df_audit_X)}`\n")
        f.write(f"- **Accuracy:** `{accuracy:.2%}`\n")
        f.write(f"- **Total shifts Detected:** `{len(shifts_df)}` on `{trials}` counterfactual tests\n\n")

        if shifts_df.empty:
            f.write("**No counterfactual shifts detected.** \n")
            print("\n[SUCCESS] counterfactual audit completed. No shifts detected.")
            return shifts_df

        f.write("### shift Distribution\n")
        f.write(shifts_df['Type'].value_counts().to_markdown() + "\n\n")

        f.write("### Key counterfactual shift Insights (Top Occurrences)\n\n")
        def get_wilson_interval(successes, trials, confidence=0.95):
            if trials == 0: return 0, 0
            from scipy import stats # Or use the manual formula below if you prefer no extra imports
            z = 1.96 # Approx for 95%
            p_hat = successes / trials
            denominator = 1 + z**2/trials
            centre_adj = p_hat + z**2/(2*trials)
            step = z * np.sqrt((p_hat*(1-p_hat)/trials) + (z**2/(4*trials**2)))
            lower = max(0, (centre_adj - step) / denominator)
            upper = min(1, (centre_adj + step) / denominator)
            return lower * 100, upper * 100
        
        # 1. Pre-calculate the baseline candidate counts (The "Attempts")
        orig_states = pd.DataFrame({
            'Original_Profile': df_audit_X.apply(lambda r: " ".join([get_val_label(col, r[col]) for col in keys]), axis=1),
            'Orig_Pred': [get_target_label(p) for p in orig_preds]
        })
        baseline_counts = orig_states.groupby(['Original_Profile', 'Orig_Pred']).size().to_dict()

        total_shifts = len(shifts_df)
        type_counts = shifts_df['Type'].value_counts().to_dict()
        
        # Group identical shifts to count how often they occur
        shift_counts = shifts_df.groupby(
            ['Type', 'Original_Profile', 'shifted_Profile', 'Result', 'Change_Details'],
            as_index=False
        ).size()
        
        # Rename the 'size' column to 'Count' to match the logic
        shift_counts = shift_counts.rename(columns={'size': 'Count'})
        
        # Sort by the most frequent occurrences
        shift_counts = shift_counts.sort_values(by=['Count', 'Type'], ascending=False)
        total_shifts_to_show = min(1000, len(shift_counts)) 
        for _, row in shift_counts.head(total_shifts_to_show).iterrows():
            res_split = str(row['Result']).split(' → ')
            from_pred = res_split[0].strip() if len(res_split) > 1 else "Unknown"
            to_pred = res_split[1].strip() if len(res_split) > 1 else "Unknown"
            
            count = row['Count']
            shift_type = row['Type']
            attempts = baseline_counts.get((row['Original_Profile'], from_pred), 0)
            
            shift_rate = (count / attempts) * 100 if attempts > 0 else 0
            ci_low, ci_high = get_wilson_interval(count, attempts)
            
            pct_total = (count / total_shifts) * 100
            total_for_this_type = type_counts.get(shift_type, 0)
            pct_type = (count / total_for_this_type) * 100 if total_for_this_type > 0 else 0
            
            # Construct the enhanced string
            impact_str = (
                f"- **{shift_type}**: Out of {attempts} patients with profile **{row['Original_Profile']}** originally predicted **{from_pred}**, "
                f"changing to **{row['shifted_Profile']}** ({row['Change_Details']}) caused a prediction change to **{to_pred}** "
                f"in **{count} cases ({shift_rate:.1f}% | 95% CI: [{ci_low:.1f}%, {ci_high:.1f}%])**. \n"
                f"  *(Impact: {pct_total:.1f}% of all shifts | {pct_type:.1f}% of {shift_type}s).* \n\n"
            )
            f.write(impact_str)

        if len(shift_counts) > total_shifts_to_show:
            f.write(f"\n*...and {len(shift_counts) - total_shifts_to_show} other less frequent combinations omitted for brevity.*\n")
        
        f.write("\n---\n")

    print(f"\n[SUCCESS] counterfactual audit completed. Total shifts: {len(shifts_df)} on {trials} tests.")
    return shifts_df

if __name__ == "__main__":
    print("This script is meant to be imported and run with your dataset. \n")
    print("If you run this file directly, it will execute a demo audit on synthetic data to show you how it works. \n")
    print("Generating synthetic data and running audit...\n")

    # Fix random seed for reproducibility and setting a number of samples for the synthetic dataset
    np.random.seed(42)
    n_samples = 300

    # Regular features
    age = np.random.randint(18, 70, n_samples)
    income = np.random.randint(30000, 150000, n_samples)

    # Sensitive features
    # Race: 0=White, 1=Black, 2=Asian
    # Gender: 0=Male, 1=Female
    race = np.random.choice([0, 1, 2], n_samples)
    gender = np.random.choice([0, 1], n_samples)

    X_test = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Race': race,
        'Gender': gender
    })

    # We create a synthetic target variable
    # We inject a strong bias so the model is forced to rely on Race and Gender.
    # Being White (0) and Male (0) artificially boosts the chance of approval.
    base_score = (age / 70) + (income / 150000)
    bias_boost = np.where((race == 0) & (gender == 0), 0.6, 0.0) 
    bias_boost += np.where((race == 0) & (gender == 1), 0.3, 0.0)

    # 1 = Approved, 0 = Denied
    y_test = ((base_score + bias_boost) > 1.2).astype(int) 

    # Define the mappings for the synthetic dataset
    feat_maps = {
        'Race': {0: 'White', 1: 'Black', 2: 'Asian'},
        'Gender': {0: 'Male', 1: 'Female'}
    }

    # Define the mapping for the target variable
    targ_map = {
        0: 'Denied',
        1: 'Approved'
    }

    # Run the audit on the synthetic dataset to demonstrate functionality
    if __name__ == "__main__":
        print("Generating synthetic data and running audit...\n")
        
        shifts_dataframe = run_counterfactual_shift_audit_FanFAIR(
            X=X_test, 
            y=y_test, 
            sensitive_cols=['Race', 'Gender'], 
            dataset_name="Synthetic_Loan_Data", 
            feature_mappings=feat_maps, 
            target_mapping=targ_map
        )
        
        print("\nCheck the 'audits/Synthetic_Loan_Data/' folder for your markdown report and CSV!")

    