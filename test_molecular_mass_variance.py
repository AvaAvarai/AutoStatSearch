#!/usr/bin/env python3
"""
Statistical test to determine if amino acid properties (from suffix.csv) correlate with 
variance in interface bonds loss.

Tests for each property:
1. Pearson correlation between property and variance
2. Spearman rank correlation (non-parametric)
3. Comparison of high vs low property groups (t-test and Mann-Whitney U)
4. Linear regression analysis

Usage:
    python test_molecular_mass_variance.py
"""

import glob
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
from sklearn.linear_model import LinearRegression
import os

def load_suffix_properties(suffix_file='suffix.csv'):
    """Load all property data from suffix.csv."""
    df = pd.read_csv(suffix_file)
    
    # Get all columns except 'Amino Acid'
    property_cols = [col for col in df.columns if col != 'Amino Acid']
    
    # Create dictionary mapping property name to {amino_acid: value}
    properties_dict = {}
    for col in property_cols:
        properties_dict[col] = dict(zip(df['Amino Acid'], df[col]))
    
    return properties_dict

def calculate_variance_per_aa(interface_bonds_file):
    """
    Calculate variance of interface bonds loss for each amino acid.
    
    Returns:
        Dictionary mapping amino acid code to variance
    """
    df = pd.read_csv(interface_bonds_file)
    
    if 'class' not in df.columns:
        return None
    
    # Get all numeric columns (exclude 'class')
    numeric_cols = [col for col in df.columns if col != 'class']
    
    variance_dict = {}
    
    for _, row in df.iterrows():
        aa_code = row['class']
        # Extract all numeric values for this amino acid
        values = []
        for col in numeric_cols:
            val = row[col]
            if pd.notna(val) and val != '':
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    continue
        
        if len(values) > 1:  # Need at least 2 values to calculate variance
            variance = np.var(values)
            variance_dict[aa_code] = variance
        elif len(values) == 1:
            variance_dict[aa_code] = 0.0  # Single value has zero variance
        else:
            variance_dict[aa_code] = np.nan
    
    return variance_dict

def perform_statistical_tests_for_property(property_data, property_name, variance_data, protein_name):
    """
    Perform statistical tests on the relationship between a property and variance.
    
    Args:
        property_data: Dictionary of amino acid -> property value
        property_name: Name of the property being tested
        variance_data: Dictionary of amino acid -> variance
        protein_name: Name of the protein for reporting
    """
    # Create combined dataframe
    combined_data = []
    for aa in property_data.keys():
        if aa in variance_data and not np.isnan(variance_data[aa]):
            prop_val = property_data[aa]
            if pd.notna(prop_val):  # Check if property value is valid
                combined_data.append({
                    'Amino_Acid': aa,
                    'Property': prop_val,
                    'Variance': variance_data[aa]
                })
    
    if len(combined_data) < 3:
        return None
    
    df = pd.DataFrame(combined_data)
    df = df.sort_values('Property')
    
    # Test 1: Pearson correlation
    corr, p_value = pearsonr(df['Property'], df['Variance'])
    
    # Test 2: Spearman rank correlation
    spearman_corr, spearman_p = spearmanr(df['Property'], df['Variance'])
    
    # Test 3: High vs Low groups
    median_prop = df['Property'].median()
    high_group = df[df['Property'] >= median_prop]['Variance']
    low_group = df[df['Property'] < median_prop]['Variance']
    
    t_p = np.nan
    u_p = np.nan
    try:
        _, t_p = ttest_ind(high_group, low_group)
        _, u_p = mannwhitneyu(high_group, low_group, alternative='two-sided')
    except:
        pass
    
    # Test 4: Linear regression
    X = df[['Property']].values
    y = df['Variance'].values
    
    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)
    
    n = len(df)
    p = 1
    ss_res = np.sum((y - model.predict(X))**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    if ss_res > 0:
        f_stat = ((ss_tot - ss_res) / p) / (ss_res / (n - p - 1))
        f_p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
    else:
        f_p_value = 0.0
    
    return {
        'protein': protein_name,
        'property': property_name,
        'n': len(df),
        'pearson_r': corr,
        'pearson_p': p_value,
        'spearman_rho': spearman_corr,
        'spearman_p': spearman_p,
        'high_group_mean': high_group.mean(),
        'low_group_mean': low_group.mean(),
        't_p': t_p,
        'u_p': u_p,
        'r_squared': r_squared,
        'f_p': f_p_value,
        'slope': model.coef_[0],
        'intercept': model.intercept_
    }

def main():
    """Main function to process all interface bonds loss files."""
    # Load all property data from suffix.csv
    print("Loading property data from suffix.csv...")
    properties_dict = load_suffix_properties('suffix.csv')
    property_names = list(properties_dict.keys())
    print(f"Loaded {len(property_names)} properties:")
    for prop in property_names:
        print(f"  - {prop}")
    
    # Find all interface bonds loss CSV files
    pattern = '*_interface_bonds_loss.csv'
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"\nFound {len(files)} interface bonds loss CSV files:")
    for f in files:
        print(f"  - {f}")
    
    # Process each file and property combination
    all_results = []
    
    for filename in files:
        protein_name = os.path.basename(filename).replace('_interface_bonds_loss.csv', '')
        
        print(f"\n\n{'='*80}")
        print(f"Processing {protein_name}")
        print(f"{'='*80}")
        
        variance_data = calculate_variance_per_aa(filename)
        
        if variance_data is None:
            print(f"  Warning: Could not process {filename}")
            continue
        
        print(f"  Calculated variance for {len(variance_data)} amino acids")
        
        # Test each property
        for property_name in property_names:
            property_data = properties_dict[property_name]
            result = perform_statistical_tests_for_property(
                property_data, property_name, variance_data, protein_name
            )
            if result:
                all_results.append(result)
    
    # Summary across all proteins and properties
    if all_results:
        print(f"\n\n{'='*80}")
        print("SUMMARY ACROSS ALL PROTEINS AND PROPERTIES")
        print(f"{'='*80}")
        
        results_df = pd.DataFrame(all_results)
        
        # Save detailed results
        results_df.to_csv('all_properties_variance_test_results.csv', index=False)
        print(f"\nDetailed results saved to: all_properties_variance_test_results.csv")
        
        # Summary by property
        print(f"\n{'='*80}")
        print("SUMMARY BY PROPERTY")
        print(f"{'='*80}")
        
        for prop in property_names:
            prop_df = results_df[results_df['property'] == prop]
            if len(prop_df) == 0:
                continue
            
            print(f"\n{prop}:")
            print(f"  Tests: {len(prop_df)}")
            print(f"  Pearson correlation:")
            print(f"    Mean r: {prop_df['pearson_r'].mean():.4f}")
            print(f"    Significant: {sum(prop_df['pearson_p'] < 0.05)}/{len(prop_df)}")
            print(f"  Spearman correlation:")
            print(f"    Mean ρ: {prop_df['spearman_rho'].mean():.4f}")
            print(f"    Significant: {sum(prop_df['spearman_p'] < 0.05)}/{len(prop_df)}")
            print(f"  Linear regression:")
            print(f"    Mean R²: {prop_df['r_squared'].mean():.4f}")
            print(f"    Significant: {sum(prop_df['f_p'] < 0.05)}/{len(prop_df)}")
        
        # Summary by protein
        print(f"\n{'='*80}")
        print("SUMMARY BY PROTEIN")
        print(f"{'='*80}")
        
        proteins = results_df['protein'].unique()
        for protein in proteins:
            prot_df = results_df[results_df['protein'] == protein]
            sig_count = sum(prot_df['pearson_p'] < 0.05)
            print(f"\n{protein}:")
            print(f"  Properties tested: {len(prot_df)}")
            print(f"  Significant correlations: {sig_count}/{len(prot_df)}")
            if sig_count > 0:
                sig_props = prot_df[prot_df['pearson_p'] < 0.05]
                print(f"  Significant properties:")
                for _, row in sig_props.iterrows():
                    direction = "positive" if row['pearson_r'] > 0 else "negative"
                    print(f"    - {row['property']}: r={row['pearson_r']:.3f} ({direction}), p={row['pearson_p']:.4f}")
        
        # Overall summary table
        print(f"\n{'='*80}")
        print("SIGNIFICANT CORRELATIONS (p < 0.05)")
        print(f"{'='*80}")
        sig_df = results_df[results_df['pearson_p'] < 0.05].sort_values('pearson_p')
        
        if len(sig_df) > 0:
            print(f"\n{'Protein':<10} {'Property':<25} {'Pearson r':<12} {'p-value':<12} {'R²':<10}")
            print("-" * 80)
            for _, row in sig_df.iterrows():
                print(f"{row['protein']:<10} {row['property']:<25} {row['pearson_r']:<12.4f} {row['pearson_p']:<12.4e} {row['r_squared']:<10.4f}")
        else:
            print("\nNo significant correlations found (p < 0.05)")
        
        # Property ranking by average correlation strength
        print(f"\n{'='*80}")
        print("PROPERTY RANKING BY AVERAGE CORRELATION STRENGTH")
        print(f"{'='*80}")
        
        prop_summary = []
        for prop in property_names:
            prop_df = results_df[results_df['property'] == prop]
            if len(prop_df) > 0:
                prop_summary.append({
                    'Property': prop,
                    'Mean_Pearson_r': prop_df['pearson_r'].mean(),
                    'Mean_Abs_r': abs(prop_df['pearson_r']).mean(),
                    'Significant_Count': sum(prop_df['pearson_p'] < 0.05),
                    'Total_Count': len(prop_df),
                    'Mean_R_squared': prop_df['r_squared'].mean()
                })
        
        prop_summary_df = pd.DataFrame(prop_summary).sort_values('Mean_Abs_r', ascending=False)
        print(f"\n{'Property':<30} {'Mean |r|':<12} {'Mean r':<12} {'Sig/Total':<12} {'Mean R²':<10}")
        print("-" * 90)
        for _, row in prop_summary_df.iterrows():
            print(f"{row['Property']:<30} {row['Mean_Abs_r']:<12.4f} {row['Mean_Pearson_r']:<12.4f} "
                  f"{row['Significant_Count']}/{row['Total_Count']:<11} {row['Mean_R_squared']:<10.4f}")
        
        # Save property summary
        prop_summary_df.to_csv('property_summary_ranking.csv', index=False)
        print(f"\nProperty summary saved to: property_summary_ranking.csv")

if __name__ == '__main__':
    main()

