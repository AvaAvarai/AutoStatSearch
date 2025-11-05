#!/usr/bin/env python3
"""
Script to calculate Pearson, Spearman, and Kendall correlation coefficients between each suffix column
and all other columns (excluding other suffix columns) in a CSV file.
"""

import csv
import sys
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. FDR correction will use manual implementation.")

def get_suffix_columns(suffix_file='suffix.csv'):
    """Get the list of suffix column names from suffix.csv."""
    if not os.path.exists(suffix_file):
        print(f"Warning: {suffix_file} not found. Using default suffix columns.")
        return []
    
    with open(suffix_file, 'r') as f:
        reader = csv.DictReader(f)
        suffix_fieldnames = reader.fieldnames
        
        if not suffix_fieldnames:
            return []
        
        # All columns except "Amino Acid" are suffix columns
        suffix_cols = [col for col in suffix_fieldnames if col != 'Amino Acid']
        return suffix_cols

def is_numeric(value):
    """Check if a value can be converted to float."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def convert_to_numeric(series):
    """Convert a series to numeric, replacing non-numeric values with NaN."""
    numeric_series = []
    for val in series:
        if is_numeric(val) and val != '':
            numeric_series.append(float(val))
        else:
            numeric_series.append(np.nan)
    return np.array(numeric_series)

def is_constant_array(arr, tolerance=1e-10):
    """Check if an array is constant (all values are the same within tolerance)."""
    if len(arr) == 0:
        return True
    valid_arr = arr[~np.isnan(arr)]
    if len(valid_arr) == 0:
        return True
    if len(valid_arr) == 1:
        return True
    return np.std(valid_arr) < tolerance

def apply_fdr_correction(p_values, alpha=0.05, method='fdr_bh'):
    """
    Apply False Discovery Rate (FDR) correction using Benjamini-Hochberg method.
    
    Parameters:
    - p_values: array of p-values
    - alpha: significance level
    - method: 'fdr_bh' for Benjamini-Hochberg
    
    Returns:
    - corrected_p_values: array of FDR-corrected p-values
    - is_significant: boolean array indicating significance
    """
    # Filter out NaN values for correction
    valid_mask = ~np.isnan(p_values)
    if not np.any(valid_mask):
        return p_values.copy(), np.zeros_like(p_values, dtype=bool)
    
    valid_p = p_values[valid_mask]
    
    if HAS_STATSMODELS:
        # Use statsmodels if available
        corrected_p, _, _, _ = multipletests(valid_p, alpha=alpha, method=method)
    else:
        # Manual Benjamini-Hochberg implementation
        n = len(valid_p)
        # Sort p-values and get indices
        sorted_indices = np.argsort(valid_p)
        sorted_p = valid_p[sorted_indices]
        
        # Calculate adjusted p-values
        corrected_p = np.zeros_like(sorted_p)
        corrected_p[n-1] = sorted_p[n-1]  # Last one
        
        for i in range(n-2, -1, -1):
            corrected_p[i] = min(sorted_p[i] * n / (i+1), corrected_p[i+1])
        
        # Restore original order
        restored_indices = np.argsort(sorted_indices)
        corrected_p = corrected_p[restored_indices]
    
    # Create full array with corrected values
    corrected_full = p_values.copy()
    corrected_full[valid_mask] = corrected_p
    
    # Determine significance
    is_significant = np.zeros_like(p_values, dtype=bool)
    is_significant[valid_mask] = corrected_p < alpha
    
    return corrected_full, is_significant

def apply_bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Bonferroni correction for multiple testing.
    
    Parameters:
    - p_values: array of p-values
    - alpha: significance level
    
    Returns:
    - corrected_p_values: array of Bonferroni-corrected p-values
    - is_significant: boolean array indicating significance
    """
    # Filter out NaN values for correction
    valid_mask = ~np.isnan(p_values)
    if not np.any(valid_mask):
        return p_values.copy(), np.zeros_like(p_values, dtype=bool)
    
    valid_p = p_values[valid_mask]
    n_tests = len(valid_p)
    
    # Apply Bonferroni correction
    corrected_p = valid_p * n_tests
    corrected_p = np.clip(corrected_p, 0, 1)  # Ensure p-values are in [0, 1]
    
    # Create full array with corrected values
    corrected_full = p_values.copy()
    corrected_full[valid_mask] = corrected_p
    
    # Determine significance
    is_significant = np.zeros_like(p_values, dtype=bool)
    is_significant[valid_mask] = corrected_p < alpha
    
    return corrected_full, is_significant

def calculate_correlations(target_file, suffix_file='suffix.csv', output_file=None):
    """Calculate Pearson, Spearman, and Kendall correlations between suffix columns and all other columns."""
    
    # Get suffix columns
    suffix_columns = get_suffix_columns(suffix_file)
    print(f"Identified {len(suffix_columns)} suffix columns:")
    for col in suffix_columns:
        print(f"  - {col}")
    
    if not os.path.exists(target_file):
        print(f"Error: {target_file} not found!")
        sys.exit(1)
    
    # Read the target CSV
    print(f"\nReading {target_file}...")
    df = pd.read_csv(target_file)
    
    print(f"Total columns: {len(df.columns)}")
    print(f"Total rows: {len(df)}")
    
    # Identify suffix and non-suffix columns
    all_columns = list(df.columns)
    suffix_cols_in_file = [col for col in all_columns if col in suffix_columns]
    non_suffix_cols = [col for col in all_columns if col not in suffix_columns]
    
    print(f"\nSuffix columns found in file: {len(suffix_cols_in_file)}")
    print(f"Non-suffix columns: {len(non_suffix_cols)}")
    
    if not suffix_cols_in_file:
        print("Error: No suffix columns found in the target file!")
        sys.exit(1)
    
    # Prepare results storage
    results = []
    
    print("\nCalculating correlations...")
    print("=" * 80)
    
    # For each suffix column
    for suffix_col in suffix_cols_in_file:
        print(f"\nProcessing suffix column: {suffix_col}")
        
        # Get suffix column data
        suffix_data = convert_to_numeric(df[suffix_col].values)
        suffix_data_valid = ~np.isnan(suffix_data)
        
        if not np.any(suffix_data_valid):
            print(f"  Warning: {suffix_col} has no valid numeric data")
            continue
        
        # Calculate correlation with each non-suffix column
        correlations = []
        for non_suffix_col in non_suffix_cols:
            # Get non-suffix column data
            other_data = convert_to_numeric(df[non_suffix_col].values)
            other_data_valid = ~np.isnan(other_data)
            
            # Find rows where both columns have valid data
            both_valid = suffix_data_valid & other_data_valid
            
            if np.sum(both_valid) < 2:
                # Need at least 2 data points for correlation
                correlations.append({
                    'suffix_column': suffix_col,
                    'other_column': non_suffix_col,
                    'pearson_correlation': np.nan,
                    'pearson_p_value': np.nan,
                    'spearman_correlation': np.nan,
                    'spearman_p_value': np.nan,
                    'kendall_correlation': np.nan,
                    'kendall_p_value': np.nan,
                    'n_valid': np.sum(both_valid)
                })
                continue
            
            # Get valid data for correlation
            suffix_valid_data = suffix_data[both_valid]
            other_valid_data = other_data[both_valid]
            
            # Check for constant arrays (correlation undefined when one variable is constant)
            suffix_is_constant = is_constant_array(suffix_valid_data)
            other_is_constant = is_constant_array(other_valid_data)
            
            # Calculate Pearson and Spearman correlations
            result_dict = {
                'suffix_column': suffix_col,
                'other_column': non_suffix_col,
                'n_valid': np.sum(both_valid)
            }
            
            # Pearson correlation
            if suffix_is_constant or other_is_constant:
                result_dict['pearson_correlation'] = np.nan
                result_dict['pearson_p_value'] = np.nan
            else:
                try:
                    pearson_corr, pearson_p = pearsonr(suffix_valid_data, other_valid_data)
                    result_dict['pearson_correlation'] = pearson_corr
                    result_dict['pearson_p_value'] = pearson_p
                except Exception as e:
                    result_dict['pearson_correlation'] = np.nan
                    result_dict['pearson_p_value'] = np.nan
                    result_dict['pearson_error'] = str(e)
            
            # Spearman correlation
            if suffix_is_constant or other_is_constant:
                result_dict['spearman_correlation'] = np.nan
                result_dict['spearman_p_value'] = np.nan
            else:
                try:
                    spearman_corr, spearman_p = spearmanr(suffix_valid_data, other_valid_data)
                    result_dict['spearman_correlation'] = spearman_corr
                    result_dict['spearman_p_value'] = spearman_p
                except Exception as e:
                    result_dict['spearman_correlation'] = np.nan
                    result_dict['spearman_p_value'] = np.nan
                    result_dict['spearman_error'] = str(e)
            
            # Kendall correlation
            if suffix_is_constant or other_is_constant:
                result_dict['kendall_correlation'] = np.nan
                result_dict['kendall_p_value'] = np.nan
            else:
                try:
                    kendall_corr, kendall_p = kendalltau(suffix_valid_data, other_valid_data)
                    result_dict['kendall_correlation'] = kendall_corr
                    result_dict['kendall_p_value'] = kendall_p
                except Exception as e:
                    result_dict['kendall_correlation'] = np.nan
                    result_dict['kendall_p_value'] = np.nan
                    result_dict['kendall_error'] = str(e)
            
            correlations.append(result_dict)
        
        # Sort by absolute Pearson correlation value
        valid_pearson = [c for c in correlations if not np.isnan(c.get('pearson_correlation', np.nan))]
        if valid_pearson:
            valid_pearson.sort(key=lambda x: abs(x['pearson_correlation']), reverse=True)
            print(f"  Top 5 strongest Pearson correlations:")
            for i, corr_info in enumerate(valid_pearson[:5]):
                print(f"    {i+1}. {corr_info['other_column']}: r={corr_info['pearson_correlation']:.4f}, p={corr_info['pearson_p_value']:.4e}, n={corr_info['n_valid']}")
        
        # Sort by absolute Spearman correlation value
        valid_spearman = [c for c in correlations if not np.isnan(c.get('spearman_correlation', np.nan))]
        if valid_spearman:
            valid_spearman.sort(key=lambda x: abs(x['spearman_correlation']), reverse=True)
            print(f"  Top 5 strongest Spearman correlations:")
            for i, corr_info in enumerate(valid_spearman[:5]):
                print(f"    {i+1}. {corr_info['other_column']}: rho={corr_info['spearman_correlation']:.4f}, p={corr_info['spearman_p_value']:.4e}, n={corr_info['n_valid']}")
        
        # Sort by absolute Kendall correlation value
        valid_kendall = [c for c in correlations if not np.isnan(c.get('kendall_correlation', np.nan))]
        if valid_kendall:
            valid_kendall.sort(key=lambda x: abs(x['kendall_correlation']), reverse=True)
            print(f"  Top 5 strongest Kendall correlations:")
            for i, corr_info in enumerate(valid_kendall[:5]):
                print(f"    {i+1}. {corr_info['other_column']}: tau={corr_info['kendall_correlation']:.4f}, p={corr_info['kendall_p_value']:.4e}, n={corr_info['n_valid']}")
        
        results.extend(correlations)
    
    # Create output DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply multiple testing corrections for both Pearson and Spearman
    print(f"\n{'=' * 80}")
    print("Applying multiple testing corrections...")
    
    # Pearson corrections
    valid_pearson_p_mask = ~results_df['pearson_p_value'].isna()
    valid_pearson_p_values = results_df.loc[valid_pearson_p_mask, 'pearson_p_value'].values
    
    if len(valid_pearson_p_values) > 0:
        # Apply FDR correction (Benjamini-Hochberg) for Pearson
        fdr_corrected, fdr_significant = apply_fdr_correction(valid_pearson_p_values, alpha=0.05)
        results_df['pearson_p_value_fdr'] = np.nan
        results_df.loc[valid_pearson_p_mask, 'pearson_p_value_fdr'] = fdr_corrected
        results_df['pearson_significant_fdr'] = False
        results_df.loc[valid_pearson_p_mask, 'pearson_significant_fdr'] = fdr_significant
        
        # Apply Bonferroni correction for Pearson
        bonf_corrected, bonf_significant = apply_bonferroni_correction(valid_pearson_p_values, alpha=0.05)
        results_df['pearson_p_value_bonferroni'] = np.nan
        results_df.loc[valid_pearson_p_mask, 'pearson_p_value_bonferroni'] = bonf_corrected
        results_df['pearson_significant_bonferroni'] = False
        results_df.loc[valid_pearson_p_mask, 'pearson_significant_bonferroni'] = bonf_significant
        
        print(f"  Pearson correlations - Number of tests: {len(valid_pearson_p_values)}")
        print(f"    Significant after FDR correction (p_fdr < 0.05): {np.sum(fdr_significant)}")
        print(f"    Significant after Bonferroni correction (p_bonf < 0.05): {np.sum(bonf_significant)}")
    else:
        print("  Warning: No valid Pearson p-values for correction")
        results_df['pearson_p_value_fdr'] = np.nan
        results_df['pearson_significant_fdr'] = False
        results_df['pearson_p_value_bonferroni'] = np.nan
        results_df['pearson_significant_bonferroni'] = False
    
    # Spearman corrections
    valid_spearman_p_mask = ~results_df['spearman_p_value'].isna()
    valid_spearman_p_values = results_df.loc[valid_spearman_p_mask, 'spearman_p_value'].values
    
    if len(valid_spearman_p_values) > 0:
        # Apply FDR correction (Benjamini-Hochberg) for Spearman
        fdr_corrected, fdr_significant = apply_fdr_correction(valid_spearman_p_values, alpha=0.05)
        results_df['spearman_p_value_fdr'] = np.nan
        results_df.loc[valid_spearman_p_mask, 'spearman_p_value_fdr'] = fdr_corrected
        results_df['spearman_significant_fdr'] = False
        results_df.loc[valid_spearman_p_mask, 'spearman_significant_fdr'] = fdr_significant
        
        # Apply Bonferroni correction for Spearman
        bonf_corrected, bonf_significant = apply_bonferroni_correction(valid_spearman_p_values, alpha=0.05)
        results_df['spearman_p_value_bonferroni'] = np.nan
        results_df.loc[valid_spearman_p_mask, 'spearman_p_value_bonferroni'] = bonf_corrected
        results_df['spearman_significant_bonferroni'] = False
        results_df.loc[valid_spearman_p_mask, 'spearman_significant_bonferroni'] = bonf_significant
        
        print(f"  Spearman correlations - Number of tests: {len(valid_spearman_p_values)}")
        print(f"    Significant after FDR correction (p_fdr < 0.05): {np.sum(fdr_significant)}")
        print(f"    Significant after Bonferroni correction (p_bonf < 0.05): {np.sum(bonf_significant)}")
    else:
        print("  Warning: No valid Spearman p-values for correction")
        results_df['spearman_p_value_fdr'] = np.nan
        results_df['spearman_significant_fdr'] = False
        results_df['spearman_p_value_bonferroni'] = np.nan
        results_df['spearman_significant_bonferroni'] = False
    
    # Kendall corrections
    valid_kendall_p_mask = ~results_df['kendall_p_value'].isna()
    valid_kendall_p_values = results_df.loc[valid_kendall_p_mask, 'kendall_p_value'].values
    
    if len(valid_kendall_p_values) > 0:
        # Apply FDR correction (Benjamini-Hochberg) for Kendall
        fdr_corrected, fdr_significant = apply_fdr_correction(valid_kendall_p_values, alpha=0.05)
        results_df['kendall_p_value_fdr'] = np.nan
        results_df.loc[valid_kendall_p_mask, 'kendall_p_value_fdr'] = fdr_corrected
        results_df['kendall_significant_fdr'] = False
        results_df.loc[valid_kendall_p_mask, 'kendall_significant_fdr'] = fdr_significant
        
        # Apply Bonferroni correction for Kendall
        bonf_corrected, bonf_significant = apply_bonferroni_correction(valid_kendall_p_values, alpha=0.05)
        results_df['kendall_p_value_bonferroni'] = np.nan
        results_df.loc[valid_kendall_p_mask, 'kendall_p_value_bonferroni'] = bonf_corrected
        results_df['kendall_significant_bonferroni'] = False
        results_df.loc[valid_kendall_p_mask, 'kendall_significant_bonferroni'] = bonf_significant
        
        print(f"  Kendall correlations - Number of tests: {len(valid_kendall_p_values)}")
        print(f"    Significant after FDR correction (p_fdr < 0.05): {np.sum(fdr_significant)}")
        print(f"    Significant after Bonferroni correction (p_bonf < 0.05): {np.sum(bonf_significant)}")
    else:
        print("  Warning: No valid Kendall p-values for correction")
        results_df['kendall_p_value_fdr'] = np.nan
        results_df['kendall_significant_fdr'] = False
        results_df['kendall_p_value_bonferroni'] = np.nan
        results_df['kendall_significant_bonferroni'] = False
    
    # Determine output filename
    if output_file is None:
        base_name = os.path.splitext(target_file)[0]
        output_file = f"{base_name}_correlations.csv"
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print(f"Total correlations calculated: {len(results)}")
    
    # Summary statistics
    print(f"\n{'=' * 80}")
    print("Summary Statistics:")
    
    # Pearson statistics
    valid_pearson = results_df[~results_df['pearson_correlation'].isna()]
    if len(valid_pearson) > 0:
        print(f"\n  Pearson Correlations:")
        print(f"    Valid correlations: {len(valid_pearson)}")
        print(f"    Mean |correlation|: {valid_pearson['pearson_correlation'].abs().mean():.4f}")
        print(f"    Max |correlation|: {valid_pearson['pearson_correlation'].abs().max():.4f}")
        print(f"    Min |correlation|: {valid_pearson['pearson_correlation'].abs().min():.4f}")
        
        print(f"\n    Significance (uncorrected, p < 0.05): {len(valid_pearson[valid_pearson['pearson_p_value'] < 0.05])}")
        if 'pearson_p_value_fdr' in valid_pearson.columns:
            print(f"    Significance (FDR corrected, p_fdr < 0.05): {len(valid_pearson[valid_pearson['pearson_significant_fdr']])}")
            print(f"    Significance (Bonferroni corrected, p_bonf < 0.05): {len(valid_pearson[valid_pearson['pearson_significant_bonferroni']])}")
        
        print(f"\n    Correlation strength categories:")
        print(f"      Strong (|r| > 0.7): {len(valid_pearson[valid_pearson['pearson_correlation'].abs() > 0.7])}")
        print(f"      Moderate (0.5 < |r| <= 0.7): {len(valid_pearson[(valid_pearson['pearson_correlation'].abs() > 0.5) & (valid_pearson['pearson_correlation'].abs() <= 0.7)])}")
        print(f"      Weak (0.3 < |r| <= 0.5): {len(valid_pearson[(valid_pearson['pearson_correlation'].abs() > 0.3) & (valid_pearson['pearson_correlation'].abs() <= 0.5)])}")
        print(f"      Very weak (|r| <= 0.3): {len(valid_pearson[valid_pearson['pearson_correlation'].abs() <= 0.3])}")
    
    # Spearman statistics
    valid_spearman = results_df[~results_df['spearman_correlation'].isna()]
    if len(valid_spearman) > 0:
        print(f"\n  Spearman Correlations:")
        print(f"    Valid correlations: {len(valid_spearman)}")
        print(f"    Mean |correlation|: {valid_spearman['spearman_correlation'].abs().mean():.4f}")
        print(f"    Max |correlation|: {valid_spearman['spearman_correlation'].abs().max():.4f}")
        print(f"    Min |correlation|: {valid_spearman['spearman_correlation'].abs().min():.4f}")
        
        print(f"\n    Significance (uncorrected, p < 0.05): {len(valid_spearman[valid_spearman['spearman_p_value'] < 0.05])}")
        if 'spearman_p_value_fdr' in valid_spearman.columns:
            print(f"    Significance (FDR corrected, p_fdr < 0.05): {len(valid_spearman[valid_spearman['spearman_significant_fdr']])}")
            print(f"    Significance (Bonferroni corrected, p_bonf < 0.05): {len(valid_spearman[valid_spearman['spearman_significant_bonferroni']])}")
        
        print(f"\n    Correlation strength categories:")
        print(f"      Strong (|rho| > 0.7): {len(valid_spearman[valid_spearman['spearman_correlation'].abs() > 0.7])}")
        print(f"      Moderate (0.5 < |rho| <= 0.7): {len(valid_spearman[(valid_spearman['spearman_correlation'].abs() > 0.5) & (valid_spearman['spearman_correlation'].abs() <= 0.7)])}")
        print(f"      Weak (0.3 < |rho| <= 0.5): {len(valid_spearman[(valid_spearman['spearman_correlation'].abs() > 0.3) & (valid_spearman['spearman_correlation'].abs() <= 0.5)])}")
        print(f"      Very weak (|rho| <= 0.3): {len(valid_spearman[valid_spearman['spearman_correlation'].abs() <= 0.3])}")
    
    # Kendall statistics
    valid_kendall = results_df[~results_df['kendall_correlation'].isna()]
    if len(valid_kendall) > 0:
        print(f"\n  Kendall Correlations:")
        print(f"    Valid correlations: {len(valid_kendall)}")
        print(f"    Mean |correlation|: {valid_kendall['kendall_correlation'].abs().mean():.4f}")
        print(f"    Max |correlation|: {valid_kendall['kendall_correlation'].abs().max():.4f}")
        print(f"    Min |correlation|: {valid_kendall['kendall_correlation'].abs().min():.4f}")
        
        print(f"\n    Significance (uncorrected, p < 0.05): {len(valid_kendall[valid_kendall['kendall_p_value'] < 0.05])}")
        if 'kendall_p_value_fdr' in valid_kendall.columns:
            print(f"    Significance (FDR corrected, p_fdr < 0.05): {len(valid_kendall[valid_kendall['kendall_significant_fdr']])}")
            print(f"    Significance (Bonferroni corrected, p_bonf < 0.05): {len(valid_kendall[valid_kendall['kendall_significant_bonferroni']])}")
        
        print(f"\n    Correlation strength categories:")
        print(f"      Strong (|tau| > 0.7): {len(valid_kendall[valid_kendall['kendall_correlation'].abs() > 0.7])}")
        print(f"      Moderate (0.5 < |tau| <= 0.7): {len(valid_kendall[(valid_kendall['kendall_correlation'].abs() > 0.5) & (valid_kendall['kendall_correlation'].abs() <= 0.7)])}")
        print(f"      Weak (0.3 < |tau| <= 0.5): {len(valid_kendall[(valid_kendall['kendall_correlation'].abs() > 0.3) & (valid_kendall['kendall_correlation'].abs() <= 0.5)])}")
        print(f"      Very weak (|tau| <= 0.3): {len(valid_kendall[valid_kendall['kendall_correlation'].abs() <= 0.3])}")
    
    return results_df

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python calculate_pearson_suffix_correlations.py <target_csv_file> [suffix_file] [output_file]")
        print("\nExample:")
        print("  python calculate_pearson_suffix_correlations.py 1brs_interface_bonds_loss_with_suffix.csv")
        print("  python calculate_pearson_suffix_correlations.py 1brs_interface_bonds_loss_with_suffix.csv suffix.csv")
        print("  python calculate_pearson_suffix_correlations.py 1brs_interface_bonds_loss_with_suffix.csv suffix.csv output.csv")
        sys.exit(1)
    
    target_file = sys.argv[1]
    suffix_file = sys.argv[2] if len(sys.argv) > 2 else 'suffix.csv'
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    calculate_correlations(target_file, suffix_file, output_file)

if __name__ == '__main__':
    main()
