#!/usr/bin/env python3
"""
Statistical test to determine if amino acid properties (from suffix.csv) correlate with 
various statistical measures of interface bonds loss.

Available measures:
1. Mean - Average interface bonds loss
2. Variance - Spread of values
3. Standard Deviation - Square root of variance
4. Range - Max - Min
5. Median - Middle value (robust to outliers)
6. Interquartile Range (IQR) - Q3 - Q1
7. Coefficient of Variation - Std/Mean (normalized spread)
8. Maximum - Peak interface bonds loss
9. Minimum - Lowest interface bonds loss
10. Skewness - Asymmetry measure
11. Kurtosis - Tail heaviness measure
12. 75th Percentile - Upper quartile
13. 90th Percentile - High values threshold
14. 95th Percentile - Very high values threshold
15. Outlier Percentage - Percentage of values that are outliers (IQR method)

Usage:
    python test_all_statistics.py [--measures MEASURE1,MEASURE2,...]
    
Example:
    python test_all_statistics.py --measures mean,median,std,max
    python test_all_statistics.py  # Tests all measures
"""

import glob
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
from sklearn.linear_model import LinearRegression
import os
import argparse

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='Precision loss occurred')
warnings.filterwarnings('ignore', message='An input array is constant')

def load_suffix_properties(suffix_file='suffix.csv'):
    """Load all property data from suffix.csv."""
    df = pd.read_csv(suffix_file)
    property_cols = [col for col in df.columns if col != 'Amino Acid']
    properties_dict = {}
    for col in property_cols:
        properties_dict[col] = dict(zip(df['Amino Acid'], df[col]))
    return properties_dict

def calculate_statistics_per_aa(interface_bonds_file, measure='variance'):
    """
    Calculate specified statistical measure of interface bonds loss for each amino acid.
    
    Args:
        interface_bonds_file: Path to CSV file
        measure: One of: mean, variance, std, range, median, iqr, cv, max, min, 
                 skewness, kurtosis, p75, p90, p95, outlier_pct
    
    Returns:
        Dictionary mapping amino acid code to statistic value
    """
    df = pd.read_csv(interface_bonds_file)
    
    if 'class' not in df.columns:
        return None
    
    numeric_cols = [col for col in df.columns if col != 'class']
    stats_dict = {}
    
    for _, row in df.iterrows():
        aa_code = row['class']
        values = []
        for col in numeric_cols:
            val = row[col]
            if pd.notna(val) and val != '':
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    continue
        
        if len(values) == 0:
            stats_dict[aa_code] = np.nan
            continue
        
        values = np.array(values)
        
        # Calculate the requested measure
        if measure == 'mean':
            stat_value = np.mean(values)
        elif measure == 'variance':
            stat_value = np.var(values) if len(values) > 1 else 0.0
        elif measure == 'std':
            stat_value = np.std(values) if len(values) > 1 else 0.0
        elif measure == 'range':
            stat_value = np.max(values) - np.min(values)
        elif measure == 'median':
            stat_value = np.median(values)
        elif measure == 'iqr':
            q75, q25 = np.percentile(values, [75, 25])
            stat_value = q75 - q25
        elif measure == 'cv':
            mean_val = np.mean(values)
            std_val = np.std(values) if len(values) > 1 else 0.0
            stat_value = std_val / mean_val if mean_val != 0 else np.nan
        elif measure == 'max':
            stat_value = np.max(values)
        elif measure == 'min':
            stat_value = np.min(values)
        elif measure == 'skewness':
            stat_value = stats.skew(values) if len(values) > 2 else np.nan
        elif measure == 'kurtosis':
            stat_value = stats.kurtosis(values) if len(values) > 3 else np.nan
        elif measure == 'p75':
            stat_value = np.percentile(values, 75)
        elif measure == 'p90':
            stat_value = np.percentile(values, 90)
        elif measure == 'p95':
            stat_value = np.percentile(values, 95)
        elif measure == 'outlier_pct':
            # Calculate percentage of outliers using IQR method
            if len(values) < 4:  # Need at least 4 values for IQR
                stat_value = np.nan
            else:
                q25, q75 = np.percentile(values, [25, 75])
                iqr = q75 - q25
                if iqr == 0:
                    # If IQR is 0, use alternative method (values beyond 3 std dev)
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    if std_val == 0:
                        stat_value = 0.0  # All values are the same
                    else:
                        outliers = np.sum(np.abs(values - mean_val) > 3 * std_val)
                        stat_value = (outliers / len(values)) * 100
                else:
                    # Standard IQR method: outliers beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outliers = np.sum((values < lower_bound) | (values > upper_bound))
                    stat_value = (outliers / len(values)) * 100
        else:
            raise ValueError(f"Unknown measure: {measure}")
        
        stats_dict[aa_code] = stat_value
    
    return stats_dict

def perform_statistical_tests_for_property(property_data, property_name, stats_data, protein_name, measure_name):
    """Perform statistical tests on the relationship between a property and a statistic."""
    combined_data = []
    for aa in property_data.keys():
        if aa in stats_data and not np.isnan(stats_data[aa]):
            prop_val = property_data[aa]
            if pd.notna(prop_val):
                combined_data.append({
                    'Amino_Acid': aa,
                    'Property': prop_val,
                    'Statistic': stats_data[aa]
                })
    
    if len(combined_data) < 3:
        return None
    
    df = pd.DataFrame(combined_data)
    df = df.sort_values('Property')
    
    # Pearson correlation (handle constant arrays)
    try:
        if df['Statistic'].nunique() > 1 and df['Property'].nunique() > 1:
            corr, p_value = pearsonr(df['Property'], df['Statistic'])
        else:
            corr, p_value = np.nan, np.nan
    except:
        corr, p_value = np.nan, np.nan
    
    # Spearman rank correlation (handle constant arrays)
    try:
        if df['Statistic'].nunique() > 1 and df['Property'].nunique() > 1:
            spearman_corr, spearman_p = spearmanr(df['Property'], df['Statistic'])
        else:
            spearman_corr, spearman_p = np.nan, np.nan
    except:
        spearman_corr, spearman_p = np.nan, np.nan
    
    # High vs Low groups
    median_prop = df['Property'].median()
    high_group = df[df['Property'] >= median_prop]['Statistic']
    low_group = df[df['Property'] < median_prop]['Statistic']
    
    t_p = np.nan
    u_p = np.nan
    try:
        # Only perform t-test if groups have sufficient variance
        if len(high_group) > 1 and len(low_group) > 1:
            if high_group.nunique() > 1 or low_group.nunique() > 1:
                _, t_p = ttest_ind(high_group, low_group)
    except:
        pass
    
    try:
        # Only perform Mann-Whitney U if groups have sufficient variance
        if len(high_group) > 1 and len(low_group) > 1:
            if high_group.nunique() > 1 or low_group.nunique() > 1:
                _, u_p = mannwhitneyu(high_group, low_group, alternative='two-sided')
    except:
        pass
    
    # Linear regression
    X = df[['Property']].values
    y = df['Statistic'].values
    
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
        'measure': measure_name,
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
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test correlation between amino acid properties and various statistics of interface bonds loss"
    )
    parser.add_argument(
        '--measures', '-m',
        type=str,
        default='mean,variance,std,range,median,iqr,cv,max,min,skewness,kurtosis,p75,p90,p95,outlier_pct',
        help='Comma-separated list of measures to test (default: all)'
    )
    args = parser.parse_args()
    
    # Parse measures
    available_measures = ['mean', 'variance', 'std', 'range', 'median', 'iqr', 'cv', 
                          'max', 'min', 'skewness', 'kurtosis', 'p75', 'p90', 'p95', 'outlier_pct']
    requested_measures = [m.strip() for m in args.measures.split(',')]
    
    # Validate measures
    invalid = [m for m in requested_measures if m not in available_measures]
    if invalid:
        print(f"Error: Invalid measures: {invalid}")
        print(f"Available measures: {', '.join(available_measures)}")
        return
    
    print(f"Testing measures: {', '.join(requested_measures)}")
    
    # Load properties
    print("\nLoading property data from suffix.csv...")
    properties_dict = load_suffix_properties('suffix.csv')
    property_names = list(properties_dict.keys())
    print(f"Loaded {len(property_names)} properties")
    
    # Find files
    pattern = '*_interface_bonds_loss.csv'
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"\nFound {len(files)} interface bonds loss CSV files")
    
    # Process each file, measure, and property combination
    all_results = []
    
    for filename in files:
        protein_name = os.path.basename(filename).replace('_interface_bonds_loss.csv', '')
        print(f"\nProcessing {protein_name}...")
        
        for measure in requested_measures:
            print(f"  Calculating {measure}...")
            stats_data = calculate_statistics_per_aa(filename, measure)
            
            if stats_data is None:
                continue
            
            for property_name in property_names:
                property_data = properties_dict[property_name]
                result = perform_statistical_tests_for_property(
                    property_data, property_name, stats_data, protein_name, measure
                )
                if result:
                    all_results.append(result)
    
    # Summary
    if all_results:
        print(f"\n\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('all_statistics_test_results.csv', index=False)
        print(f"\nDetailed results saved to: all_statistics_test_results.csv")
        
        # Summary by measure
        print(f"\n{'='*80}")
        print("SUMMARY BY MEASURE")
        print(f"{'='*80}")
        
        for measure in requested_measures:
            measure_df = results_df[results_df['measure'] == measure]
            if len(measure_df) == 0:
                continue
            
            print(f"\n{measure.upper()}:")
            print(f"  Total tests: {len(measure_df)}")
            print(f"  Significant correlations (p < 0.05): {sum(measure_df['pearson_p'] < 0.05)}/{len(measure_df)}")
            print(f"  Mean |Pearson r|: {abs(measure_df['pearson_r']).mean():.4f}")
            print(f"  Mean R²: {measure_df['r_squared'].mean():.4f}")
            
            # Top properties for this measure
            if len(measure_df) > 0:
                prop_summary = measure_df.groupby('property').agg({
                    'pearson_r': 'mean',
                    'pearson_p': lambda x: sum(x < 0.05)
                }).sort_values('pearson_r', key=abs, ascending=False)
                
                print(f"  Top 3 properties by |correlation|:")
                for prop, row in prop_summary.head(3).iterrows():
                    print(f"    - {prop}: r={row['pearson_r']:.3f}, sig={int(row['pearson_p'])}/4")
        
        # Significant results
        print(f"\n{'='*85}")
        print("SIGNIFICANT CORRELATIONS (p < 0.05)")
        print(f"{'='*85}")
        sig_df = results_df[results_df['pearson_p'] < 0.05].sort_values('pearson_p')
        
        if len(sig_df) > 0:
            print(f"\n{'Measure':<15} {'Protein':<10} {'Property':<25} {'Pearson r':<12} {'p-value':<12} {'R²':<10}")
            print("-" * 85)
            for _, row in sig_df.iterrows():
                print(f"{row['measure']:<15} {row['protein']:<10} {row['property']:<25} "
                      f"{row['pearson_r']:<12.4f} {row['pearson_p']:<12.6f} {row['r_squared']:<10.4f}")
        else:
            print("\nNo significant correlations found (p < 0.05)")
        
        # Measure ranking
        print(f"\n{'='*85}")
        print("MEASURE RANKING BY SIGNIFICANT CORRELATIONS")
        print(f"{'='*85}")
        
        measure_summary = results_df.groupby('measure').agg({
            'pearson_r': lambda x: abs(x).mean(),
            'pearson_p': lambda x: sum(x < 0.05),
            'r_squared': 'mean'
        }).sort_values('pearson_p', ascending=False)
        
        print(f"\n{'Measure':<15} {'Mean |r|':<12} {'Sig/Total':<12} {'Mean R²':<10}")
        print("-" * 85)
        for measure, row in measure_summary.iterrows():
            total = len(results_df[results_df['measure'] == measure])
            print(f"{measure:<15} {row['pearson_r']:<12.4f} {int(row['pearson_p'])}/{total:<11} {row['r_squared']:<10.4f}")
        
        # Save measure summary
        measure_summary.to_csv('measure_summary_ranking.csv')
        print(f"\nMeasure summary saved to: measure_summary_ranking.csv")

if __name__ == '__main__':
    main()

