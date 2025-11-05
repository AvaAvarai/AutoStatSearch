#!/usr/bin/env python3
"""
Analyze which specific amino acids are driving significant correlations.
For each significant correlation, identify:
1. Amino acids with high p90 values
2. Their property values
3. Which specific values within each amino acid's distribution are in the top 10% (p90+)
"""

import pandas as pd
import numpy as np
import glob
import os

def load_suffix_properties(suffix_file='suffix.csv'):
    """Load all property data from suffix.csv."""
    df = pd.read_csv(suffix_file)
    property_cols = [col for col in df.columns if col != 'Amino Acid']
    properties_dict = {}
    for col in property_cols:
        properties_dict[col] = dict(zip(df['Amino Acid'], df[col]))
    return properties_dict

def calculate_p90_per_aa(interface_bonds_file):
    """
    Calculate p90 (90th percentile) of interface bonds loss for each amino acid.
    Also return the actual values that exceed p90 for each amino acid.
    """
    df = pd.read_csv(interface_bonds_file)
    
    if 'class' not in df.columns:
        return None, None
    
    numeric_cols = [col for col in df.columns if col != 'class']
    p90_dict = {}
    high_values_dict = {}  # Values that are >= p90 for each amino acid
    
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
            p90_dict[aa_code] = np.nan
            high_values_dict[aa_code] = []
            continue
        
        values = np.array(values)
        p90_value = np.percentile(values, 90)
        p90_dict[aa_code] = p90_value
        
        # Find values that are >= p90 (in the top 10%)
        high_values = values[values >= p90_value].tolist()
        high_values_dict[aa_code] = sorted(high_values, reverse=True)
    
    return p90_dict, high_values_dict

def analyze_significant_correlations():
    """Analyze amino acid drivers for significant correlations."""
    
    # Define significant correlations (from our analysis)
    significant_correlations = [
        {'protein': '1qa9', 'property': 'Volume of Side Chains', 'r': 0.511, 'p': 0.021},
        {'protein': '1qa9', 'property': 'Molecular Mass', 'r': 0.494, 'p': 0.027},
        {'protein': '1qa9', 'property': 'Residue Weight', 'r': 0.494, 'p': 0.027},
        {'protein': '1qa9', 'property': 'SASA', 'r': 0.469, 'p': 0.037},
        {'protein': '1qa9', 'property': 'MaxASA', 'r': 0.517, 'p': 0.020},
        {'protein': '1sq0', 'property': 'Abundance', 'r': -0.550, 'p': 0.012},
    ]
    
    # Load properties
    properties_dict = load_suffix_properties('suffix.csv')
    
    results = []
    
    for corr in significant_correlations:
        protein = corr['protein']
        property_name = corr['property']
        
        # Load interface bonds file
        filename = f"{protein}_interface_bonds_loss.csv"
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found")
            continue
        
        # Calculate p90 per amino acid
        p90_dict, high_values_dict = calculate_p90_per_aa(filename)
        
        if p90_dict is None:
            continue
        
        # Get property values
        property_data = properties_dict[property_name]
        
        # Combine data
        combined_data = []
        for aa in p90_dict.keys():
            if aa in property_data and not np.isnan(p90_dict[aa]):
                prop_val = property_data[aa]
                if pd.notna(prop_val):
                    combined_data.append({
                        'Amino_Acid': aa,
                        'Property': prop_val,
                        'p90': p90_dict[aa],
                        'High_Values': high_values_dict[aa],
                        'Num_High_Values': len(high_values_dict[aa])
                    })
        
        if len(combined_data) == 0:
            continue
        
        df = pd.DataFrame(combined_data)
        
        # Calculate percentage of values in top 10% for each amino acid
        for idx, row in df.iterrows():
            total_values = len(high_values_dict[row['Amino_Acid']]) + len([v for v in high_values_dict[row['Amino_Acid']] if v < row['p90']])
            # We need to get total count from the file
            interface_df = pd.read_csv(filename)
            numeric_cols = [col for col in interface_df.columns if col != 'class']
            aa_row = interface_df[interface_df['class'] == row['Amino_Acid']]
            if len(aa_row) > 0:
                total_vals = sum(1 for col in numeric_cols if pd.notna(aa_row.iloc[0][col]) and aa_row.iloc[0][col] != '')
                if total_vals > 0:
                    df.at[idx, 'Percent_In_Top10'] = (row['Num_High_Values'] / total_vals) * 100
                else:
                    df.at[idx, 'Percent_In_Top10'] = 0
            else:
                df.at[idx, 'Percent_In_Top10'] = 0
        
        # Sort by p90 (descending) to see which amino acids have highest p90
        df_sorted = df.sort_values('p90', ascending=False)
        
        # Determine correlation direction
        corr_direction = "positive" if corr['r'] > 0 else "negative"
        
        # For positive correlations: high p90 should have high property values
        # For negative correlations: high p90 should have low property values
        if corr_direction == "positive":
            # Amino acids driving the correlation: high p90 AND high property
            df_sorted['Drives_Correlation'] = (
                (df_sorted['p90'] >= df_sorted['p90'].quantile(0.75)) & 
                (df_sorted['Property'] >= df_sorted['Property'].quantile(0.75))
            )
        else:
            # Amino acids driving the correlation: high p90 AND low property
            df_sorted['Drives_Correlation'] = (
                (df_sorted['p90'] >= df_sorted['p90'].quantile(0.75)) & 
                (df_sorted['Property'] <= df_sorted['Property'].quantile(0.25))
            )
        
        results.append({
            'correlation': corr,
            'data': df_sorted,
            'correlation_direction': corr_direction
        })
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"PROTEIN: {protein} | PROPERTY: {property_name}")
        print(f"Correlation: r={corr['r']:.3f}, p={corr['p']:.4f} ({corr_direction})")
        print(f"{'='*80}")
        
        # Add interpretation
        if corr_direction == "positive":
            print(f"\nINTERPRETATION: Higher {property_name} → Higher p90 (higher interface bonds loss)")
            print(f"  Amino acids with HIGH {property_name} tend to have HIGH interface bonds loss in top 10%")
        else:
            print(f"\nINTERPRETATION: Higher {property_name} → Lower p90 (lower interface bonds loss)")
            print(f"  Amino acids with HIGH {property_name} tend to have LOW interface bonds loss in top 10%")
            print(f"  OR: Amino acids with LOW {property_name} tend to have HIGH interface bonds loss in top 10%")
        
        print(f"\nTop 10 amino acids by p90 value:")
        print(f"{'AA':<5} {'p90':<10} {'Property':<15} {'High Values Count':<20} {'Drives':<10}")
        print("-" * 80)
        for _, row in df_sorted.head(10).iterrows():
            high_vals_str = f"{len(row['High_Values'])} values ≥ {row['p90']:.2f}"
            drives = "✓" if row['Drives_Correlation'] else ""
            print(f"{row['Amino_Acid']:<5} {row['p90']:<10.2f} {row['Property']:<15.2f} {high_vals_str:<20} {drives:<10}")
        
        # Show amino acids that drive the correlation
        drivers = df_sorted[df_sorted['Drives_Correlation']]
        if len(drivers) > 0:
            if corr_direction == "positive":
                print(f"\nAmino acids driving the {corr_direction} correlation:")
                print(f"  (High p90 AND High {property_name} - these have high interface bonds loss AND high property values)")
            else:
                print(f"\nAmino acids driving the {corr_direction} correlation:")
                print(f"  (High p90 AND Low {property_name} - these have high interface bonds loss BUT low property values)")
            for _, row in drivers.iterrows():
                print(f"  {row['Amino_Acid']}: p90={row['p90']:.2f}, {property_name}={row['Property']:.2f}")
                if len(row['High_Values']) > 0:
                    print(f"    Top values in 90th+ percentile: {row['High_Values'][:10]}")
    
    # Save detailed results
    all_details = []
    for result in results:
        corr = result['correlation']
        df = result['data']
        for _, row in df.iterrows():
            all_details.append({
                'Protein': corr['protein'],
                'Property': corr['property'],
                'Correlation_r': corr['r'],
                'Correlation_p': corr['p'],
                'Correlation_Direction': result['correlation_direction'],
                'Amino_Acid': row['Amino_Acid'],
                'p90': row['p90'],
                'Property_Value': row['Property'],
                'Num_High_Values': row['Num_High_Values'],
                'Drives_Correlation': row['Drives_Correlation']
            })
    
    details_df = pd.DataFrame(all_details)
    details_df.to_csv('amino_acid_drivers_analysis.csv', index=False)
    print(f"\n\nDetailed results saved to: amino_acid_drivers_analysis.csv")
    
    return results

if __name__ == '__main__':
    analyze_significant_correlations()

