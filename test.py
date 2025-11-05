#!/usr/bin/env python3
"""
Script to compute min/max values for interface bonds loss CSV files,
assign a scale from N groups (user-selectable) to each value,
and plot the scale distribution as ASCII bar plots.
Additionally, print the cell locations for all 'max loss' entries per file.

Usage:
    python <script.py> [--groups N]

Default grouping: 9
"""

import csv
import os
import glob
import argparse
import numpy as np
import pandas as pd

def get_default_scale_labels(n):
    """Return scale group names for a given number of groups (n)."""
    # Predefined names for n=9 (legacy)
    if n == 9:
        return [
            'none', 'extremely low', 'very low', 'low', 'moderate',
            'moderately high', 'high', 'very high', 'max loss'
        ]
    # Predefined names for some other common n
    if n == 5:
        return ['none', 'low', 'moderate', 'high', 'max loss']
    if n == 3:
        return ['none', 'moderate', 'max loss']
    # Otherwise just numerical names, ensuring 'none' and 'max loss'
    scale_labels = ['none']
    for i in range(1, n-1):
        scale_labels.append(f"group {i+1}")
    scale_labels.append('max loss')
    return scale_labels

def assign_scale(value, min_val, max_val, n_groups=9, scale_labels=None):
    """
    Assign scale based on value position between min and max using n_groups.
    Optionally use given scale_labels (must be of length n_groups).
    """
    if pd.isna(value) or value == '':
        return ''
    
    value = float(value)

    # Handle case where min == max
    if min_val == max_val:
        if value == min_val:
            return scale_labels[0] if scale_labels else 'none'
        else:
            return scale_labels[0] if scale_labels else 'none'

    range_val = max_val - min_val
    position = (value - min_val) / range_val  # in [0, 1]

    # Determine which bin
    # Edges: [0, 1/n), [1/n, 2/n), ... [n-2/n, n-1/n), [n-1/n, 1]
    bin_width = 1.0 / n_groups
    label_idx = int(position / bin_width)
    if position >= 1.0:  # Exactly max, put in final bin
        label_idx = n_groups - 1
    if label_idx >= n_groups:
        label_idx = n_groups - 1
    # Sanity: if negative position, clamp to zero
    if label_idx < 0:
        label_idx = 0
    label = (scale_labels[label_idx]
             if scale_labels is not None and len(scale_labels) == n_groups
             else f"group {label_idx + 1}")
    return label

def ascii_bar(count, max_count, max_width=40, symbol='#'):
    """Produce an ASCII bar with width proportional to count/max_count."""
    if max_count == 0:
        return ''
    width = int(round((count / max_count) * max_width))
    return symbol * width

def plot_ascii_distribution(scale_counts, total_count, scales, max_bar_width=40):
    """Print an ASCII bar plot of the scale distribution."""
    max_count = max(scale_counts.values()) if scale_counts else 1
    print("  Scale distribution:")
    for scale in scales:
        count = scale_counts.get(scale, 0)
        percentage = (count / total_count) * 100 if total_count > 0 else 0
        bar = ascii_bar(count, max_count, max_bar_width)
        print(f"    {scale.ljust(18)}: {str(count).rjust(5)} ({percentage:5.1f}%) |{bar}")

def process_interface_bonds_loss_file(filename, n_groups=9, scale_labels=None):
    """Process a single interface bonds loss CSV file."""
    print(f"\nProcessing {filename}...")
    
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # Get all numeric columns (exclude 'class' column)
    numeric_cols = [col for col in df.columns if col != 'class']
    
    # Extract all numeric values (excluding NaN and empty strings)
    all_values = []
    for col in numeric_cols:
        for val in df[col]:
            if pd.notna(val) and val != '':
                try:
                    all_values.append(float(val))
                except (ValueError, TypeError):
                    continue

    if len(all_values) == 0:
        print(f"  Warning: No valid numeric values found in {filename}")
        return None

    # Calculate min and max
    min_val = min(all_values)
    max_val = max(all_values)

    print(f"  Min value: {min_val}")
    print(f"  Max value: {max_val}")
    print(f"  Total numeric values: {len(all_values)}")
    print(f"  Group count: {n_groups}")

    # Create scale columns more efficiently using vectorized operations
    if scale_labels is None or len(scale_labels) != n_groups:
        scale_labels = get_default_scale_labels(n_groups)

    scale_columns = {}
    for col in numeric_cols:
        scale_col = f"{col}_scale"
        scale_columns[scale_col] = df[col].apply(lambda x: assign_scale(x, min_val, max_val, n_groups, scale_labels))
    
    # Create a copy of the dataframe and add all scale columns at once
    df_scaled = df.copy()
    scale_df = pd.DataFrame(scale_columns)
    df_scaled = pd.concat([df_scaled, scale_df], axis=1)

    # Determine output filename
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}_scaled.csv"

    # Save the scaled version
    df_scaled.to_csv(output_filename, index=False)
    print(f"  Saved scaled version to: {output_filename}")

    # Print summary statistics and ASCII histogram
    scale_counts = {}
    for col in numeric_cols:
        scale_col = f"{col}_scale"
        for scale in df_scaled[scale_col]:
            if scale and scale != '':
                scale_counts[scale] = scale_counts.get(scale, 0) + 1

    # Print ASCII bar chart only (no simple tabular distribution)
    plot_ascii_distribution(scale_counts, len(all_values), scale_labels)

    # --- Print cell locations of all 'min loss' entries ---
    print("  Locations of 'min loss' entries:")
    max_print = 10
    min_loss_cells = []
    for col in numeric_cols:
        scale_col = f"{col}_scale"
        for idx, val in enumerate(df_scaled[scale_col]):
            if val == 'none':
                min_loss_cells.append((idx, col, df_scaled.iloc[idx][col]))
    if min_loss_cells:
        for idx, (row_idx, col, value) in enumerate(min_loss_cells):
            if idx < max_print:
                print(f"    Cell: row {row_idx+1}, column '{col}' (value={value})")
        if len(min_loss_cells) > max_print:
            print(f"    (... {len(min_loss_cells) - max_print} more)")
    else:
        print("    (None found)")

    # --- Print cell locations of all 'max loss' entries ---
    print("  Locations of 'max loss' entries:")
    has_max_loss = False
    for col in numeric_cols:
        scale_col = f"{col}_scale"
        # Use enumerate to get (row_index, scale_label)
        for idx, val in enumerate(df_scaled[scale_col]):
            if val == 'max loss':
                has_max_loss = True
                row_label = (
                    df_scaled.index[idx]
                    if df_scaled.index is not None
                    else idx
                )
                print(f"    Cell: row {idx+1}, column '{col}' (value={df_scaled.iloc[idx][col]})")
    if not has_max_loss:
        print("    (None found)")
    # ------------------

    return {
        'filename': filename,
        'min_val': min_val,
        'max_val': max_val,
        'output_filename': output_filename
    }

def main():
    """Main function to process all interface bonds loss CSV files."""
    parser = argparse.ArgumentParser(
        description="Compute min/max values for interface bonds loss CSV files and assign scales from N groups."
    )
    parser.add_argument("--groups", "-g", type=int, default=9,
                        help="Number of groups/bins to scale values into (default: 9)")
    args = parser.parse_args()
    n_groups = args.groups
    scale_labels = get_default_scale_labels(n_groups)

    # Find all interface bonds loss CSV files
    pattern = '*_interface_bonds_loss.csv'
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(files)} interface bonds loss CSV files:")
    for f in files:
        print(f"  - {f}")
    
    # Process each file
    results = []
    for filename in sorted(files):
        result = process_interface_bonds_loss_file(filename, n_groups, scale_labels)
        if result:
            results.append(result)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")
    print(f"{'Filename':<40} {'Min':<10} {'Max':<10}")
    print(f"{'-' * 60}")
    for result in results:
        print(f"{result['filename']:<40} {result['min_val']:<10.2f} {result['max_val']:<10.2f}")
    
    print(f"\n{'=' * 60}")
    print(f"Processed {len(results)} files successfully!")

if __name__ == '__main__':
    main()
