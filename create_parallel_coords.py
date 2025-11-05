#!/usr/bin/env python3
"""
Script to create parallel coordinates plots for each *_interface_bonds_loss_with_suffix.csv file.

Usage:
    python create_parallel_coords.py
"""

import glob
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

def create_parallel_coords_plot(csv_file):
    """
    Create a parallel coordinates plot for a CSV file.
    
    Args:
        csv_file: Path to the CSV file
    """
    print(f"Processing {csv_file}...")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get the protein name from filename
    protein_name = os.path.basename(csv_file).replace('_interface_bonds_loss_with_suffix.csv', '')
    
    # Separate the 'class' column (amino acid types) for coloring
    if 'class' not in df.columns:
        print(f"  Warning: 'class' column not found in {csv_file}")
        return
    
    class_col = df['class']
    
    # Get all numeric columns (exclude 'class')
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    
    # Remove 'class' from numeric columns if it's there
    if 'class' in numeric_cols:
        numeric_cols.remove('class')
    
    if len(numeric_cols) == 0:
        print(f"  Warning: No numeric columns found in {csv_file}")
        return
    
    print(f"  Found {len(numeric_cols)} numeric columns")
    print(f"  Found {len(df)} rows")
    
    # Create a subset dataframe with only numeric columns and class
    plot_df = df[['class'] + numeric_cols].copy()
    
    # Convert numeric columns to numeric type (handling any string values)
    for col in numeric_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    
    # Remove rows where all numeric values are NaN
    plot_df = plot_df.dropna(subset=numeric_cols, how='all')
    
    # Get unique classes for color mapping
    unique_classes = sorted(plot_df['class'].unique())
    n_classes = len(unique_classes)
    
    # Create a numeric mapping for color (plotly needs numeric values)
    class_to_num = {cls: i for i, cls in enumerate(unique_classes)}
    plot_df['class_num'] = plot_df['class'].map(class_to_num)
    
    # Create parallel coordinates plot using plotly express
    # Limit to first 50 dimensions for readability
    dimensions_to_plot = numeric_cols[:50]
    
    # Use numeric color mapping
    fig = px.parallel_coordinates(
        plot_df,
        dimensions=dimensions_to_plot,
        color='class_num',
        labels={col: col for col in dimensions_to_plot},
        title=f'Parallel Coordinates Plot: {protein_name}',
        color_continuous_scale=px.colors.qualitative.Set3
    )
    
    # Update colorbar to show class names
    if n_classes > 0:
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Amino Acid",
                tickmode='array',
                tickvals=list(range(n_classes)),
                ticktext=list(unique_classes),
                len=0.6
            )
        )
    
    # Update layout for better readability
    fig.update_layout(
        width=1800,
        height=800,
        title_font_size=16,
        font=dict(size=10)
    )
    
    # Save as HTML
    output_file = f"{protein_name}_parallel_coords.html"
    fig.write_html(output_file)
    print(f"  Saved plot to: {output_file}")
    
    # Also create a version with all dimensions (may be very wide)
    if len(numeric_cols) > 50:
        print(f"  Creating full version with all {len(numeric_cols)} dimensions...")
        fig_full = px.parallel_coordinates(
            plot_df,
            dimensions=numeric_cols,
            color='class_num',
            labels={col: col for col in numeric_cols},
            title=f'Parallel Coordinates Plot (All Dimensions): {protein_name}',
            color_continuous_scale=px.colors.qualitative.Set3
        )
        
        # Update colorbar to show class names
        if n_classes > 0:
            fig_full.update_layout(
                coloraxis_colorbar=dict(
                    title="Amino Acid",
                    tickmode='array',
                    tickvals=list(range(n_classes)),
                    ticktext=list(unique_classes),
                    len=0.6
                )
            )
        
        fig_full.update_layout(
            width=max(1800, len(numeric_cols) * 30),  # Scale width based on number of dimensions
            height=800,
            title_font_size=16,
            font=dict(size=9)
        )
        
        output_file_full = f"{protein_name}_parallel_coords_full.html"
        fig_full.write_html(output_file_full)
        print(f"  Saved full plot to: {output_file_full}")

def main():
    """Main function to process all matching CSV files."""
    # Find all files matching the pattern
    pattern = "*_interface_bonds_loss_with_suffix.csv"
    csv_files = sorted(glob.glob(pattern))
    
    if len(csv_files) == 0:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(csv_files)} file(s) matching pattern:")
    for f in csv_files:
        print(f"  - {f}")
    print()
    
    # Process each file
    for csv_file in csv_files:
        try:
            create_parallel_coords_plot(csv_file)
            print()
        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")
            print()
    
    print("Done!")

if __name__ == "__main__":
    main()

