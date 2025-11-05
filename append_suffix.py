#!/usr/bin/env python3
"""
Script to append columns from suffix.csv to a selected CSV file.
The script matches rows based on amino acid codes and appends all columns from suffix.csv.
"""

import csv
import sys
import os

def find_amino_acid_column(fieldnames):
    """Find the column that contains amino acid codes."""
    possible_names = ['class', 'inserted_residue', 'Amino Acid', 'amino_acid', 'residue']
    for col in possible_names:
        if col in fieldnames:
            return col
    return None

def load_suffix_data(suffix_file='suffix.csv'):
    """Load data from suffix.csv into a dictionary keyed by amino acid."""
    suffix_data = {}
    if not os.path.exists(suffix_file):
        print(f"Error: {suffix_file} not found!")
        sys.exit(1)
    
    with open(suffix_file, 'r') as f:
        reader = csv.DictReader(f)
        suffix_fieldnames = reader.fieldnames
        
        if not suffix_fieldnames:
            print(f"Error: {suffix_file} appears to be empty or invalid!")
            sys.exit(1)
        
        amino_acid_col = suffix_fieldnames[0]  # First column should be amino acid
        
        for row in reader:
            aa = row[amino_acid_col].strip()
            if aa:
                suffix_data[aa] = row
    
    print(f"Loaded {len(suffix_data)} amino acid entries from {suffix_file}")
    return suffix_data, suffix_fieldnames

def append_suffix_to_csv(target_file, suffix_file='suffix.csv', output_file=None):
    """Append columns from suffix.csv to target CSV file."""
    
    # Load suffix data
    suffix_data, suffix_fieldnames = load_suffix_data(suffix_file)
    
    if not os.path.exists(target_file):
        print(f"Error: {target_file} not found!")
        sys.exit(1)
    
    # Read target CSV
    with open(target_file, 'r') as f:
        reader = csv.DictReader(f)
        target_fieldnames = reader.fieldnames
        rows = list(reader)
    
    if not target_fieldnames:
        print(f"Error: {target_file} appears to be empty or invalid!")
        sys.exit(1)
    
    # Find the amino acid column in target file
    aa_col = find_amino_acid_column(target_fieldnames)
    if not aa_col:
        print(f"Error: Could not find amino acid column in {target_file}")
        print(f"Available columns: {', '.join(target_fieldnames[:10])}...")
        sys.exit(1)
    
    print(f"Using '{aa_col}' column for matching amino acids")
    
    # Determine output filename
    if output_file is None:
        base_name = os.path.splitext(target_file)[0]
        output_file = f"{base_name}_with_suffix.csv"
    
    # Get suffix columns to append (all except the amino acid column)
    amino_acid_col_suffix = suffix_fieldnames[0]
    suffix_cols_to_add = [col for col in suffix_fieldnames if col != amino_acid_col_suffix]
    
    # Check for existing columns to avoid duplicates
    existing_cols = set(target_fieldnames)
    cols_to_add = [col for col in suffix_cols_to_add if col not in existing_cols]
    
    if not cols_to_add:
        print("Warning: All suffix columns already exist in target file!")
    
    # Create new fieldnames
    new_fieldnames = list(target_fieldnames) + cols_to_add
    
    # Append suffix data to each row
    matched_count = 0
    unmatched_count = 0
    
    for row in rows:
        aa = row[aa_col].strip()
        
        # Handle cases where amino acid might have protein suffix (e.g., "A-1brs")
        if '-' in aa:
            aa_base = aa.split('-')[0].strip()
        else:
            aa_base = aa
        
        if aa_base in suffix_data:
            # Append all suffix columns
            for col in cols_to_add:
                row[col] = suffix_data[aa_base][col]
            matched_count += 1
        else:
            # Fill with empty values if no match
            for col in cols_to_add:
                row[col] = ''
            unmatched_count += 1
            if unmatched_count <= 5:  # Show first 5 unmatched
                print(f"  Warning: No match found for '{aa}' (base: '{aa_base}')")
    
    # Write output file
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nResults:")
    print(f"  Matched rows: {matched_count}")
    print(f"  Unmatched rows: {unmatched_count}")
    print(f"  Columns added: {len(cols_to_add)}")
    print(f"  Output saved to: {output_file}")
    
    return output_file

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python append_suffix.py <target_csv_file> [suffix_file] [output_file]")
        print("\nExample:")
        print("  python append_suffix.py 1brs_combined.csv")
        print("  python append_suffix.py 1brs_combined.csv suffix.csv")
        print("  python append_suffix.py 1brs_combined.csv suffix.csv output.csv")
        sys.exit(1)
    
    target_file = sys.argv[1]
    suffix_file = sys.argv[2] if len(sys.argv) > 2 else 'suffix.csv'
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    append_suffix_to_csv(target_file, suffix_file, output_file)

if __name__ == '__main__':
    main()

