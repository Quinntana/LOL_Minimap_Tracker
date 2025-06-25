import pandas as pd
from pathlib import Path

def delete_and_shift_rows(csv_file, a, b, output_file=None):
    """
    Deletes rows from index a to b (inclusive) and shifts remaining rows up.
    
    Args:
        csv_file (str): Path to input CSV file
        a (int): First row to delete (0-based index)
        b (int): Last row to delete (0-based index)
        output_file (str): Path to save modified CSV (if None, overwrites input)
    
    Raises:
        FileNotFoundError: If csv_file doesn't exist
        ValueError: If a or b is invalid
    """
    # Check if file exists
    if not pd.io.common.file_exists(csv_file):
        raise FileNotFoundError(f"Input file '{csv_file}' not found")

    # Read the CSV file
    df = pd.read_csv(csv_file)
    num_rows = len(df)

    # Validate indices
    if a < 0 or b < 0:
        raise ValueError("Indices 'a' and 'b' must be non-negative")
    if a > b:
        raise ValueError(f"Start index 'a' ({a}) must be less than or equal to end index 'b' ({b})")
    if b >= num_rows:
        raise ValueError(f"End index 'b' ({b}) exceeds number of rows ({num_rows})")
    if a >= num_rows:
        raise ValueError(f"Start index 'a' ({a}) exceeds number of rows ({num_rows})")

    # Split the dataframe into parts
    part_before = df.iloc[:a]       # Rows before deletion range
    part_after = df.iloc[b+1:]      # Rows after deletion range
    
    # Concatenate the parts, effectively deleting a to b
    new_df = pd.concat([part_before, part_after], ignore_index=True)
    
    # Check if result is empty
    if new_df.empty:
        print("Warning: Resulting DataFrame is empty after deletion")

    # Save the result
    if output_file is None:
        output_file = csv_file
    new_df.to_csv(output_file, index=False)
    print(f"Deleted rows {a} to {b} and saved to {output_file}")

# Example usage
try:
    delete_and_shift_rows(Path(__file__).parent / 'timeline.csv', 2, 2749, Path(__file__).parent / 'output.csv')
except Exception as e:
    print(f"Error: {e}")