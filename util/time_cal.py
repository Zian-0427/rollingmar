import re

# --- Configuration ---

# The path to your log file
file_name = "/home/lizian/mar/bs256_iter100.log"

# The identifiers to search for
identifiers = [
    "Rolling MAR Enc-Dec runtime",
    "Rolling MAR Diff Head",
    "Rolling MAR runtime"
]

# ---------------------

# Initialize a dictionary to hold the totals for each identifier
totals_dict = {identifier: 0.0 for identifier in identifiers}
lines_processed_dict = {identifier: 0 for identifier in identifiers}

print(f"Starting processing for file: {file_name}")

try:
    with open(file_name, 'r') as f:
        for line_number, line in enumerate(f, 1):
            
            # Iterate through each identifier to see which one matches
            for identifier in identifiers:
                if identifier in line:
                    
                    # Found the identifier, now try to extract the time
                    # This regex looks for "runtime: 0.12345s"
                    match = re.search(r'runtime: (\d+\.?\d*)s', line)
                    
                    time_val = None
                    if match:
                        try:
                            time_val = float(match.group(1))
                        except ValueError:
                             print(f"  [Line {line_number}] WARNING: Found identifier '{identifier}' but could not convert time: {match.group(1)}")
                    
                    # Fallback: Check for time at the end of the line, e.g., "0.285733699798584s"
                    # This is based on your first example line
                    if time_val is None:
                        match_end = re.search(r'(\d+\.\d+)s$', line.strip())
                        if match_end:
                            try:
                                time_val = float(match_end.group(1))
                            except ValueError:
                                print(f"  [Line {line_number}] WARNING: Found identifier '{identifier}' but could not convert time (end-of-line): {match_end.group(1)}")

                    # If we successfully got a time, add it to the correct total
                    if time_val is not None:
                        totals_dict[identifier] += time_val
                        lines_processed_dict[identifier] += 1
                        # print(f"  [Line {line_number}] Matched '{identifier}', Added: {time_val}")

                    # Once we've matched an identifier for this line,
                    # we can stop checking other identifiers for the same line.
                    break 

    print("\n--- Results ---")
    print(f"Lines processed per identifier: {lines_processed_dict}")
    print("Total summed runtime per identifier:")
    print(totals_dict)

except FileNotFoundError:
    print(f"\nERROR: The file '{file_name}' was not found.")
    print("Please check the 'file_name' variable in the script.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")