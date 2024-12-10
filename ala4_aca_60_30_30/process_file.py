def process_file(file_name, n, stride):
    try:
        with open(file_name, 'r') as infile:
            lines = infile.readlines()

        # Initialize counters and output file index
        output_index = 0
        i = 0  # Start at 0-based index

        while i < len(lines):
            output_lines = []
            output_count = 0

            # Read lines and store every 'stride'-th line, up to 'n' lines
            while output_count < n and i < len(lines):
                output_lines.append(lines[i].strip())  # Store the current line
                output_count += 1
                i += stride  # Move to the next valid line, skipping 'stride' lines

            # Write the stored lines to the output file
            if output_lines:
                output_file_name = f"{file_name}.{output_index}"
                with open(output_file_name, 'w') as outfile:
                    outfile.write("\n".join(output_lines))
                print(f"Written to {output_file_name}")

            # Increment output file index
            output_index += 1

    except FileNotFoundError:
        print(f"Error: The file '{file_name}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
if __name__ == "__main__":
    FILE = input("Enter the input file name: ")
    n = int(input("Enter the number of lines to output per file (n): "))
    stride = int(input("Enter the stride (lines to skip between outputs): "))

    process_file(FILE, n, stride)
