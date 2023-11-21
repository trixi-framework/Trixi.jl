import argparse

def add_code_after_end(input_file, output_file, code_to_add):
    with open(input_file, 'r') as f:
        julia_code = f.read()

    # Split the Julia code by "end" keywords
    code_lines = julia_code.split("end")

    # Initialize the modified code with the first part of the code
    modified_code = code_lines[0]

    # Loop through each "end" and add the code after it, excluding the last "end"
    for i in range(1, len(code_lines) - 1):
        # Check for consecutive "end" keywords
        if code_lines[i].strip() == "":
            continue
        
        modified_code += f"{code_to_add}{code_lines[i]}"

    # Add the last "end" without the code
    modified_code += code_lines[-1]

    with open(output_file, 'w') as f:
        f.write(modified_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add code after 'end' keyword in a Julia source file.")
    parser.add_argument("input_file", help="Input Julia source file")

    args = parser.parse_args()

    code_to_add = """  # Ensure that we do not have excessive memory allocations 
    # (e.g., from type instabilities) 
    let 
      t = sol.t[end] 
      u_ode = sol.u[end] 
      du_ode = similar(u_ode) 
      @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000 
    end
  end"""

    add_code_after_end(args.input_file, args.input_file, code_to_add)
