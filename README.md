# Homomorphic Minimax Comparison with SEAL

This Python script demonstrates how to perform a homomorphic minimax comparison using the Simple Encrypted Arithmetic Library (SEAL). The goal is to securely compare two floating-point numbers while preserving privacy through encryption and homomorphic operations.

## Requirements

- Python 3.x
- [Microsoft SEAL](https://github.com/microsoft/SEAL) library for homomorphic encryption

## Installation

1. Install the Microsoft SEAL library by following the instructions on the official [SEAL GitHub repository](https://github.com/microsoft/SEAL).

2. Clone this repository or copy the script into your project directory.

3. Make sure you have the necessary Python packages installed. You can install them using pip:

## Usage

1. Modify the `input_pairs` list with the floating-point numbers you want to compare. You can add more pairs as needed.

2. Configure the parameters for the comparison, including `alpha`, `epsilon`, `depth`, and `margin`.

3. Run the script using Python:

4. The script will perform homomorphic minimax comparisons on the input pairs and display the results, indicating whether the first number is less than, equal to, or greater than the second number.

5. The running time for each comparison operation and the total running time will also be printed.

## Example

```python
# Example inputs for comparison
input_pairs = [
 (1.0001, 1),
 (1, 1),
 (0, 0),
 (10, 1),
 (-10, 0)    
 # Add more pairs as needed
]