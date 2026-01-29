# MATLAB Test Program

This directory contains a simple MATLAB test program (`test_matlab.m`) designed to verify that MATLAB extensions and basic functionality are working correctly.

## Purpose

The test program checks various MATLAB capabilities to ensure your MATLAB installation and extensions are functioning properly.

## Test Coverage

The program tests the following MATLAB features:

1. **Basic arithmetic operations** - Addition, subtraction, multiplication, division
2. **Matrix operations** - Matrix multiplication, display, and properties
3. **Mathematical functions** - Trigonometric, exponential, logarithmic functions
4. **String operations** - String concatenation and manipulation
5. **Control structures** - For loops and conditional statements
6. **Array operations** - Statistical functions on arrays
7. **Date and time functions** - Current time and date operations
8. **Random number generation** - Uniform and normal distributions
9. **Plotting functionality** - Basic 2D plotting and figure handling
10. **File I/O operations** - Saving and loading .mat files
11. **System information** - MATLAB version and OS information

## How to Run

### Using MATLAB Desktop:
1. Open MATLAB
2. Navigate to the directory containing `test_matlab.m`
3. Run the script by typing: `test_matlab`
4. Or click "Run" in the MATLAB editor

### Using MATLAB Command Line:
```bash
matlab -batch "test_matlab"
```

### Using MATLAB Online:
1. Upload `test_matlab.m` to MATLAB Online
2. Open the file and click "Run"

## Expected Output

The program will output test results for each category. If MATLAB is working properly, you should see:
- All arithmetic results displayed correctly
- Matrix operations showing proper calculations
- Mathematical functions returning expected values
- System information showing your MATLAB version
- A final message: "If you see this message, MATLAB is working properly!"

## Troubleshooting

### If plotting fails:
- This is normal in headless environments
- The program handles this gracefully with try-catch blocks

### If file I/O fails:
- Check directory permissions
- Ensure MATLAB has write access to the current directory

### If any test fails:
- Check MATLAB installation
- Verify required toolboxes are installed
- Check if running in restricted environments

## Clean Up

The program automatically cleans up temporary files created during testing, including:
- `test_data.mat` (file I/O test)
- `test_plot.png` (plotting test)

## Customization

You can modify the script to:
- Add additional tests specific to your needs
- Change test parameters
- Add custom functionality checks

## Requirements

- MATLAB R2016b or later
- No special toolboxes required (basic MATLAB functionality only)
