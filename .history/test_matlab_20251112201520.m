% Simple MATLAB Test Program
% This script tests basic MATLAB functionality to verify extensions are working

fprintf('=== MATLAB Functionality Test ===\n\n');

% Test 1: Basic arithmetic operations
fprintf('Test 1: Basic arithmetic operations\n');
a = 5;
b = 3;
sum_result = a + b;
diff_result = a - b;
mult_result = a * b;
div_result = a / b;
fprintf('%d + %d = %d\n', a, b, sum_result);
fprintf('%d - %d = %d\n', a, b, diff_result);
fprintf('%d * %d = %d\n', a, b, mult_result);
fprintf('%d / %d = %.2f\n\n', a, b, div_result);

% Test 2: Matrix operations
fprintf('Test 2: Matrix operations\n');
A = [1, 2, 3; 4, 5, 6; 7, 8, 9];
B = [9, 8, 7; 6, 5, 4; 3, 2, 1];
C = A * B;
fprintf('Matrix A:\n');
disp(A);
fprintf('Matrix B:\n');
disp(B);
fprintf('Matrix C = A * B:\n');
disp(C);

% Test 3: Matrix properties
fprintf('\nTest 3: Matrix properties\n');
fprintf('Size of A: %dx%d\n', size(A,1), size(A,2));
fprintf('Determinant of A: %.2f\n', det(A));
fprintf('Rank of A: %d\n', rank(A));
fprintf('Trace of A: %d\n\n', trace(A));

% Test 4: Mathematical functions
fprintf('Test 4: Mathematical functions\n');
x = 0:pi/10:2*pi;
y_sin = sin(x);
y_cos = cos(x);
y_exp = exp(x(1:6)); % First 6 elements to avoid very large numbers
fprintf('sin(π/2) = %.4f\n', sin(pi/2));
fprintf('cos(π) = %.4f\n', cos(pi));
fprintf('exp(1) = %.4f\n', exp(1));
fprintf('log(10) = %.4f\n', log(10));
fprintf('sqrt(16) = %.4f\n\n', sqrt(16));

% Test 5: String operations
fprintf('Test 5: String operations\n');
str1 = 'Hello';
str2 = 'MATLAB';
combined = [str1, ' ', str2, '!'];
fprintf('Combined string: %s\n', combined);
fprintf('Length of string: %d\n', length(combined));
fprintf('Upper case: %s\n\n', upper(combined));

% Test 6: Control structures
fprintf('Test 6: Control structures\n');
for i = 1:5
    if mod(i, 2) == 0
        fprintf('%d is even\n', i);
    else
        fprintf('%d is odd\n', i);
    end
end

% Test 7: Array operations
fprintf('\nTest 7: Array operations\n');
numbers = 1:10;
fprintf('Original array: ');
disp(numbers);
fprintf('Sum: %d\n', sum(numbers));
fprintf('Mean: %.2f\n', mean(numbers));
fprintf('Standard deviation: %.2f\n', std(numbers));
fprintf('Maximum: %d\n', max(numbers));
fprintf('Minimum: %d\n\n', min(numbers));

% Test 8: Date and time functions
fprintf('Test 8: Date and time functions\n');
current_time = datetime('now');
fprintf('Current date and time: %s\n', datestr(current_time));
fprintf('Year: %d\n', year(current_time));
fprintf('Month: %d\n', month(current_time));
fprintf('Day: %d\n\n', day(current_time));

% Test 9: Random number generation
fprintf('Test 9: Random number generation\n');
rng(42); % Set seed for reproducibility
random_uniform = rand(1, 5);
random_normal = randn(1, 5);
fprintf('Uniform random numbers: ');
fprintf('%.4f ', random_uniform);
fprintf('\nNormal random numbers: ');
fprintf('%.4f ', random_normal);
fprintf('\n\n');

% Test 10: Simple plotting (if graphics is available)
fprintf('Test 10: Plotting test\n');
try
    figure('Visible', 'off'); % Create figure without displaying
    x_plot = 0:0.1:2*pi;
    y1_plot = sin(x_plot);
    y2_plot = cos(x_plot);
    
    plot(x_plot, y1_plot, 'b-', 'LineWidth', 2);
    hold on;
    plot(x_plot, y2_plot, 'r--', 'LineWidth', 2);
    hold off;
    
    title('Sin and Cos Functions');
    xlabel('x');
    ylabel('y');
    legend('sin(x)', 'cos(x)');
    grid on;
    
    % Save the plot
    saveas(gcf, 'test_plot.png');
    fprintf('Plot created and saved as test_plot.png\n');
    close(gcf);
catch ME
    fprintf('Plotting failed: %s\n', ME.message);
    fprintf('This might be normal in headless environments\n');
end

% Test 11: File I/O operations
fprintf('\nTest 11: File I/O operations\n');
try
    % Write test data to file
    test_data = [1, 2, 3; 4, 5, 6];
    save('test_data.mat', 'test_data');
    fprintf('Data saved to test_data.mat\n');
    
    % Read it back
    loaded = load('test_data.mat');
    fprintf('Data loaded successfully\n');
    fprintf('Loaded data:\n');
    disp(loaded.test_data);
    
    % Clean up
    delete('test_data.mat');
    fprintf('Test file cleaned up\n');
catch ME
    fprintf('File I/O test failed: %s\n', ME.message);
end

% Test 12: System information
fprintf('\nTest 12: System information\n');
fprintf('MATLAB version: %s\n', version);
fprintf('Operating system: %s\n', computer);

fprintf('\n=== All MATLAB Tests Completed ===\n');
fprintf('If you see this message, MATLAB is working properly!\n');

% Clean up any remaining plot files
if exist('test_plot.png', 'file')
    delete('test_plot.png');
end
