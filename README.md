### Numpy

#### Array creation
Create arrays using `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`, etc.  

#### Check shape , Reshape , Flatten
```
# Reshape an array
reshaped = arr.reshape()

# Flatten an array
flattened = matrix.flatten()

```
#### Array Indexing and Slicing
```
# Indexing
element = arr[2]

# Slicing
slice_arr = arr[1:4]

# Boolean indexing
mask = arr > 2
filtered_arr = arr[mask]

```

#### Mathematical operations
```
# Element-wise addition
add_result = arr + 2

# Element-wise multiplication
mul_result = arr * 2

# Element-wise square
square_result = np.square(arr)

```

#### Mathematical Functions
```
# Trigonometric functions
sin_arr = np.sin(arr)

# Exponentiation
exp_arr = np.exp(arr)

# Square root
sqrt_arr = np.sqrt(arr)

```

#### Linear Algabra
```
# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
matrix_mult = np.dot(A, B)

# Matrix inverse
A_inv = np.linalg.inv(A)

# Matrix determinant
det = np.linalg.det(A)

# Eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(A)

```

#### Statistical Operations
```
# Mean
mean_val = np.mean(arr)

# Median
median_val = np.median(arr)

# Standard deviation
std_dev = np.std(arr)

# Sum of elements
sum_val = np.sum(arr)

# Variance
variance_val = np.var(arr)

```

#### Random Number Generation
```
# Random integers
rand_integers = np.random.randint(0, 10, 5)

# Random floats from uniform distribution
rand_floats = np.random.rand(5)

# Random numbers from normal distribution
rand_norm = np.random.randn(5)

# Random choice from an array
rand_choice = np.random.choice(arr, size=3)

# Seed for reproducibility
np.random.seed(42)

```

#### Fourier Transform
```
# Compute Fourier Transform
fft_result = np.fft.fft(signal)

# Compute inverse Fourier Transform
ifft_result = np.fft.ifft(fft_result)
```
#### Polynomial Operations
```
# Define a polynomial: x^2 - 3x + 2
coeffs = [1, -3, 2]

# Find roots of the polynomial
roots = np.roots(coeffs)

# Evaluate polynomial at specific values
values = np.polyval(coeffs, [0, 1, 2])

```

#### Save and Load 
```
# Save array to file
np.save('array.npy', arr)

# Load array from file
loaded_arr = np.load('array.npy')

```

#### concatenation and split
```
# Concatenate arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
concat_arr = np.concatenate((arr1, arr2))

# Split an array
split_arr = np.split(arr, 2)

```

#### Indexing
```
# Indexing with a list
indices = [0, 2, 4]
fancy_index = arr[indices]

# Indexing with boolean array
mask = np.array([True, False, True, False, True])
fancy_boolean = arr[mask]

```

#### AND , OR , NOT
```
# Logical AND
logical_and = np.logical_and(arr > 2, arr < 5)

# Logical OR
logical_or = np.logical_or(arr == 1, arr == 4)

# Logical NOT
logical_not = np.logical_not(arr > 3)

```

#### Array Comparison and Searching
```
# Maximum and minimum
max_val = np.max(arr)
min_val = np.min(arr)

# Argmax and Argmin (index of max/min)
argmax_val = np.argmax(arr)
argmin_val = np.argmin(arr)

# Search for an element in an array
found = np.where(arr == 3)

```

#### Array Stacking
```
# Stack arrays vertically
vstack_arr = np.vstack((arr1, arr2))

# Stack arrays horizontally
hstack_arr = np.hstack((arr1, arr2))

```
