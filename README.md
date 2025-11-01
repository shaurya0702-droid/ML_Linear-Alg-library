# 📐 Mini Linear Algebra Library

## Overview
This project is a **from-scratch implementation of fundamental Linear Algebra functions** in Python. Instead of just using NumPy, we build the core algorithms ourselves to understand how linear algebra operations really work under the hood! It's like learning to cook by understanding recipes instead of just using a microwave.

**Why does this matter?**  
Linear algebra is the **foundation of machine learning, computer graphics, physics simulations, and data science**. By building these functions yourself, you'll truly understand what's happening when you use libraries like NumPy, TensorFlow, or scikit-learn!

## Features
- 🔢 **Matrix Operations:** Addition, subtraction, multiplication from scratch
- 📊 **Vector Operations:** Dot products, cross products, norms
- 🔄 **Matrix Transformations:** Transpose, determinant, inverse
- 🎯 **Decompositions:** Eigenvalues, eigenvectors, matrix factorization
- 📈 **Solving Systems:** Gaussian elimination, solving linear equations
- 🎓 **Educational:** Clean code with detailed explanations of every algorithm
- 💡 **Beginner Friendly:** Perfect for learning how algorithms work

## What You'll Learn
- How to implement matrix addition and multiplication
- Understanding determinants and matrix inverses
- Solving systems of linear equations (Ax = b)
- Eigenvalues and eigenvectors concepts
- Vector spaces and linear transformations
- Why NumPy is fast (and why we need it!)

## Installation

### Step 1: Clone the Repository
Download this project to your computer:
```bash
git clone https://github.com/shaurya0702-droid/ML_Linear-Alg-library.git
cd ML_Linear-Alg-library
```

### Step 2: Install Required Libraries
Make sure you have Python installed, then install dependencies:
```bash
pip install numpy matplotlib jupyter
```

### Step 3: Open the Notebook
Launch Jupyter Notebook:
```bash
jupyter notebook
```
Then open `mini_linear_algebra.ipynb`

## Usage

### Running the Project
1. Open the Jupyter notebook
2. Run each cell from top to bottom (Press `Shift + Enter`)
3. See how linear algebra operations work step by step!

### What the Code Covers:
```python
# 1. Create vectors and matrices
# 2. Basic operations (addition, subtraction)
# 3. Matrix multiplication (the heart of ML!)
# 4. Vector operations (dot product, norms)
# 5. Matrix properties (trace, transpose, determinant)
# 6. Inverse and solving linear systems
# 7. Eigenvalues and eigenvectors
# 8. Compare with NumPy for verification
```

### Example Usage:
```python
# Create a simple matrix
A = [[1, 2], [3, 4]]

# Calculate determinant
det_A = determinant(A)  # Output: -2.0

# Find inverse
A_inv = matrix_inverse(A)

# Solve Ax = b
b = [5, 6]
x = solve_linear_system(A, b)
```

## Core Concepts Explained (Simple Terms)

### Vectors 🎯
A vector is just a list of numbers arranged in a line (or column).
- **Example:** [1, 2, 3] is a vector with 3 components
- **Use:** Represent positions, velocities, or features

### Matrices 📦
A matrix is a grid of numbers (think Excel spreadsheet).
- **Example:** [[1, 2], [3, 4]] is a 2×2 matrix
- **Use:** Store multiple vectors, represent transformations

### Dot Product (Inner Product) 🔗
Multiply corresponding elements and sum them up.
- **Formula:** a · b = a₁b₁ + a₂b₂ + a₃b₃ + ...
- **Why:** Measures how similar two vectors are

### Matrix Multiplication ✖️
Combine two matrices to create a new one (used in neural networks!).
- **Rule:** Number of columns in first matrix = number of rows in second matrix
- **Output:** Each element is the dot product of a row and column

### Transpose 🔄
Flip a matrix so rows become columns.
- **Example:** [[1, 2], [3, 4]]ᵀ = [[1, 3], [2, 4]]
- **Why:** Useful for solving problems and math operations

### Determinant 🎲
A special number that tells us about a matrix's properties.
- **For 2×2:** det = ad - bc (where matrix is [[a,b],[c,d]])
- **Meaning:** If det = 0, matrix doesn't have an inverse

### Inverse Matrix 🔙
The "opposite" of a matrix. If A × A⁻¹ = I (identity matrix).
- **Like:** 5 × (1/5) = 1 (but for matrices!)
- **Use:** Solve equations and undo transformations

### Eigenvalues & Eigenvectors 🌟
Special vectors that only stretch (not rotate) when multiplied by a matrix.
- **Eigenvalue:** The stretch factor
- **Eigenvector:** The direction that stretches
- **Why:** Critical for understanding matrix behavior

### Gaussian Elimination 📋
A systematic way to solve systems of linear equations.
- **Steps:** Row operations to simplify matrix → back substitution
- **Goal:** Transform messy equations into easy ones

## Linear Algebra Concepts Used

| Concept | What It Is | Why It Matters |
|---------|-----------|----------------|
| **Vector Addition** | Adding two vectors element-wise | Combining forces, movements, features |
| **Scalar Multiplication** | Multiplying a vector by a number | Scaling, resizing, changing magnitude |
| **Dot Product** | Multiply elements and sum | Similarity measure, projections |
| **Matrix Multiplication** | Combining transformations | Neural networks, rotations, scaling |
| **Determinant** | Special number describing matrix | Checking if inverse exists |
| **Matrix Inverse** | Reversing a transformation | Solving Ax = b equations |
| **Eigenvalues** | Stretch factors of eigenvectors | PCA, stability analysis |
| **Eigenvectors** | Special direction vectors | Finding principal components |

## Environment Variables
No environment variables needed for this project!

## Running Tests
Test your implementations against NumPy:
```python
import numpy as np

# Your implementation
your_result = matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])

# NumPy verification
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
numpy_result = np.dot(A, B)

# Compare
print(your_result == numpy_result.tolist())  # Should be True!
```

## FAQ

**Q: Do I need to know advanced math?**  
**A:** No! This project teaches you as you go. Start with basic operations and build up.

**Q: Why build from scratch instead of using NumPy?**  
**A:** Because understanding the "why" helps you use NumPy better. Plus, you'll impress interviewers!

**Q: How does this relate to Machine Learning?**  
**A:** Almost every ML algorithm uses linear algebra:
- Neural networks use matrix multiplication
- Principal Component Analysis uses eigenvalues
- Gradient descent uses derivatives (calculus + linear algebra)

**Q: Will my code be as fast as NumPy?**  
**A:** No! NumPy uses optimized C code. But your code will be clear and understandable (and fast enough for learning).

**Q: Can I use this in real projects?**  
**A:** For learning—yes! For production—use NumPy/SciPy. They're battle-tested and optimized.

**Q: What happens if I divide by zero?**  
**A:** The code will crash (or give an error). Real libraries handle this with checks. You could add them!

## Screenshots
*(Add screenshots of your notebook showing:)*
- Visualization of matrix operations
- Comparison plots: Your implementation vs NumPy
- Graphs showing eigenvalues and eigenvectors
- Step-by-step calculation displays

**Example screenshots to include:**
- Matrix multiplication visualization
- Determinant calculation steps
- Eigenvalue decomposition graphs
- System of equations being solved

## Roadmap
Future improvements planned:
- ✅ Add matrix decomposition (LU, QR, SVD)
- ✅ Implement more advanced operations
- ✅ Add visualization of geometric transformations
- ✅ Create web app to visualize operations
- ✅ Add numerical stability improvements
- ✅ Optimize performance using NumPy tricks

## Contributing
Want to improve this library? Great!
1. Fork this repository
2. Add new functions or improvements
3. Submit a pull request

**Ideas for contribution:**
- Add more decomposition methods
- Improve numerical stability
- Add performance optimizations
- Create visualization tools
- Add support for complex numbers
- Write comprehensive tests

## Authors
**Shaurya Rawat**  
First-year B.Tech Engineering Student | Machine Learning & Mathematics Enthusiast

Connect with me:
- 🐙 GitHub: [shaurya0702-droid](https://github.com/shaurya0702-droid)
- 💼 LinkedIn: [Shaurya Rawat](https://www.linkedin.com/in/shaurya-rawat-714349366)

## Acknowledgements
- **Inspiration:** Andrew Ng's ML course, 3Blue1Brown's Linear Algebra series
- **Resources:** NumPy documentation, Gilbert Strang's Linear Algebra MIT course
- **Math Resources:** Khan Academy, MIT OpenCourseWare
- **Community:** Stack Overflow, Math Stack Exchange

## License
MIT License - Feel free to use this project for learning and educational purposes!

---

## Github Profile - About Me
🎓 B.Tech Engineering Student passionate about Mathematics & Machine Learning  
💻 Building ML projects from scratch to understand algorithms deeply  
🚀 Focused on mathematical foundations of AI  
🌱 Currently learning: Deep Learning, Numerical Methods, Advanced Linear Algebra

**Tech Stack:**
- Languages: Python, C, SQL
- Libraries: NumPy, Pandas, Matplotlib, Jupyter
- Tools: VS Code, Git, Jupyter Notebook
- Focus: Mathematical implementations, Algorithm design

## Github Profile - Links
- 🌐 GitHub: [https://github.com/shaurya0702-droid](https://github.com/shaurya0702-droid)
- 💼 LinkedIn: [https://www.linkedin.com/in/shaurya-rawat-714349366](https://www.linkedin.com/in/shaurya-rawat-714349366)
- 📧 Email: [Your Email]
- 🌐 Portfolio: [Your Portfolio Link]

---

## Key Takeaway
**"The best way to understand linear algebra is to implement it yourself."**

This library helps you:
- Understand what NumPy does behind the scenes
- Ace math interviews with deep knowledge
- Build intuition for ML algorithms
- Appreciate optimized libraries like NumPy!

---

**⭐ If you found this helpful, please star this repository!**

**Made with ❤️, 🧠, and 📐 (geometry!)**
