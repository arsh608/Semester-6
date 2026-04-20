Implementation of Input Specific Neural Networks (ISNN)
1. Introduction

In this assignment, we implemented two architectures of Input Specific Neural Networks (ISNN-1 and ISNN-2). These networks are designed to enforce structural constraints such as convexity and monotonicity on specific inputs while allowing flexibility on others.

2. Dataset Generation

Two toy datasets were generated based on the function:

f(x,y,t,z)=e
−0.5x
+log(1+e
0.4y
)+tanh(t)+sin(z)−0.4
Training dataset:
500 samples
Range: [0,4]
Testing dataset:
5000 samples
Range: [0,6]

Latin Hypercube Sampling (LHS) was used to ensure uniform coverage of the input space.

3. ISNN Architectures
ISNN-1
Separate branches for each input (x, y, t, z)
Combined in later layers
Enforces:
Convexity in x
Convex + monotonic in y
Monotonic in t
Arbitrary in z
ISNN-2
Uses skip connections and shared interactions
More expressive than ISNN-1
4. Implementation

Two implementations were developed:

4.1 PyTorch Implementation
Automatic differentiation used
Adam optimizer
Mean Squared Error loss
4.2 Manual Implementation (NumPy)
Forward pass using matrix multiplication
Backpropagation manually implemented:
Gradient of loss
Weight updates
5. Training and Evaluation

Both models were trained for 200 epochs.

Observations:
ISNN-2 achieved lower training loss
ISNN-1 showed better generalization in some cases
Both outperformed simple manual NN
6. Results
Loss Graphs
Plotted:
Epoch vs Training Loss
Epoch vs Testing Loss
Behavioral Analysis

Models evaluated for:

x=y=t=z∈[0,6]
ISNN models followed true function more closely than basic NN
7. Conclusion
ISNN models successfully enforce structural constraints
Better extrapolation compared to unconstrained models
Manual backpropagation provided deeper understanding of training dynamics
8. Files Submitted
train_dataset.csv
test_dataset.csv
Colab notebook (.ipynb)
Plots (loss + behavior)
