## Neural Network Basics 🔑

### **How Neural Networks Work**
A neural network is a machine learning program, or model, that makes decisions in way that is similar to the human brain, just like how biological neurons work together to reach conclusions.

Each individual node in a neural network has its own linear regression model, which includes inputs \( x_i \), weights \( w_i \), a bias (or threshold), and produces an output.

The formula can be represented as:

\[ \sum_{i} w_i x_i + \text{bias} = w_1 x_1 + w_2 x_2 + w_3 x_3 + \text{bias} \]

###  $\color{magenta}{\textbf{\textsf{Bias}}}$
In non-deep learning context, bias is referred to as the difference between expected value and true value, aka systematic error. It basically is the measurement of how well the model fits the data. In deep learning context, the bias value allows the activation function to be shifted to the left or right to better fit the data. Unlike the conventional bias, it interacts with the output rather than the input value. Moving forward, whenever bias is mentioned, we will clarify if it is the former or the latter meaning.

###  $\color{magenta}{\textbf{\textsf{Weights}}}$
Once an input layer is determined, weights are assigned. Weights help determine the importance or significance of any given variable, with larger values contributing more significantly to the output.

### $\color{magenta}{\textbf{\textsf{Activation}}}$
All inputs are then multiplied by their respective weights and then summed. Afterward, the output is passed through an activation function, which determines the output. If that output exceeds a given threshold, it “fires” (or activates) the node, passing data to the next layer in the network. This results in the output of one node becoming in the input of the next node. This process of passing data from one layer to the next layer defines this neural network as a feedforward network.

### $\color{magenta}{\textbf{\textsf{Cost function}}}$
A cost function, often referred to as loss function quantifies the difference between predicted and actual values, serving as an indicator of how the model has improved. For a model, the ultimate goal is to minimize our cost function to ensure correctness of fit for any given observation. Cost function takes both predicted outputs by the model and actual outputs and calculates how much wrong the model was in its prediction. It outputs a higher number if our predictions differ a lot from the actual values. As the model adjusts its weights and bias, it uses the cost function and reinforcement learning to reach the point of convergence, or the local minimum. 

Note: Although cost function and lost function are often used interchangeably, they have different meanings. 
Cost function refers to the average of loss functions whereby loss functions refer to the error for a single training example.

### $\color{magenta}{\textbf{\textsf{Back Propagation}}}$
Backpropagation is the practice of fine-tuning the weights of a neural net based on the error rate (i.e. loss) obtained in the previous epoch (i.e. iteration.) Proper tuning of the weights ensures lower error rates, making the model reliable by increasing its generalization.

The process in which the algorithm adjusts its weights is through gradient descent, allowing the model to determine the direction to take to reduce errors (or minimize the cost function). With each training example, the parameters of the model adjust to gradually converge at the minimum.    

Backpropagation allows us to calculate and attribute the error associated with each neuron, allowing us to adjust and fit the parameters of the model(s) appropriately.

### $\color{magenta}{\textbf{\textsf{Forward Propagation}}}$
Most deep neural networks are feedforward, meaning they flow in one direction only, from input to output. 
The steps to forward propagation are as follows:

1. Getting the weighted sum of inputs of a particular unit using the h(x) function we defined earlier.
2. Pluck in the values we get from step one into the activation function, we have (f(a)=a, in this example) and using the activation value we get the output of the activation function as the input feature for the connected nodes in the next layer.

Units X0, X1, X2 and Z0 do not have any units connected to them providing inputs. Therefore, the steps mentioned above do not occur in those nodes. However, for the rest of the nodes/units, this is how it all happens throughout the neural net for the first input sample in the training set:

## $\color{magenta}{\textbf{\textsf{Types of Neural Networks}}}$
Neural networks can be classified into different types, which are used for different purposes. 
Common types include:
1. Perceptron (oldest neural network, well at least according to IBM)
2. Feedforward neural networks, or multi-layer perceptrons (MLPs)
- They are comprised of an input layer, a hidden layer or layers, and an output layer. While these neural networks are also commonly referred to as MLPs, it’s important to note that they are actually comprised of sigmoid neurons, not perceptrons, as most real-world problems are nonlinear. Data usually is fed into these models to train them, and they are the foundation for computer vision, natural language processing, and other neural networks.

3. Convolutional neural networks (CNNs)
- They are similar to feedforward networks, but they’re usually utilized for image recognition, pattern recognition, and/or computer vision. These networks harness principles from linear algebra, particularly matrix multiplication, to identify patterns within an image.

4. Recurrent neural networks (RNNs)
- Often involve feedback loop (we love feedbacks dont we). These learning algorithms are primarily leveraged when using time-series data to make predictions about future outcomes, such as stock market predictions or sales forecasting.

## $\color{magenta}{\textbf{\textsf{Deep Learning vs Neural Networks}}}$ 
Deep Learning and neural networks tend to be used interchangeably in conversation, which can be confusing. The “deep” in deep learning is just referring to the depth of layers in a neural network. A neural network that consists of more than three layers—which would be inclusive of the inputs and the output—can be considered a deep learning algorithm. A neural network that only has two or three layers is just a basic neural network.

## Example Code

1. **Simple Neural Network**
- Not available yet
3. **Convolutional Neural Network (CNN)**


## References
- [Backpropagation in Neural Networks](https://builtin.com/machine-learning/backpropagation-neural-network)
- [Understanding Cost Function in Machine Learning](https://www.analyticsvidhya.com/blog/2021/02/cost-function-is-no-rocket-science/)
- [IBM Neural Networks Overview](https://www.ibm.com/topics/neural-networks)
