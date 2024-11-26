# fun with ai

## building_micrograd.ipynb

This is the python code that I wrote following Andrej's tutorial, "The spelled-out intro to neural networks and backpropagation: building micrograd". The reference YouTube video: https://www.youtube.com/watch?v=VMj-3S1tku0

## building_makemore.ipynb

This is the python code that I wrote following Andrej's tutorial, "The spelled-out intro to language modeling: building makemore". The reference YouTube video: https://www.youtube.com/watch?v=PaCmpygFfXo

## building_makemore_part2.ipynb

This is the python code that I wrote following Andrej's tutorial, "Building makemore Part 2: MLP". The reference YouTube video: https://www.youtube.com/watch?v=TCH_1BHY58I

Here are notes on key concepts covered in the video:

### Embedding
An embedding is a mapping of a discrete - categorical - variable to a vector of continuous numbers. In the context of neural networks, embeddings are low-dimensional, learned continuous vector representations of discrete variables. Neural network embeddings are useful because they can reduce the dimensionality of categorical variables and meaningfully represent categories.

In the tutorial, a vocabulary of 27 characters is used, with bigrams represented by a 27 x 27 matrix. The matrix is then embedded into a 27x2 matrix, effectively capturing the relationships between the characters in a more compact form.

Reference: https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526

### Tensor.view
Tensor.view returns a new tensor with the same data as the “self” tensor but of a different shape. The returned tensor shares the same data and must have the same number of elements but may have a different size.

Function signature: Codeview(dtype) -> Tensor

The tutorial, at timestamp 23:33, demonstrates Tensor.view by transforming a 1x18 tensor into different shapes: a 2x9 tensor (view(2,9)), 9x2 tensor (view(9,2)), and 3x3x2 tensor (view(3,3,2)). These examples illustrate how Tensor.view works. When implementing the hidden layer of the neural network, Tensor.view is used to ensure that the shapes of “emb” variable and “W1” variable  are compatible for multiplication.

Code excerpt from tutorial: emb.view(32,6) @ W1 + b1.

Reference: https://pytorch.org/docs/stable/generated/torch.Tensor.view.html

### Tensor.storage
torch.Storage is an alias for the storage class that corresponds with the default data type. For instance, if the default data type is torch.float, torch.Storage resolves to torch.FloatStorage. A torch.TypedStorage is a contiguous one-dimensional array of elements of a particular torch.dtype.

The tutorial introduces storage() to illustrate why Tensor.view is an efficient operation. The storage() method reveals the underlying tensor as a one dimensional array reflecting how the tensor is stored in memory. When view() is called, it does not create a new tensor or modify the original memory layout; instead, it provides a different view of the existing data. This means view() is efficient because it operates on the same underlying memory without incurring the overhead of copying data.

References: https://pytorch.org/docs/stable/storage.html

### torch.nn.functional.cross_entropy
This function computes the cross entropy loss between input logits and target. Cross entropy originates from information theory. In regards to entropy specifically, it measures the degree of randomness or disorder within a system. In the context of information theory, the entropy of a random variable is the average uncertainty, surprise, or information inherent to the possible outcomes. In the context of machine learning, cross-entropy, also known as logarithmic loss or log loss, is a widely used loss function that evaluates the performance of a classification model.

In the tutorial, this function is used to replace a multi-step custom implementation for calculating loss likelihood. The built-in function is more efficient and numerically stable. It is more efficient for two main reasons: first, it is more memory efficient, as PyTorch optimizes the operations avoiding the creation of multiple new tensors, unlike the custom implementation. Second, the backward pass is more computationally streamlined, as PyTorch simplifies the underlying mathematical expressions.

Before code:
counts = logits.exp()
prob = counts / counts.sum(1, keep(dims=True)
loss = -prob(torch.arange(32), Y).log().mean()

After code:
F.cross_entropy(logits, Y)

Reference1: https://www.datacamp.com/tutorial/the-cross-entropy-loss-function-in-machine-learning
Reference2: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html

### Overfitting one batch
Overfitting occurs when an algorithm fits too closely or even exactly to its training data, resulting in a model that struggles to make accurate predictions on any data outside of that training set. Generalization of a model to new data is crucial for a model’s effectiveness, as it enables machine learning algorithms to make predictions and classify data in real-world scenarios. 

Overfitting a single batch, which refers to a subset of the entire training dataset, happens when a model learns the noise or specific details of just one batch of training data instead of generalizing from the overall dataset. 

The tutorial illustrates a training loop using a single batch consisting of 32 bigrams derived from 5 words (I.e., examples). In contrast, the model has 3481 defined parameters. This disparity in size contributes to the very low loss of 0.25521 observed, as the model has effectively memorized the limited training data instead of learning to generalize.

Reference: https://www.ibm.com/topics/overfitting#:~:text=IBM-,What%20is%20overfitting%3F,of%20the%20machine%20learning%20model.

### Learning rate
In the context of optimization algorithms, the term “learning rate” refers t o the scalar value that controls how much to update the parameters (or weights) of a model during training. The learning rate affects how big each update to the parameters will be. A larger learning rate can lead to faster learning but if it’s too large, it might overshoot the optimal solution.

The purpose of using the gradient in optimization is to minimize a loss function. The gradient of a function points in the direction of the steepest ascent (i.e., where the function increases most rapidly). If you want to minimize the function, you need to move in the opposite direction which is why the negative gradient is used. 

By updating the parameters in the direction of the negative gradient, you effectively reduce the loss value. 

In the tutorial, at 45:54, a learning rate of 0.1 is initially used.

Relevant code block:
For p in parameters:
	p.data += -0.1 * p.grad

### Train/val/test splits
The training set (Train) is the portion of the dataset used to train the model. The model learns patterns and relationships from this data. Typically, this set comprises the majority of the data, often around 70-80\%. Simply put, the importance of the training set is that it allows the model to learn from data.

The dev/validation set is the portion of the dataset used to tune model hyper parameters (e.g., size of the hidden layer, size of embeddings, …) and make decisions about model architecture. It helps assess how well the model generalizes to unseen data during training. Usually, this set accounts for about 10-15\% of the total dataset. After training, the model’s performance on this set can help guide adjustments. Simply put, the importance of the validation set is that it provides feedback for tuning and avoids overfitting.

The test set is the portion of the dataset used to evaluate the final model’s performance after training and validation are complete. It acts as a proxy for how the model will perform on completely unseen data in real-world applications. The test set typically comprises 10-15\% of the total dataset. Simply put, the importance of the testing set is that it provides an unbiased evaluation of the final model’s performance.

In the tutorial, the proportion of the dataset used for training, validation, and testing are 80\%, 10\%, and 10\% respectively. These proportions result in 32033, 3203, and 3204 examples (i.e., bigrams derived from words) respectively,

## building_makemore_part3.ipynb

This is the python code that I wrote following Andrej's tutorial, "Building makemore Part 3: Activations & Gradients, BatchNorm". The reference YouTube video: https://www.youtube.com/watch?v=P6sfmUTpUmc

Here are notes on key concepts covered in the video:

### Saturated tanh
tanh refers to the hyperbolic tangent function. 

In the context of neural networks, tanh serves as an activation function that introduces non-linearity into a neuron’s output. The input to the tanh activation function is typically the weighted sum of inputs to a neuron, often denoted as z. This can be expressed mathematically as: z = w1x1+w2x2+…+wnxn+b where wi are the weights, xi are the input features, and b is the bias term. As Andrej, describes, tanh is a squashing function - it takes arbitrary numbers and squashes them into a range in -1 and 1, doing so smoothly.

Saturated tanh refers to the behavior of the hyperbolic tangent activation function when its input values are far from zero, particularly in the positive or negative direction. The tanh function outputs values between -1 and 1, and when the input is large (either positive or negative), the function approaches its asymptotic values of -1 or 1. In these regions, the gradient (or derivative) of the function becomes very small, leading to what is known as “saturation”. When neurons operate in this saturated region, small changes in input can result in negligible changes in output, making it difficult for the network to learn effectively during backpropogation.

In the tutorial, the preactivations that feed into the tanh function are broadly distributed taking values in a range between -15 and 15. This is why the output values can take on extreme values near -1 and 1. The preactivations are represented by the variable named “hpreact”.

### Initializing scales (Kaiming)
Initializing scales using Kaiming initialization (also known as He initialization) is a method for setting the initial weights of neural network layers. Proposed by Kaiming He and his colleagues, this technique helps mitigate the vanishing and exploding gradient problems often encountered during the training of deep neural networks.

In the tutorial, Andrej runs through an example using the following code block:

x = torch.randn(1000, 10)
w = torch.randn(10, 200) / 10**0.5
y = x @ w

x is the input tensor (representing pre-activations) with a shape of (1000,10). 1000 represents the number of samples (or datapoints), indicating that we are generating a dataset with 1000 individual samples. 10 represents the number of features (or variables) for each sample, meaning each of the 1000 samples has 10 attributes or measurements. x is initialized with values drawn from a standard normal distribution (mean=0, variance=1) giving it a standard deviation of std(x) = 1.

w is the weight matrix that connects the input features to the output features in a layer of a neural network. w is initialized with values from a standard normal distribution. The resulting distribution of y will depend on both the input distribution and the variance of the weights.

Since y = x @ w, the variance of the output y can be influenced by the weights. When performing the multiplication, if the weights have a variance of 1 (which they do in w = torch.randn(10,200)), the variance of the output becomes: var(y) = var(x)*var(w)*n_in where n_in is the number of input features (10 in this case).

By dividing the weights w by 10**0.5, you effectively scale the variance of the weights down. This adjustment helps keep the overall variance of the output y more consistent with that of the input x: var(w) = 1/10 => var(y) ≈ var(x). This scaling ensures that the output distribution remains Gaussian and does not become excessively spread out.

### Batch normalization
Batch normalization is a technique used in training deep neural networks, including multilayer perceptrons, to improve their performance and stability. During training, the input to each layer can vary significantly, which can lead to issues such as slow convergence or even instability. Batch normalization normalizes the output of a layer (activations) so that they have a mean of zero and a standard deviation of one. 

For each mini-batch of data, batch normalization computes the mean and variance of the activations. The activations are then normalized using these statistics. After normalization, the layer applies a linear transformation, scaling and shifting the normalized values using learnable parameters (gamma and beta). This allows the network to retain the capacity to model complex functions.

In the tutorial, the following code block is the implementation of batch normalization.

hpreact = embcat @ W1 + b1 #hidden layer pre-activation
hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True))/ hpreact.std(0, keepdim=True) + bnbias

“hpreact = embcat @ W1 + b1” computes the pre-activation of the hidden layer by multiplying the input embeddings (embcat) with the weight matrix (W1) and adding the bias (b1)

“hpreact.mean(0, keepdim=True)” calculates the mean of the pre-activation values along the first dimension (batch dimension), keeping the dimensions for broadcasting. In this context, the first dimension corresponds to the number of samples. For a tensor shape of (N,D) where N is the number of samples and D is the number of features. Mean across first dimension computes the mean for each feature across all samples resulting in a tensor of shape (1,D). In contrast, mean across the 2nd dimension computes the mean across each sample for all features, resulting in a tensor shape of (N,1).

“hpreact.std(0, keepdim=True)” calculate the standard deviation of the pre-activation values along the first dimension (batch dimension), also keeping the dimensions for broadcasting.

“bngain = torch.ones((1, n_hidden))” initializes the scale parameter (gamma) for batch normalization with ones, shape (1, n_hidden). By initializing gamma to ones, you effectively start with an identity transformation for the normalized activations. This means at the beginning of training the normalized activations will be unchanged.

“bnbias = torch.zeros((1, n_hidden))” initializes the shift parameter (beta) for batch normalization with zeros, shape (1, n_hidden). By initializing gamma to zeros, you effectively start with an identity transformation for the normalized activations. This means at the beginning of training the normalized activations will be unchanged.

### Forward pass activation statistics
Forward pass activation statistics involve creating histograms of the outputs from the forward pass activations, specifically those from the tanh function. These histograms provides a visual representation to identify potentially problematic patterns, such as skewness (whether the distribution leans towards low or high values), modality (number of peaks), or saturation (if most values are pushed to extremes).

In the tutorial, here is the code block that creates the histograms:
plt.figure(figsize=(20,4)) #width and height of the plot
legends= [] 
for i, layer in enumerate(layers[:-1]): #note: exclude the output layer
	if isinstance(layer, Tanh): 
		t = layer.out 
		print(‘layer %d (%10s): mean +.2f, std %.2f, saturated: %.2f%%’ % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100)
		hy, hx = torch.histogram(t, density=True) 
		plt.plot(hx[:-1].detach(), hy.detach()) 
		legends.append(f'layer {i} ({layer.__class__.__name__}') 
plt.legend(legends); 
plt.title('activation distribution')

### Backward pass gradient statistics
Backward pass gradient statistics involve creating histograms of the gradients computed during the backward pass of a neural network. These histograms help identify how gradients distributed - whether they are predominantly small, large, or concentrated around particular values.

In the tutorial, histograms are created for each layer (except the output layer) allowing for a comparison across layers. The comparison provide insights into how effectively different parts of the network are learning and whether any adjustments are necessary. The histograms created were similar, indicating that the gradients were consistent across layers which is the desired outcome.

In the tutorial, here is the code block that creates the histograms:
plt.figure(figsize=(20,4)) #width and height of the plot
legends= [] 
for i, layer in enumerate(layers[:-1]): #note: exclude the output layer
	if isinstance(layer, Tanh): 
		t = layer.out.grad 
		print(‘layer %d (%10s): mean +.2f, std %.2f, saturated: %.2f%%’ % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100)
		hy, hx = torch.histogram(t, density=True) 
		plt.plot(hx[:-1].detach(), hy.detach()) 
		legends.append(f'layer {i} ({layer.__class__.__name__}') 
plt.legend(legends); 
plt.title(‘gradient distribution)

## building_makemore_part4.ipynb

This is the python code that I wrote following Andrej's tutorial, "Building makemore Part 4: Becoming a Backprop Ninja". The reference YouTube video: https://www.youtube.com/watch?v=q8SA3rM6ckI&t=5302s

