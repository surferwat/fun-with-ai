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

This is the python code that I wrote following Andrej's tutorial, "Building makemore Part 4: Becoming a Backprop Ninja". The reference YouTube video: https://www.youtube.com/watch?v=q8SA3rM6ckI

## building_makemore_part5.ipynb

This is the python code that I wrote following Andrej's tutorial, "Building makemore Part 5: Building a WaveNet". The reference YouTube video: https://www.youtube.com/watch?v=t3YJ5hKiMQ0

Here are notes on key concepts covered in the video:

### torch.nn
torch.nn (short for Neural Networks) is a submodule that provides tools for building, training, and managing neural networks. It contains pre-defined layers, loss functions, optimizers, and other utilities that make it easier to implement and experiment with deep learning models.

### torch.nn.Container
Containers refer to the data structures that hold and manage collections of data or models. These are not special, unique objects defined in PyTorch, but rather general terms for classes or modules that help organize, store, and manipulate data efficiently in machine learning workflows.

This tutorial focuses on `torch.nn.Sequential`, which is a container in PyTorch. Modules will be added to it in the order they are passed in the constructor. This container allows you to define a neural network by arranging layers (or modules) one after the other in a straight sequence, where the out output of one layer is passed directly as the input to the next layer (i.e., stack layers in a linear fashion).

The code block is a simplified implementation of PyTorch’s Sequential:
```
class Sequential:
	def __init__(self, layers):
		self.layers = layers
	
	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		self.out = x
		return self.out

	def parameters(self):
		#get parameters of all layers and stretch them out into one list
		return [p for layer in self.layers for p in layer.parameters()]
```

### WaveNet
A WaveNet is a deep neural network architecture designed for generating raw audio waveforms. It was introduced by DeepMind in 2016 and is particularly known for producing highly realistic, human-like speech and other audio signals. Unlike traditional speech synthesis methods that use pre-recorded sound units (like phonemes or audio frames), WaveNet directly generates audio samples, working at the level of individual sound wave points.

In this tutorial, the concept of progressive fusion is introduced. The process beings with two characters being fused into bigram representations. These bigrams are then combined into four-character level chunks, and this process is repeated in a tree-like hierarchical manner. The key idea is to gradually fuse information from previous context as the network depends. At each level, consecutive elements-such as pairs of characters, bigrams, and four-grams - are progressively fused together. This approach helps to model increasingly complex dependencies as the network grows deeper.

### Batchnorm1d bug
The bug occurs when applying batch normalization to an input tensor with the shape (32, 4, 68) in a neural network model. Here's a breakdown of the issue

The input tensor has shape (32, 4, 68): 32 is the batch size; 4 could be a sequence length or a feature dimension; 68 represents the number of channels (features).

Here is the the implementation for when calculating the mean and variance for normalization:
```
emean = e.mean(0, keepdim=True)  # Mean across batch (dimension 0)
evar = e.var(0, keepdim=True)    # Variance across batch (dimension 0)
ehat = (e - emean) / torch.sqrt(evar + 1e-5)
```

This produces `emean` and `evar` with the shape (1, 4, 68), which seems correct because we want to maintain statistics per channel (68 channels).
The bug arises because batch normalization is incorrectly treating the second dimension (4) as independent features instead of treating it as part of the batch. This means that statistics (mean and variance) are being calculated separately for each of the 4 positions, which is not what we want. Instead, we want to calculate the statistics across both the batch dimension (i.e., 32) and the 4th dimension (sequence length or feature dimension) together, ensuring that the normalization is applied correctly across the channels.

The fix is to treat the second dimension (i.e., 4) as part of the batch and apply normalization accordingly. This can involve flattening the dimensions or applying the normalization across the appropriate axes so that the statistics are shared across the 68 channels, not the separate positions in the 4th dimension.

### Experimental harness
Experimental harness refers to a structured framework or environment to run controlled experiments, test hypotheses, and track results effectively.

In the context of machine learning or deep learning, this could involve: data collection and preprocessing pipelines; model tracking; evaluation metrics; and reproducibility.

### Dilated causal convolutions
In this tutorial, convolutions are used as a more efficient way to process sequences, which is inspired by the WaveNet paper. The main idea is that while convolutions make the model more efficient, they don’t fundamentally change the model itself.

Let’s take the name “diondre” as an example. This name has 7 letters, and in the model, each letter is treated as an independent example. So, you have 8 separate inputs for the model, one for each letter in the name, including the starting position and the final period ( . ) added at the end.

Here is the code implementation:
```
for x,y in zip(Xtr[7:15], Ytr[7:15]):
	print(‘’.join(itos[ix.item()] for ix in x), ‘—>’, itos[y.item()])

…….. —> d
…….d —> i
……di —> o
…..dio —> n
….dion —> d
…diond —> r
..diondr —> e
.diondre —> .
```

The model processes the input sequence step-by-step. For example: 
The first row (input `d`) predicts the output `d`
The second row (input `di`) predicts the output `i`
The third row (input `dio`) predicts the output `o`
And so on

In the code: 
`Xtr[7:15]` represents 8 different sequences of the name “diondre.”
`Ytr[7:15]` contains the corresponding targets (the next letter in the sequence for each input)

Now, you can forward a single example (i.e., one row) through the model like this:
``` 
logits = model(Xtr[[7]]) #forward a single example
```

This would output the predictions for just the first row. If you want to do this for all 8 rows, you can look over them:
```
logits = torch.zeros(8,27)
for i in range(8):
	logits[i] = model(Xtr[[7 + i]]) # Forward each row through the model
```

However, this loop is relatively inefficient since each row is processed independently in Python. What convolutions allow you to do is “slide the model” over the input sequence all at once, instead of looping through each row manually. This sliding operation happens inside a CUDA kernel (a specialized piece of code that runs on the GPU), making the process much faster. In simple terms, instead of using a for-loop to apply the model to each input row one-by-one, the convolution does this sliding operation in a highly optimized way, which saves time and computational resources.

In the context of the “diondre” example, dilated causal convolutions can be thought of as a way to apply the model to all 8 rows at once. The “dilated” part of dilated convolutions means that the model skips over some positions, allowing it to capture patterns over a wider range of the sequence without increasing the number of parameters.

## building_gpt.ipynb

This is the python code that I wrote following Andrej's tutorial, "Let's build GPT: from scratch, in code, spelled out.. The reference YouTube video: https://www.youtube.com/watch?v=kCc8FmEb1nY

Here are notes on key concepts covered in the video:

### Tokenization
Process of converting raw text (usually strings of words or characters) into smaller, meaningful units called tokens. These tokens can represent words, subwords, or characters, depending on the type of tokenization used.

### Data
Data is defined as a tensor that contains the encoded text of mini version of Shakespeare’s works.

### Data loading
We do not feed all of the data into the transformer at once since that would be computationally expensive especially when working with large datasets. Instead to manage this, we use data loading techniques that break the data into more manageable chunks of the data which are processed in smaller batches. The maximum length of the chunk is referred to as block size.

In the tutorial, a block size of 8 is used. The target y is created by shifting the data by one token in order to ensure that for every context of length block size, there is a corresponding target token.

```
x = train_data[:block_size]
y = train_data[[1:block_size+1]
for t  in range(block_size):
	context = x[:t+1]
	target = y[t]
```

### Version 1: Averaging past context with for loops
In a batch of up to 8 tokens, the tokens currently don’t interact with each other. We want them to communicate, but only with previous tokens, not future ones. This way, the information flows from earlier context to the current token. A simple way for tokens to "talk" is to compute the average of the previous tokens. This average then becomes a feature vector for the current token. Although this approach is very simplistic and may lose some information, it serves as a reasonable first step. This method is similar to a "bag of words" model, where the focus is on the frequency and context of words without considering their order.

The following code block calculates a cumulative average of the tokens in a sequence up to each time step t. For each token at time step t, it computes the average of all previous tokens (including the current token) and stores it in the xbow tensor. For each token in the sequence, the code is computing a “bag-of-words” (BOW)-like representation, where each token’s vector is replaced by the average of all previous tokens, including itself.

```
xbow = torch.zeros(B,T,C))
for b in range(B):
	for t in range(T):
		xprev = x[b, :t+1] # (t,C)
		xbow[b, t] = torch.mean(xprev, 0)
 ```
 
### Version 2: Using matrix multiply
```
a = torch.ones(3,3)
1, 1, 1 
1, 1, 1
1, 1, 1

a = torch.tril(torch.ones(3,3))
1, 0, 0
1, 1, 0
1, 1, 1

a = torch.tril(torch.ones(3,3)) / torch.sum(a, 1, keepdim=True)
1, 0, 0
0.5, 0.5, 0
0.33, 0.33, 0.33
```

So now if we were to matrix multiply `a @ b` where `b` is a `3 x 2` matrix, we would get a `3 x 2` matrix where the first row represents the average of the first column, the second row, the average of the first and second columns, and the third row, the average of the first, second, and third columns.

Here is the code applying matrix multiply where “a” is represented by “wei” and “b” is represented by “x”.
```
wei = torch.tril(torch.ones(T,T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) ——> (B, T, C)
```

### Version 3: Using softmax
`wei.masked_fill(mask, value)` sets the values of `wei` to `-inf` wherever the mask is True. So, in this case, it fills the upper triangular part of `wei` with `-inf` (because `tril` has zeros in the upper triangle). This is a common technique in attention mechanisms, where we mask out certain positions prevent the model from attending to future tokens. 

Mask refers to a Boolean tensor (a tensor of True/False values) that is used to selectively hide or modify certain positions in a matrix or tensor. The mask tells the program which values should be altered or ignored.

In machine learning, particularly in the context of attention mechanisms (like in Transformers), “attending” refers to the process of focusing on specific parts of an input sequence when processing or generating each output element.

Masking out future tokens prevents the model from using information about tokens that occur later in the sequence when predicting or generating the current token, ensuring that the model generates predictions in an autoregressive or causal manner.

The softmax function is a mathematical function that transforms a vector (or matrix) of real numbers into a probability distribution, where each value in the transformed vector will like between `0` and `1`, and the sum of all the values will equal `1`. In this case, `wei` is a matrix that typically holds attention scores (or weights) before normalization. The softmax function is being applied along the last dimension (`dim=-1`) which is the columns in this case for a matrix. Applying softmax along this axis means that for each row, the function will compute a probability distribution across the elements (columns).

Here is the code applying the Softmax function.
```
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float(‘-inf’))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)
```

### Positional encoding
Andrej states that often not only encoding the identity of these tokens, but also their positions. In neural networks, particularly in models like transformers, self-attention is often used to understand the relationships between words (or tokens) in a sequence. To do this effectively, the model must know both what the tokens are (i.e., their identities) and where they are in the sequence (i.e., their positions).

`self.token_embedding_table` represents the identity of the tokens in the sequence (e.g., words or subwords). For instance, if your vocabulary consists of words like [“hello”, “world”], each word is assigned a unique index. ‘vocab_size’ is the size of the vocabulary (the total number of unique tokens the model can recognize). `n_embd` is the size of the embedding vector that represents each token. The higher the value of `n_embd`, the more complex the representation of each token. So, `self.token_embedding_table` is a lookup table where each token in the vocabulary gets mapped to a vector of size `n_embd` (which could be, for example, a 256-dimensional vector). This allows the model to understand the “meaning” of a token busing these learned vectors.

`self.position_embedding_table` provides a way to encode the position of each token in the sequence (i.e., the order of tokens). Transformers (and similar models) are position-invariant by default, meaning they don’t have any inherent understanding of the order of tokens in a sequence. For example, the sentence “I love dogs” is treated the same as “dogs love I,” unless we explicitly tell the model the positions of tokens. `block_size` represents the maximum length of the input sequence (the number of positions). `n_emd` is again the size of the embedding vector, which is the same as the token embeddings to ensure the position and token embeddings can be combined easily. So, `self.position_embedding_table` is another lookup table, where each position (from `0` to `block_size - 1`) in the sequence gets a unique embedding vector of size `n_embd`. These position embeddings are added to the token embeddings to provide the model with both the identity of each token and its position in the sequence.

Here is the code taking into consideration the position of the token.
```
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
self.position_embedding_table = nn.Embedding(block_size, n_embd)
```

### Version 4: Self-attention
We are going to implement self-attention for a single head. In Version 3, using Softmax applies a simple average of all the past tokens and the current token. However, different tokens will find certain other tokens more or less relevant. For instance, if the token is a vowel, it might focus on consonants from the past to gather information that flows to the current token. The goal is to gather information from the past in a data-dependent manner, and this is exactly what self-attention addresses. In self-attention, each token at each position generates two vectors: a query and a key. The query roughly represents “what am I looking for?” while the key represents “what do I contain?” The interaction between tokens is determined by calculating the dot product between the attention weights: if a query and key are aligned, their interaction is stronger, meaning the current token will learn more from that token in the sequence than from others.

The below code block implements the first steps of calculating the self-attention mechanism for a single attention head.

`x = torch.randn(B,T,C)` creates a tensor `x` of shape `(B, T, C)` with random values drawn from a normal distribution. This represents the input data (e.g., a batch of token embeddings or feature vectors). `head_size = 16` defines the size of each attention head (16 features for the query, key, and value vectors). `key = nn.Linear(C, head_size, bias=False)` defines a linear transformation (a fully connected layer) for computing the key vector. It takes the input of size `C` (32 features) and outputs a vector of size `head_size` (16 features). The `bias=False` means no bias term is used in the transformation. `query = nn.Linear(C, head_size, bias=False)` defines a similar linear transformation for computing the query vector, which also outputs a vector of size `head_size` (16 features) from the input of size `C` (32 features). `value = nn.Linear(C, head_size, bias=False)` defines another linear transformation for computing the value vector which outputs a vector of size `head_size` (16 features) from the input of size `C` (32 features). `k=key(x)` applies the `key` transformation to the input tensor `x` to get the key vectors `k`. This results in a tensor of shape `(B,T,16)` representing the key for each token in the sequence. `q = query(x)` applies the `query` transformation to the input tensor `x` to get the query vectors `q`. 

`wei = q @ k.transpose(-2,-1)` computes the affinity matrix (attention weights) between queries and keys using the dot product. The term affinity matrix is used to describe a matrix that represents how much each item (or token, in the case of self-attention) “cares about” or is related to every other item in a sequence. `q` has shape `(B,T,16)` and `k` has shape `(B,T, 16)`. `k.transpose(-2,-1)` transposes the last two dimensions of `k`, changing its shape to `(B, 16, T)’. The dot product `q @ k.transpose(-2,-1)` computes the attention weights and results in a matrix `wei` of shape `(B, T, T)` where each element represents the attention score between different tokens in the sequence.

```
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)
```

# let’s see a single Head perform self-attention
```
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False) 
k = key(x) # (B, T, 16)
q = query(x) # (B, T, 16)
wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) —-> (B, T, T)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float(‘-inf’))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
#out = wei @ x
```

### Note 1: Attention as a communication mechanism
Attention can be seen as nodes in a directed graph looking at each other and aggregating via with a weighted sum from all nodes that point to them, with data-dependent weights. 

In the context of attention (like in attention mechanisms used in transformers), think of the nodes as elements in a sequence (for example, words in a sentence). Each element (or node) can “attend” to or pay attention to, other elements in the sequence to understand the relationships or dependencies between them. The directed graph part refers to how each element (node) can look at others: a node can have edges (connections) that point to other nodes, showing which other nodes it will pay attention to. The weighted sum means that when one node looks at other nodes, it does so with a certain level of importance, or “weight”. The weight tells how much attention or influence one node has over another. These weights are data-dependent, meaning they are learned from the data itself and are adjusted during training.

### Note 2: Attention has no notion of space, operates over sets
There is no notion of space. Attention is simply acts over a set of vectors. This is why we need to positionally encode tokens. If you want them to have a notion of space, then you need to add it yourself which is what was done when calculating the positional encoding and added that information to the vectors.

What is meant here is that the attention mechanism itself doesn’t inherently know where each element (like a word in a sentence) is located in a sequence. Attention just works with sets of vectors, and sets don’t care about the order or position of elements within them. So, when the attention mechanism looks at these vectors, it’s not aware of whether one word comes before or after another.

### Note 3: No communication across batch dimension
Each example across batch dimension is of course processed completely independently and never “talk” to each other.

“Batch dimension” in machine learning refers to the grouping of multiple examples (like sentences or images) that are processed at once, instead of processing one example at a time. This is done for efficiency during training. No communication across batch dimension means that each example in the batch is processed independently of the others. In other words, when the model is processing one example (e.g., one sentence), it doesn’t have any information about or communicate with other examples in the batch.

### Note 4: encoder blocks vs. decoder blocks
In an “encoder” attention block just delete the single line that does masking with `tril`, allowing all tokens to communicate. What we are implementing here is called  a “decoder” attention block because it has triangular masking, and is usually used in autoregressive settings, like modeling.

The key difference between “encoder blocks” and “decoder blocks” lies in how they handle attention, specifically with respect to masking. In an encoder block (such as in a transformer model), there is no masking. All tokens can communicate with one another freely. In other words, each token can “attend to” or look at every other token in the sequence, including tokens that come after it in the sequence. The purpose of the encoder block is to learn a rich representation of the entire sequence, so it doesn’t need to restrict the flow of information. In an decoder block, masking is used to prevent tokens from seeing future tokens. This is where the triangular masking (with `tril`) comes in. Triangular masking ensures that a token at position `t` can only attend to tokens up to a position `t`, but not to tokens that come after it in the sequence. This is important in autoregressive settings like language modeling or text generation, where each token is generated step-by-step and the model can’t “cheat” by looking ahead at future tokens.

### Note 5: attention vs. self-attention vs. cross-attention
“Self-attention” just means that the keys and values are produced from the same source as queries. In “cross-attention”, the queries get produced from x but the keys and values come from some other external source (e.g., an encoder module).

“Attention”, in general, is a mechanism where the model looks at different parts of the input (like different words in a sentence) and decides how much focus or “weight” to give each part. This is done by computing a query, a key, and a value for each element. The query looks for the relevant keys, and the values provide the actual information. “Self-attention” the keys, values, and queries all come from the same source - so typically, it’s the same set of tokens or elements. For example, when processing a sentence, each word is compared to all other words in the sentence to figure out which ones should be paid attention to (the “attention” part). All of the words are involved in this process, and they are using their own values to help make decisions about which words are most important to the current word. This helps the model understand relationships between words in a sentence, even if the words are far apart. In“cross-attention”, the queries come from one source (say, a sequence `x`) but the keys and values come from a different source (e.g., another sequence, like the output of an encoder in a transformer model). For example, in a sequence-to-sequence model (like machine translation), the query might come from the decoder (which is generating a translation), while the keys and values come from the encoder (which has processed the source sentence). Here, the model is looking at the encoder’s output to help generate the next token in the decoder. This is why cross-attention is useful in tasks like translation, where the model needs to connect parts of the input sequence (like a source sentence) to parts of the output sequence (like the translated sentence).

### Note 6: “scaled” self-attention
The particular attention that is implemented in the reference paper is called “Scaled Dot-Product Attention” and additionally divides `wei` by 1/sqrt(head_size). This makes it so when input  `q` and `k` are unit variance, `wei` will be unit variance too and Softmax will stay diffuse and not saturate too much.

In a typical self-attention mechanism (like in transformers), the attention score is calculated by taking the dot product of the query, `q`, and key, `k` vectors. This dot product measures how much attention one token should give to another token. However, if you just compute the dot product directly, the values of the resulting attention scores can become very large when the dimension of the vectors (the size of `q` and `k`) is large. This can cause problems when applying the Softmax function (which is used to turn those scores into probabilities). Because very large numbers in the Softmax can cause it to saturate - meaning the output will become overly confident (I.e., most tokens would get 0 probability, and just one token might get 1). To prevent this saturation issue, the Scaled Dot-Product Attention scales the dot product by dividing it by a factor of sqrt(head_size) where `head_size` is the dimension of the query and key vectors. This scaling helps keep the values of the attention scores in a reasonable range. Why does this work? When you divide by `sqrt(head_size)` it ensures that the dot product will have unit variance - meaning that it doesn’t get too large or too small, even when the vector dimensions are large.

### Multi-headed attention
In simple terms, multi-head attention is a mechanism used in neural networks (particularly in Transformers) that applies attention multiple times in parallel. Instead of focusing on a single representation of the input (which is what single-head attention does), multi-head attention uses multiple “attention heads” that can each focus on different parts of the input. Afterward, the outputs from these heads are concatenated and processed together. The key idea behind multi-head attention is to allow the model to capture various relationships or interactions in the input data from different perspectives simultaneously, rather than limiting it to just one.

Here is the code block that implements multi-head attention in the tutorial.
```
def __init__(self, num_heads, head_size):
	super().__init__()
	self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

def forward(self, x):
	return torch.cat([h(x) for h in self.heads], dim=-1) 
```

`def __init__(self, num_heads, head_size):` defines the constructor for a class (likely a subclass of `nn.Module`). `num_heads` defines the number of attention heads that you want in the multi-head attention mechanism. Each head will process the input independently. `head_size` defines the size of each attention head, which typically refers to how much dimensionality each head will focus on. `nn.ModuleList(…)` is a PyTorch container that holds modules (in this case, individual `Head` modules). This is important for properly registering the components as PyTorch modules, so they are treated as parameters and can be learned during training. `[Head(head_size) for _ in range(num_heads)]` is a list comprehension that creates `num_heads` instances of the `Head` module. Each head is initialized with `head_size`, which likely refers to the dimensionality of the representation each head will use for attention.

`def forward(self, x):` defines the `forward` method, which tells how the input `x` (typically a tensor) is processed by the multi-head attention mechanism. The `x` could be a sequence of embeddings or any other tensor you pass through the model. In the case of multi-head attention, it could represent input data like a sequence of words or tokens. `[h(x) for h in self.heads]` applies each attention head (`h`) to the input `x`. Each head performs its own independent attention operation on the input. The result will be a list of tensors (each one corresponding to the output of an individual attention head). `torch.cat(…, dim=-1)` concatenates the results of all the attention heads along the last dimension (`dim=-1`), which typically means concatenating them along the feature dimension (i.e., horizontally). 

### Feed Forward
Feedforward refers to the neural network layer that processes token representations in a transformer architecture after the self-attention mechanism has computed relationships between the tokens. It helps refine and transfer these representations by applying transformations and activations before passing them to the next layer. In other words, the feedforward helps the model learn higher-level, non-linear features of the data, thus allowing it to better capture complex patterns in the input sequence.

Here is the implementation of feedforward presented in the tutorial:
```
class FeedForward(nn.Module):
	“”” a simple linear layer followed by a non-linearity “””
	def __init__(self, n_embd):
		super().__init__()
		self.net == nn.Sequential(
			nn.Linear(n_embd, n_embd),
			nn.ReLU(),
		)
	
	def forward(self, x):
		return self.net(x)
```

The first part of this feedforward layer is a linear transformation: `nn.Linear(n_embd, n_embd)`. This is essentially a weighted sum of the inputs. Each token’s representation is transformed through this linear layer, which applies learned weights and biases to the data. After the linear transformation, we apply the ReLU activation function (`nn.ReLU()`). This introduces non-linearity to the model. Without this non-linearity, the model would just be a series of linear transformations, which would limit its ability to model complex, non-linear relationships. In a transformer model, like GPT, the feedforward layer is applied independently to each token’s representation, after the self-attention mechanism has computed relationships between tokens.  The attention mechanism captures the dependencies between tokens (e.g., relationships between words or subwords in a sentence), and the feedforward layer refines and transforms these token representations through a linear transformation and non-linearity.

### Residual connections
One of the key optimizations to address the challenges faced by deep neural networks is the use of skip connections also known as residual connections, as introduced in the paper Deep Residual Learning for Image Recognition. A skip connection, combined with addition, creates a residual pathway that allows for more efficient computation. At least during initialization, this mechanism enables a direct path from the input to the output, ensuring that gradient flow is unimpeded. Over time, the network blocks begin to contribute effectively, facilitating better learning and training.

Here is the code presented in the tutorial before implementing residual connections.
```
class MultiHeadAttention(nn.Module):
	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
	def forward(self, x):
		return torch.cat([h(x) for h in self.heads], dim=-1) 

class FeedFoward(nn.Module):
	def __init__(self, n_embd):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, n_embd),
			nn.ReLU(),
		)
	
	def forward(self, x):
		return self.net(x) 

class Block(nn.Module):
	def __init__(self, n_embd, n_head):
		super().__init__()
		head_size = n_embd // n_head
		self.sa = MultiHeadAttention(n_head, head_size)
		self.ffwd = FeedForward(n_embd)
	
	def forward(self, x):
		x = self.sa(x)
		x = self.ffwd(x)
		return x
```

Here is the code presented in the tutorial after implementing residual connections.
```
class MultiHeadAttention(nn.Module):
	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.proj(out)
		return out

class FeedFoward(nn.Module):
	def __init__(self, n_embd):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, n_embd),
			nn.ReLU(),
			nn.Linear(n_embd, n_embd),
		)
	
	def forward(self, x):
		return self.net(x)

class Block(nn.Module):
	def __init__(self, n_embd, n_head):
		super().__init__()
		head_size = n_embd // n_head
		self.sa = MultiHeadAttention(n_head, head_size)
		self.ffwd = FeedForward(n_embd)
	
	def forward(self, x):
		x = x + self.sa(x)
		x = x + self.ffwd(x)
		return x
```

### LayerNorm
The other the key optimization to address the challenges faced by deep neural networks is the use of layer normalization (i.e., LayerNorm) which is a technique used to improve the training of deep neural networks by normalizing the inputs to each layer. To understand it better, let’s break down the concept step by step.

In machine learning, especially in deep learning, normalization techniques are often used to make training faster and more stable. The main idea is to scale and shift the data to a standard format, which helps the model converge faster and reduces training time. 

There are two main types of normalization: BatchNorm and LayerNorm. BatchNorm normalizes the activations of each layer across the batch dimension (i.e., across the examples in a mini-batch). LayerNorm normalizes the activations of each layer across the feature dimension (I.e., across the units of a particular example).

Here is the code before implementing LayerNorm in the tutorial.
```
def call__(self, x):
	if self.training:
		xmean = x.mean(0, keepdim=True) # batch mean
		xvar = x.var(0, keepdim=True) #batch variance
	else:
		xmean = self.running_mean
		xvar = self.running_var
	xhat = (x - xmean) / torch.sqrt(xvar + self.eps) 
	self.out = self.gamma * that + self.beta
	if self.training:
		with torch.no_grad():
			self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
			self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
	return self.out
```

Here is the code after implementing LayerNorm in the tutorial.
```
def call__(self, x):
	if self.training:
		xmean = x.mean(1, keepdim=True) # batch mean
		xvar = x.var(1, keepdim=True) #batch variance
	else:
		xmean = self.running_mean
		xvar = self.running_var
	xhat = (x - xmean) / torch.sqrt(xvar + self.eps) 
	self.out = self.gamma * that + self.beta
	if self.training:
		with torch.no_grad():
			self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
			self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
	return self.out
```
