# fun with ai

## TOC
- [building_micrograd.ipynb](#building_microgradipynb)
- [building_makemore.ipynb](#building_makemoreipynb)
- [building_makemore_part2.ipynb](#building_makemore_part2ipynb)
- [building_makemore_part3.ipynb](#building_makemore_part3ipynb)
- [building_makemore_part4.ipynb](#building_makemore_part4ipynb)
- [building_makemore_part5.ipynb](#building_makemore_part5ipynb)
- [building_gpt.ipynb](#building_gptipynb)
- [building_gpt_tokenizer.ipynb](#building_gpt_tokenizeripynb)
- [building_nanogpt.py](#building_nanogptpy)

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

Let’s see a single Head perform self-attention.
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

## building_gpt_tokenizer.ipynb

### Unicode code points
A unicode code point is a unique number assigned to each character in the Unicode Standard, which is a system designed to represent text from all the world’s writing systems in a consistent way. Version 16.0 of the standard defines 154998 characters and 168 scripts used in various ordinary, literary, academic, and technical contexts.

### Unicode byte encodings
Unicode byte encodings refer to how Unicode characters (which are abstract code points) are represented as bytes for storage or transmission. The most common Unicode encodings are: UTF-8, UTF-16, UTF-32. 

UTF-8
1-4 bytes per character
Variable length
ASCII compatible
Commonly used for web, files

UTF-16
2 or 4 bytes
Variable length
ASCII incompatible
Commonly used for windows, java

UTF-32
4 bytes
Fixed length
ASCII incompatible
Commonly used for internal use, simplicity

### Byte Pair Encoding (BPE)
BPE is an algorithm, first described in 1994 by Philip Gage, for encoding strings of text into smaller strings by creating and using a translation table. The original version of the algorithm focused on compression. It replaces the highest-frequency pair of bytes with a new byte that was not contained in the initial dataset. A lookup table of the replacement is required to rebuild the initial dataset.

Example: aaabdaaabac. Let Z=aa. aaabdaaabac => ZabdZabac. Let Y = ab.  ZabdZabac => ZYdZYac. Let X = ZY. ZYdZYac => XdXac. https://en.wikipedia.org/wiki/Byte_pair_encoding

tiktoken
Tiktoken is a fast BPE tokenizer for use with OpenAI’s models.

### Special Tokens
In the context of tokenization for machine learning models, particularly in natural language processing (NLP), the idea of special tokens extends beyond the usual linguistic tokens that come from raw text and subword merging techniques like BPE. Special tokens are a unique set of tokens that have specific roles within the token stream and are used to manage the structure of the data, as well as facilitate certain functionalities for models like GPT-3, BERT, and other transformer-based architectures. Special tokens are inserted into the token stream to provide context and structure that the model can interpret in a way that is meaningful for a particular task. They help the model differentiate between different sections of input, mark certain actions, or indicate the start or end of specific sequences. Types of special tokens include: classification token (CLS), separator token (SEP), padding token (PAD), unknown token (UNK), mask token (MASK), beginning of sentence (BOS), end of sentence (EOS), among other custom tokens.

### SentencePiece
SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training. It is particularly well-suited for languages that don’t use spaces to separate words (like Japanese) and for neural network models that require subword-level tokenization (like BERT, T5, or GPT variants).

### Prompt Compression
Prompt compression refers to techniques used to reduce the size or length of a prompt (input text) sent to a language model - without significantly degrading the quality or accuracy of the model’s output. This is important because: language models have input length limits (measured in tokens); smaller prompts are faster and cheaper to process; compressing prompts can help maintain context over longer interactions or documents. In the tutorial, Andrej introduces Gist Tokens in the context of prompt compression. In systems like OpenAI’s memory features or retrieval-augmented generation (RAG), gist tokens are generated as short textual or semantic representations of longer documents or conversations.

### Multimodal tokenization
The process of converting inputs from different modalities - like text, images, audio, or video - into a unified format (usually tokens) that can be processed by a multimodal AI model. Why it’s important? Most large AI models operate on sequences of tokens. For multimodal models like GPT -4o or CLIP, each input type (e.g., a word or a pixel patch) is converted into tokens that the model can process in the same computational framework.

### Vector quantization
A technique used to compress data by mapping continuous input vectors (like image or audio features) to a limited set of representative vectors called codebook entries. It’s widely used in areas like signal compression, image processing, and more recently, in machine learning models like Vector Quantized Variational Autoencoder (VQ-VAE) and large multimodal models.


## building_nanogpt.py

This is the python code that I wrote following Andrej's tutorial, "Let's reproduce GPT-2 (124M)". The reference YouTube video: https://www.youtube.com/watch?v=l8pRSuU81PU

Here are notes on key concepts covered in the video:

### Implementing the GPT-2 nn.Module
GPT-2 is a decoder-only Transformer, meaning it does not include an encoder or cross-attention layers, which are typically used in encoder-decoder architectures. There are two key architectural differences from the original Transformer design (as presented in the GPT paper):
1. LayerNorm Placement: In GPT-2, Layer Normalization is applied before the attention and MLP blocks (this is known as Pre-LN), unlike the original Post-LN setup where normalization is done afterward.
2. Additional Final LayerNorm: GPT-2 includes an extra LayerNorm at the end of the model (after all Transformer blocks), which is not present in the original Transformer architecture.

### Class GPT
This class defines configuration settings and module structures used throughout the GPT-2 architecture. Below are the key components it utilizes: 

nn.ModuleDict()
nn.ModuleDict is a PyTorch container that stores submodules in a dictionary-like structure, allowing access via string keys. It’s useful for organizing multiple submodules (like layers or blocks) in a way that is both easy to reference and compatible with `nn.Module` parameter tracking (so everything is properly registered and moves to the correct device). In the GPT-2 implementation, it can be used to store components like attention or MLP blocks, especially if you want to access them by name.

nn.Embedding()
`nn.Embedding` is essentially a learnable lookup table. It’s a wrapper around a tensor where each row corresponds to the embedding vector for a specific token ID. When you index into it with a batch of token IDs, it returns their corresponding embedding vectors. In GPT-2, `nn.Embedding` is used to: Convert token IDs into input embeddings; Optionally include positional embeddings to encode the position of each token in the sequence. This allows the model to start with dense vector representations of tokens instead of raw IDs.

nn.ModuleList()
`nn.ModuleList` is a Python list specifically for storing submodules in PyTorch. It behaves like a regular list, but it ensures that all the layers you add are registered as part of the model, so PyTorch can track their parameters, move them to the correct device, and include them in back propagation. In GPT-2, it’s commonly used to hold the sequence of Transformer blocks, so you can loop over them or access specific layers by index during the forward pass. 

nn.LayerNorm()
`nn.LayerNorm` applies layer normalization, which normalizes the inputs across the feature dimension - stabilizing training and helping gradients flow better. As used in GPT-2: It normalizes each token’s embedding independently across its features. It’s applied in a pre-norm setup, meaning it comes before the attention and MLP blocks (unlike the original Transformer, which used post-norm). This ensures consistent input scale for the attention and MLP submodules. “It’s just standard LayerNorm - nothing fancy.”

nn.Linear()
`nn.Linear` is a fully connected (dense) layer that applies a linear transformation to the its input: output = input @ weight.T + bias. In the GPT-2 model: It’s used in both the attention mechanism (to project queries, keys, values, and outputs). And in the MLP block (for the two-layer feedforward network with a GELU nonlinearity in between). These linear layers give GPT-2 its ability to learn rich, high-dimensional transformations. “It’s just a matrix multiply + bias - nothing too magical.”

### Class Block
The Block class defines a core component of the Transformer architecture. In the `forward(self, x)` method, the input tensor `x` is processed in the following sequence:
1. Layer Normalization
2. Self-Attention
3. Layer Normalization (again)
4. Feedforward Neural Network (MLP)
5. Residual Connections

While this might look different from the original GPT paper - where layer normalization is applied after attention or feedforward layers - this variation reflects a more modern design. Specifically, by applying normalization before attention and MLP layers (a pattern known as “Pre-LN”), we can maintain a clean residual stream throughout the network.

This design has practical benefits. Because the residual pathway remains unmodified by normalization or other operations, gradients from the output layer can flow directly back to the input tokens. If you recall from micrograd, addition in the residual path distributes gradients equally to both branches during back propagation. This means that residual connections act as clean gradient highways, allowing stable training and deeper networks. 

In summary, this block structure ensures that both the signal and its gradients can flow efficiently through the model - from input tokens to output predictions and back - enabling powerful learning dynamics.

### Class MLP
The MLP class (Multi-Layer Perceptron) implements a simple feedforward network, commonly used in transformer architectures. It’s relatively straightforward. In the __init__(self, config) method, the layers are constructed as follows:
1. `self.c_fc`: a linear projection layer (nn.Linear) that expands the dimensionality of the input.
2. `self.gelu`: a non-linearity, specifically the GELU (Gaussian Error Linear Unit) which resembles a smoother version of ReLU.
3. `self.c_proj`: Another linear layer that projects the expanded hidden dimension back down to the original size.

The GELU activation behaves similarly to ReLU but doesnt have a hard zero threshold; instead, it transitions smoothly. This function was introduced in the Gaussian Error Linear Units paper. Historically, an approximate version of GELU was used (especially in early models like BERT and GPT) due to performance issues evaluating the exact erf function in TensorFlow at the time.

Daniel Henrycks (the author of GELU) discussed this on GitHub, explaining that the approximation was a practical necessity back then. Today, thanks to improved libraries and hardware, the exact GELU can be used without performance concerns, and the difference in results between the approximate and exact versions is negligible.

### Class CausalSelfAttention
This class implements multi-head self-attention, which is more sophisticated than standard attention. In earlier tutorials, we saw how multi-head attention works by running several attention “heads” in parallel and then concatenating their outputs. This same mechanism is implemented here, but instead of using multiple distinct attention modules, it’s all done within a single efficient module using tensor operations - what Andrej calls “tensor gymnastics.” 

In this implementation, we work with a sequence of tokens - up to 1024 of them. Each token at this point is transformed into three vectors: query (`Q`), key (`K`), and value (`V`). The attention mechanism relies on computing similarity between queries and keys to determine how much focus each token should give to others in the sequence. To compute this efficiently, we first compute a combined QKV tensor in one projection using `c_attn`, and then split it into `Q`, `K`, and `V`. To parallelize across multiple attention heads, we reshape and transpose the tensors such that the head dimension `nh` becomes part of the batch dimension. This enables efficient computation over all heads in parallel using PyTorch’s broadcasting and batch operations. The core steps are: 
1. Compute dot products between queries and keys to get attention scores. 
2. Apply a causal mask to enforce autoregressive behavior (each token can only attend to past or current tokens, not future ones). 
3. Normalize the scores using softmax. 
4. Use the attention weights to compute a weighted sum over the value vectors. 
5. Recombine the multiple heads back together using `transpose` and contiguous().view() operations.

All of this is functionally equivalent to the multi-headed attention implementation we’ve seen before, but it’s written in a more efficient and compact PyTorch style. 

A note on naming: variable names like `c_attn` match those used in Hugging Face’s transformer library. This naming consistency allows us to load pertained weights from Hugging Face directly, ensuring compatibility.

At this stage, the GPT-2 implementation is complete. The entire model is under ~100 lines of PyTorch code, compared to ~2000 lines in Hugging Face’s full version. We can now load the weights and proceed to text generation using our own simplified model. 

### Class GPTConfig
The GPTConfig class defines the architectural hyperparamters of the GPT-2 model and ensures that the model being implemented matches the specifications of the GPT-2 small variant (124M parameters). These configuration settings are essential for constructing the transformer layers and defining the overall structure of the model.
`block_size: int = 1024` This sets the maximum context length (i.e., the number of tokens) that the model can attend to. Each input sequence can be up to 1024 tokens long. This value controls how much “memory” the model has of previous tokens in a sequence.
`vocab_size: int = 50257` This defines the number of unique tokens the model can recognize and output. GPT-2 uses byte pair encoding (BPE) and the vocabulary consist of 50,000 BPE merges plus additional byte-level tokens (0-255) and one special token, <|endoftext|> used to denote document boundaries or indicate where text generation should begin.
`n_layer: int = 12` The number of transformer blocks (also called layers). Each block contains components such as multi-head self-attention and feed-forward neural networks. Increasing the number of layers increases the depth and representation capacity of the model.
`n_head: int = 12` The number of attention heads used in the multi-head self-attention mechanism. Multiple heads allow the model to attend to different parts of the input sequence in parallel, capturing richer contextual relationships.
`n_embd: int = 768` The size of the token embeddings and also the dimensionality of all hidden layers in the transformer. It defines the width of the model. All layers (input embeddings, attention mechanisms, feed-forward networks) operate in this 768-dimensional space.

Load params from Hugging Face and initialize GPT class with those parameters
```
@classmethod
def from_pretrained(cos, model_type):
```
This method supports four model variants:
- `gpt2` (124M parameters)
- `gpt2-medium` (350M)
- `gpt2-large` (774M)
- `gpt2-xl` (1558M)
Step 1: Load Model Hyperparamters
First, the corresponding hyper parameters for each GPT-2 variant are set in a dictionary:
```
config_args = {
	‘gpt2’: dict(n_layer=12, n_head=12, n_embd=768), #124M
	‘gpt2-medoum’: dict(n_layer=24, n_head=16, n_embd=1024), #350M
	‘gpt2-large: dict(n_layer=36, n_head=20, n_embd=1280), #774M
	‘gpt2-xl’: dict(n_layer=48, n_head=25, n_emdb=1600), #1558M
}[model_type]
config_args[‘vocab_size’] = 50257 # always 50257 for GPT model checkpoints
config_args[‘block_size’] = 1024 # always 1024 for GPT model checkpoints
```
Step 2: Create a minGPT Model with the same config
```
config = GPTConfig(**config_args)
model = GPT(config)
```
Now `model` is a randomly initialized minGPT model matching the chosen GPT-2 variant.

Step 3: Prepare the State Dicts
```
sd = model.state_dict()
sd_keys = sd.keys()
sd_keys = [ k for k in sd_keys if not k.endswith(‘.attn.bias’)] # Remove buffer mask

This filters out non-weight tensors like ‘.attn.bias’, which aren’t present in Hugging Face’s weights.

Step 4: Handle Transposed Weights
A key annoyance is that some Hugging Face weights are stored with their dimensions transposed. These must be manually identified and transposed before loading.

```
transposed = [
    'attn.c_attn.weight',
    ‘attn.c_proj.weight',
    …
]
assert len(sd_keys_hf) == length(sd_keys) …
for k in sd_keys_hf: …
```
Step 5: Load the weights
```
model = GPT.from_pretrained(‘gpt2’)
print(“didn’t crash yay!”)
```

### Add `forward()` to GPT class
Before we can generate text, we need to define how the model performs a forward pass. This is implemented in the `forward()` method of the GPT class. 
Input: The method takes idx, a tensor of token indices with shape `(B,T)` where: `B` is the batch size (number of sequences); `T` is the sequence length (a number of tokens per sequence). 
Step 1 Validation: We assert that `T` does not exceed the model’s block_size. This ensures the input fits within the maximum context window the model was trained on . 
Step 2 Embedding Layers: The input token indices are passed through a token embedding layer to obtain token representations. We also compute positional embeddings of shape `(1,T,C)` to encode token positions. These embeddings are added together to form the input to the transformer blocks.
Step 3 Transformer Blocks: The input is passed sequentially through a stack of TransformerBlocks. Each block applies self-attention and feedforward layers with residual connections and layer normalization.
Step 4 Output Layers: After the blocks, a final LayerNorm is applied. The result is passed through a linear projection (the language modeling head) to obtain logits over the vocabulary for each position. This method defines the full forward computation of the model, enabling training and inference. 

Match Hugging Face code (right side)
Hugging Face code
```
from transformers import pipeline, set_seed
generator = pipeline(“text-generation”, model=‘gpt2’)
set_seed(42)
generator(“Hello, I’m a language model,” max_length=30, num_return_sequences=5)
```
Step 1 Initialize model (equivalent to pipeline(“text-generation”,model=‘gpt2’)

```
max_return_sequences = 5
max_length = 30
model = GPT.from_pretrained(‘gpt2’) # Load retrained model weights 
model.eval() # probably will have no effect here, Set model to evaluation mode (may not affect inference here)
model.to('cuda') # Move model to GPU for faster computation
```

Step 2 Create Input Tokens (equivalent to prompt: “Hello, I’m a language model,”)

```
import tiktoken
cnc = tiktoken.get_encoding(‘gpt2’) # Load GPT-2’s tokenizer
tokens = enc.encode(“Hello, I’m a language model.”) # Tokenize input text
tokens = torch.tensor(tokens, dtype=torch.long) # Convert to tensor of shape (seq_len,) in this case, (8,).
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # Duplicate prompt for each sequence: (5, seq_len) i.e., (5,8), 
x = tokens.to('cude') # Move tokens to GPU; these will be passed into the model
```

Generate
This loop grows the input sequence `x` token by token using top-k sampling (with `k=50`). In every iteration of the loop we will be adding a column of new indices into each one of these rows. With each loop iteration we get one more column.

```
# generate right now x is (B, T) where B = S, T = B
# set the seed to 42
Torch.manual_seed(42) #ensures reproducibility by setting the random seed 
while x.size(1) < max_length:
	# forward the model to get the logits
	with torch.no_grad():
		logits = model(x)		
		#the model outputs logits of shape (B,T, vocab_size). Since we only want to sample the next token, we focus on the logits at the last time step.
		logits = logits[:, -1, :]
		# Apply softmax to get a probability distribution over the vocabulary
		probs = F.softmax(logits, dim=-1)
		# Limit choices to the top 50 most probable tokens. Do top-k sampling of 50 (higgingface pipeline default). topk_probs here becomes (5, 50), topk_indices is (5, 50)
		topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
		# Randomly sample from the top-k probabilities
		ix = torch.multinomial(topk_probs, 1) # (B,1)
		col = torch.gather(topk_indices, -1, ix) # (B,1)
		# Concatenate the sampled token to the input sequence
		x = torch.cat((x,xcol),dim=1)
```

Auto detect device availability
Ensure model utilizes the fasted supported hardware available on your system. Detect the best available device for computation. Use cuda if available (for NVIDIA GPUs), otherwise check for MPS, which is Apple’s Metal Performance Shaders backend optimized for Apple Silicon GPUs. If neither is available, default to CPU.

```
device = “cpu”
If torch.cuda.is_available():
	device = “cuda”
Else if hasattr(torch.backends, “mps”) and torch.backends.mps.is_available():
	device = “mps”
print(f”using device: {device}”)
```

### Tiny Shakespeare Dataset
To train the model, we need a a dataset. Andrej recommends using the Tiny Shakespeare dataset, which is a popular, simple dataset for language modeling. It can be downloaded from this URL: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`. The dataset contains: ~40,000 lines, ~200,000 words, ~1M bytes or characters (since it’s ASCII-only, each byte corresponds to one character).

### Tokenizing the Dataset
To prepare the data for training, we need to tokenize it using the same encoding that GPT-2 uses. This is done with the `tiktoken` library, which provides access to the GPT-2 tokenizer. This step converts the raw text into a sequence of integer tokens, which are the input format required for training the model.

Here’s how to encode the dataset:
```
import  tiktoken
enc = tiktoken.get_encoding(‘gpt2’)
tokens = enc.encode(data)
print(tokens[:24])
```

### Creating a B x T Tensor to be fed into forward
Andrej prefers working with long sequences and processes them in 2D as batches of time steps. To train the model, we construct input (`x`) and target (`y) tensors from a 1D stream of tokens. We start by loading a buffer of `BxT+1` tokens. The extra token (+1) ensures we have a valid target (ground truth) for every input token. From this buffer, we construct: `x`: the input to the model, `y`: the target labels (same shape as `x`, but shifted by one position). For example, suppose `tokens` is a 1D sequence of 25 elements. If we reshape it into 4 sequences of 6 time steps (i.e., `B=4`, `T=6`), then the following code block gives us `x` and `y` each of shape `(4,6)` where `y[i,j]` is the expected next token for `x[I,j]`.

```
import torch
buf = torch.tensor(tokens[:24+1]) # +1 for the ground truth of the last token
x = buf[:-1].view(4,6) # everything up to but not including the last token
y = buf[1:].view(4,6) # skip the first element
print(x)
```

### Getting logits and loss
At this point, we can modify the model’s forward pass to return not only the logits but also the loss, making training easier.

First, instantiate the model and move it to the appropriate device:
```
model = GPT(GPTConfig())
model.to(device)
logits, loss = model(x, y)
```

Next, update the `forward` method of the model to optionally compute and return the loss if target labels `y` are provided.
```
def forward(self, idx, **targets=None**):
…
**loss = None
If targets is not None:
	loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))**
…
return logits, **loss**

This approach allows the same forward function to be used during both training (when targets are available and loss is needed) and inference (when only logits are needed).

### Applying optimization
To train the model, we use an optimizer - specifically, the AdamW optimizer provided by PyTorch. While SGD (Stochastic Gradient Descent) is a common baseline, Adam is often preferred because it adapts the learning rate for each parameter, making training more efficient and stable. AdamW is a corrected version of Adam that properly decouples weight decay from the gradient update, which Andrej mentions is effectively a “bug fix”.

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
	optimizer.zero_grad() # clear previous gradients (i.e., zero gradients)
	logits, loss = model(x, y) # forward pass: compute predictions and loss
	loss.backward() # backward pass: compute gradients
	optimizer.step() # update model parameters
	print(f”step {i}, loss: {loss.item()}”) # Convert scalar tensor to Python float for printing

The `loss.item()` call extracts the scalar value from the loss tensor (which may reside on the GPU depending on what’s configured to be used for training) and brings it to the CPU as a standard Python float. This is helpful for logging or visualization during training.

### Dataloader Lite
After overfitting a single batch, the next step is to optimize across the entire dataset. For this, we need a data loader that continuously serves fresh `(x,y)` training batches instead of repeating the same one. The `DataLoaderLite` class handles this. It uses `tiktoken` to tokenize the full contents of a text file into a flat list of token IDs, stores them in memory as a PyTorch tensor, and prints the total token count along with he number of batches per epoch (defined as how many distinct batches of size `B x T` can be extracted before wrapping around). The data loader starts at position 0 and steps through the token tensor in chunks of `B x T`, advancing the position by exactly `BxT` each time. However, when fetching a chunk, it takes `B x T + 1` tokens to ensure we can construct both the input `x` and the target `y` by shifting the sequence one token forward. If fetching the next batch would go out of bounds, it resets the position to 0 and wraps around.

### Parameter sharing wte and lm_head
In this section, Andrej points out a subtle but important detail about GPT-2’s training: parameter sharing (or weight tying) between the input and output embeddings. When loading retrained weights from Hugging Face, it’s easy to overlook that `transformer.wte.weight` and `lm_head.weight` are not just similar - they are exactly the same tensor.

```
# These tensors have the same shape:
print(sd_hf[“lm_head.weight”].shape) # [50257, 768] - the output projection (logits layer)
print(sd_hf[“transformer.wte.weight”].shape) # [50257, 768] - the token embedding layer
```

But more than just having the same shape, they are identical at the memory level:

```
# All values are equal
(sd_hf[“lm_head.weight”] == sd_hf[“transformer.wte.weight”]).all()

# Data pointers match - they share memory
print(sd_hf[“lm_head.weight”].data_ptr())
print(sd_hf[“transformer.wte.weight”].data_ptr())
```

This is a common and intentional design choice known as weight tying. It was popularized in the Attention Is All You Need paper and earlier work like Using the Output Embedding to Improve Language Models (2017). As the Transformer paper states: “ We share the same weight matrix between the two embedding layers and the pre-softmax linear transformation…”

The intuition, as Andrej explains, is that similar tokens should be treated similarly both at the input and output ends of the model. If two tokens are semantically or morphologically close - for example, lowercase and uppercase versions of the same word - we’d expect them to have nearby vectors in both the input embedding space and the output logits space. That symmetry motivates sharing the weights across the embedding and output projection layers.

To fix the code and properly implement this, we explicitly tie the weights. This ensures that only one tensor is used in both places, reducing redundancy and saving parameters.

```
# Weight sharing scheme
self.transformer.wte.weight = self.lm_head.weight
```

It matters because it improves generalization and parameter efficiency. Shared weights help the model maintain consistency between how it reads and generates tokens. Tying the weights removes the need for a separate output projection matrix. Since the embedding matrix is of shape `[50257, 768]`, this saves about 40 million parameters - roughly 30\% of the model’s total parameter count.

### Model Initialization: std 0.02, residual unit
The GPT-2 and GPT-3 papers don’t go into much detail about weight initialization, but the official GPT-2 implementation provides some insight. By examining `model.py`, we can infer how the weights are initialized.

Specifically:
Linear layer weights are initialized from a normal distribution with mean 0 and standard deviation 0.02.
Biases are initialized to zero, overriding PyTorch’s default (which is typically uniform).
Token embeddings (`wte`) and positional embeddings (`wpe`) are both initialized using `std=0.02` (even though GPT-2’s code uses `std=0.01` for positional embeddings, we follow 0.02 for simplicity and consistency). 

This kind of initialization helps maintain stable gradients early in training, especially in deep models with residual connections like GPT-2.

To apply this scheme across the whole model, the following line is added at the end of the `GPT` class’ `__init__` method:

```
self.apply(self._init_weights)
```

This calls `nn.Module.apply` which recursively visits every submodule (like `nn.Linear`, `nn.Embedding`) and applies a custom initialization function.

Here’s how the `_init_weights` method should look. There is a check for the existence of `bias` before trying to initialized it. Not all linear layers have bias terms (e.g., sometimes for projection layers).

```
def _init_weights(self, module):
	if isinstance(module, nn.Linear):
		torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
		if module.bias is not None:
			torch.nn.init.zeros_(module.bias)
	elif isinstance(module, nn.Embedding):
		torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

Typically, when initializing weights in neural networks, a common heuristic is to use a standard deviation proportional to: 1/sqrt(fan_in) where fan_in is the number of input features to a layer. This approach, often called Xavier or He initialization (depending on specifics), helps keep the variance of activations stable across layers. For GPT-2’s model dimension d_model of 768, this formula suggests: 1/sqrt(768) ≈ 0.036. For larger GPT variants with d_model around 1600: 1/sqrt(1600) ≈ 0.025. The GPT-2 implementation uses a fixed std = 0.02 which falls comfortably within this ballpark. While it’s a fixed hyper parameter rather than dynamically computed from the model size, it’s not an unreasonable choice:
it’s slightly smaller than 1/sqrt(768) which can help training stability
It’s roughly consistent with larger hidden sizes (e.g., 1600)
fixing the std keeps initialization simple and reproducible

In the GPT-2 paper, there’s a line that says: “A modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of 1/sqrt(N) where N is the number of residual layers.”

Andrej elaborates on why this scaling is necessary. In a Transformer, the residual stream (i.e., the output passed through the layers) is updated in a pattern like this: `x = x + something`.

Each block adds a bit to the stream, so over many layers, the contributions accumulate. If we’re not careful, the variance of the residual activations can grow significantly as more layers are added.

```
x = torch.zeros(768)
n = 100 
for I in range(n):
	x += torch.randn(768)

print(x.std()) # Outputs ~10
``` 

Here, because we add `n` random vectors (each with std ≈ 1), the resulting standard deviation `x` grows like sqrt(n) - about 10 when `n=100`. To counteract this accumulation, GPT-2 scales each residual contribution down by `1/sqrt(n)` so that the total variance stays constant:

```
x = torch.zeros(768)
n = 100 
for i in range(n):
	x += n**-0.5 * torch.randn(768)

print(x.std()) # Outputs ~1
```

In the `CausalSelfAttention` and `MLP` class, the following is added in their `__init__` methods in order to implement this:

```
self.c_proj.NANOGPT_SCALE_INIT = 1
```
This flag indicates that the module should use scaled initialization.
Then, in the `GPT` class where parameters are initialized, there is a check for this flag. By default, weights are initialized with standard deviation `std=0.02` But if the module has the `NANOGPT_SCALE_INIT` attribute, there is an adjustment by dividing `std` by `sqrt(num_layers)`.

### Tensor Cores
Tensor Cores are specialized hardware instructions in NVIDIA’s A100 GPU architecture designated to accelerate matrix multiplication operations. Specifically, they perform small 4x4 matrix multiplications very efficiently. Since most of the computation in a transformer model like GPT-2 is dominated by matrix multiplications - especially within the linear layers - Tensor Cores play a critical role in speeding things up.

While there are other operations such as residual additions and nonlinearities (like GELUs), these are relatively insignificant in terms of computational cost. In the 124M parameter GPT-2 model, the most expensive operation by far is the final classification layer - a large matrix multiplication from 768 to 50,257 dimensions. This single operation dwarfs all others in terms of computational load.

Tensor Cores help accelerate these matrix multiplications dramatically. For instance, using FP32 (32-bit floating point), they can provide up to an 8x speedup. However, this speedup comes with a trade-off: a slight reduction in precision. Although the inputs, outputs, and accumulations are in FP32, the internal operations may use reduced precision to boost performance. This makes the results slightly more approximate.

In practice, though, this approximation is negligible - training results remain virtually unaffected. That’s why Andrej favors using FP32 with Tensor Cores. The slight precision loss is often imperceptible, and the performance gains are substantial and essentially “free,” as this optimization happens entirely under the hood without requiring changes to your code. It’s a sweet spot in performance vs. accuracy.

### Torch.Compile
In this section, Andrej introduces `torch.compile`, a powerful optimization feature from the PyTorch team. It acts like a compiler for your neural network code - similar to how GCC compiles C/C++ - and can dramatically speed up training and inference by reducing Python overhead and optimizing GPU usage. Using `torch.compile` is extremely simple. You wrap your model in a single line: 

```
model = torch.compile(model)
```

The trade-off is a small upfront compilation time, but the payoff is significantly faster runtime. Andrej emphasizes there’s rarely a good reason not to use it - it’s becoming the default way to run PyTorch models efficiently. The performance gains come from two main optimizations:

Python Overhead Removal: The model is compiled into a single, optimized object that no longer relies on the Python interpreter during execution.
GPU Kernel Fusion: Instead of sending data to the GPU for each individual operation, `torch.compile` combines (or “fuses”) multiple element-wise operations into a single kernel. This minimizes costly memory read/write operations between the CPU and GPU - a major bottleneck in deep learning workloads.

As Andrej puts it, having the full view of the computation ahead of time lets the system plan smarter memory access patterns, which leads to better performance.

### Flash Attention
While `torch.compile` is a powerful tool for optimizing PyTorch code, it doesn’t catch every possible optimization. One notable example is FlashAttention, introduced in the 2022 Stanford paper “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”. FlashAttention is a highly optimized algorithm for computing attention, designed to be significantly faster and more memory-efficient than the standard approach. In the traditional implementation of attention, we typically write:

```
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k..size(-1)))
att = att.masked_fill(self.bias(:,:,:T,:T) == 0, float(‘inf’))
att = F.softmax(att, dim=-1)
y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
```
This can be replaced with a single, highly optimized call to: 

```
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```
This function leverages FlashAttention under the hood, which effectively fuses multiple operations into a single, efficient kernel. `torch.compile` cannot automatically discover this optimization because it requires a fundamental algorithmic rewrite of how attention is computed - not just standard kernel fusion. 

What makes FlashAttention remarkable is its performance: although it performs more FLOPs than the naive implementation, it can be up to 7.6x faster, according to the paper. This speed up comes from being IO-aware: it minimizes the number of reads and writes to high-bandwidth memory (HBM) by carefully managing on-chip resources like shared memory. Unlike the naive approach, it avoids materializing the full N x N attention matrix in HBM, which drastically improves memory efficiency and throughput.

### Nice/ugly numbers
Andrej talks about “nice” and “ugly” numbers - terms that refer to how well a number plays with hardware like GPUs. Nice numbers are typically powers of 2, such as 64 or 128. These are preferred because many deep learning operations (especially on CUDA) are optimized for them. Ugly numbers, like 9 or 13, don’t align well with this structure and can cause inefficiencies. 

One example of an ugly number in the code is the original `vocab_size` of 50,257. It’s an odd number and not particularly friendly for CUDA’s tile/block computation structure. A better alternative would be rounding it up to something like 50,304 which is a “nicer” number - closer to a multiple of 64. While this technically increases the number of computations, it often results in faster execution because the hardware is optimized for handling such sizes efficiently.

CUDA kernels typically operate on chunks (or tiles) of size 32, 64, etc. If your data doesn’t neatly fit into these chunks, extra boundary-handling logic gets triggered, which is slower. By padding your inputs to a nice number, you avoid this and allow CUDA to process everything in clean, efficient blocks.

This seemingly small change - just rounding up to a nice number - can lead to around a 4\% speedup. Interestingly, this is the kind of low-level performance detail that `torch.compile` doesn’t catch, showing that manual optimization still has its place.

### GPT-3 vs GPT-2 Key Differences
The GPT-3 paper provides more detailed insights into the training process compared to GPT-2. However, unlike GPT-2, the weights for GPT-3 have not been publicly released. Architecturally, GPT-3 and GPT-2 are very similar, with the main difference being an increased context window: GPT-3 uses a context length of 2048 tokens, compared to GPT-2’s 1024. GPT-3 is also dramatically larger, scaling up from GPT-2’s 1.5 billion parameters to 175 billion, and was trained on a significantly more diverse and expansive dataset. 

Other important distinctions include:
- Training Data: ~570GB of filtered text data vs ~40GB.
- Scale Efficiency: GPT-3 demonstrated that simply scaling up model size, data, and compute - without major architectural changes - can lead to strong performance improvements on a wider range of NLP tasks.
- Few-Shot Learning: GPT-3’s increased size and training diversity enabled it to perform much better at a few-shot and zero-shot learning compared to GPT-2, which typically required fine-tuning to perform well on downstream tasks.
- Layer Normalization Placement: GPT-3 reportedly uses pre-layer normalization (as opposed to GPT-2’s post-layer norm), which improves training stability at large scale.
- No Weight Sharing: Unlike some earlier models, GPT-3 does not share weights between layers or between the embedding and output layers.

Despite these advancements, GPT-3 remains a decoder-only transformer like GPT-2, and both models are autoregressive.

### Gradient Clipping
To prevent exploding gradients, we clip the global norm of the gradients to a maximum value of 1.0. After calling `loss.backward()`, the gradients are stored in the `.grad` attributes of the model’s parameters. Sometimes, especially in deep networks, these gradients can become too large and destabilize training. Gradient clipping helps keep them in check by scaling them down if their total norm exceeds a specified threshold.

In PyTorch, this is commonly done with the following utility function:
```
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

This clips the gradients of all model parameters to ensure their global norm doesn’t exceed 1.0, while preserving their direction as much as possible.

Learning Rate Scheduler
The learning rate scheduler used in this tutorial is based on a technique called cosine decay with warm-up. This means the learning rate starts at zero and increases linearly during a warm-up phase. After reaching a peak, it decays following a cosine-shaped curve back down to zero.

In the GPT-3 paper, the authors describe a similar strategy: “We use cosine decay for the learning rate down to 10\% of its value, over 260 billion tokens (after 260 billion tokens, training continues at 10\% of the original learning rate).” 

However, the implementation in this tutorial differs slightly. In GPT-3, the decay phase ends before the total number of training steps, with the learning rate plateauing at 10\% for the remainder. In contrast, the tutorial’s implementation decays all the way to zero, with the decay duration equal to the total number of training steps.

### Batch Size Schedule
In the GPT-3 paper, a gradual batch size increase is used - starting with a small batch size and linearly ramping it up over time. This technique helps with system throughput and stability during training, especially at scale.

However, Andrej chooses to skip this approach in his GPT-2 reproduction tutorial. He avoids dynamic batch sizing because it complicates the arithmetic of training - specifically, it changes the number of tokens processed per optimization step, making things less transparent. He prefers to keep the setup simple and focused on core concepts.

Moreover, Andrej argues that gradually increasing batch size is not an algorithmic optimization improvement (i.e., it does not fundamentally improve how the model learns) but rather a system-level or training-speed optimization. In the early stages of training, when the model is still in a relatively naive state, it primarily learns broad token frequencies and simple biases - such as which tokens to ignore. During this phase, gradients from different examples tend to be highly correlated, making large batch sizes inefficient.

Only later in training, when gradients become more de-correlated and nuanced, does a larger batch size provide more statistical benefit. So, in his simplified setup, Andrej opts for a constant batch size, keeping the implementation straightforward while still capturing the essence of model training.

### Weight Decay
Weight decay is used in all models as a form of regularization. In Andrej’s implementation, a weight decay value of 0.1 is used, which is 10x larger than the default 0.01 used in the AdamW optimizer (as seen in models like GPT-3).

The implementation involves splitting parameters into two groups: those that should be affected by weight decay and those that shouldn’t. Typically, biases and one-dimensional parameters - such as those used in LayerNorms, scaling factors, and biases - are excluded from weight decay. This is because it doesnt make conceptual sense to penalize those small-scale parameters. 

Instead, weight decay is applied to the weights involved in matrix multiplications and embeddings, which are the primary contributors to the model’s capacity. Regularizing these weights helps prevent any single parameter from becoming excessively large. This encourages the network to distribute learning across multiple channels, leading to better generalization. Andrej describes this effect as a kind of “gravitational pull” on the weights, nudging them toward smaller magnitudes and preventing over-reliance on individual units.

### FusedAdamW
The implementation first checks if the `fused` optimizer is available, and uses it if so. By default, it’s not enabled because it’s still relatively new and needs more time to mature. When running on CUDA, `FusedAdamW` offers significant speedups. Normally, updating parameters involves looping over each tensor and launching a separate CUDA kernel for each, which adds overhead. `FusedAdamW` fuses these operations into a single kernel launch, reducing overhead and improving performance. In essence, it’s kernel fusion applied to the AdamW optimizer.

### Gradient Accumulation
Gradient accumulation allows us to simulate large batch sizes by splitting them into multiple smaller “micro-batches” that are processed sequentially. This is especially useful when working with limited GPU memory. For example, even if we want to simulate a total batch size of 524,288 tokens (~0.5M), we can still do it by running more steps and accumulating gradients across multiple forward and backward passes.

Here is a simple setup used in the tutorial:
```
total_batch_size = 524288 # 2**19, ~.5M, in number of tokens
B = 16 # micro batch size
T = 1024 # sequence length
Assert total_batch_size \% (B * T) == 0, “make sure total_batch_size is divisible by B * T”
grad_accum_steps = total_batch_size // (B * T)
print(f”total desired batch size: {total_batch_size}”)
print(f”=> calculated gradient accumulate steps: {grad_accum_steps}”)
```
By default, `F.cross_entropy` uses ‘mean’ reduction, meaning it averages the loss over all `B*T` tokens in a micro-batch. However, when performing gradient accumulation, we need to scale the loss ourselves to ensure correct normalization. Dividing the loss by `grad_accum_steps` ensures that when we sum up gradients across steps (via `loss.backward()`), we’re correctly simulating the effect of a large batch that would have been processed all at once.

Here is the loop with gradient accumulation:
```
for micro_step in range(grad_accum_steps):
	x, y = train_loader.next_batch()
	x, y = x.to(device, y.to(device)
	with torch.autocast(device_type=device, dtype=torch.bloat16):
		logits, loss = model(x,y)
	loss = loss / grad_accum_steps # normalize loss across accumulation steps
	loss.backward()
```

In summary, gradient accumulation gives the flexibility to train with arbitrarily large effective batch size by spreading the computation across multiple smaller, memory-friendly micro-batches, while still maintaining the correct gradient scale.

### Distributed Data Parallel
Distributed Data Parallel (DDP) allows training to be scaled across multiple GPUs by launching one process per GPU. In Andrej’s implementation, 8 GPUs are used, so `torchrun` launches 8 separate processes - each bound to a specific GPU. Each process runs the same training code independently but operates on a different subset of the data. Behind the scenes, PyTorch handles synchronizing the gradients across all GPUs by averaging them during the backward pass. This ensures all models stay in sync after each update, effectively simulating data-parallel training.

### torchrun
`torchrun` is a utility that simplifies launching distributed training jobs. When executed, it spawns multiple processes (one per GPU) and sets important environment variables so that each process knows its role in the distributed setup. Specifically, `torchrun` sets:
- `RANK`: The global rank (unique ID) of the process across all nodes.
- `LOCAL_RANK`: The rank of the process on the local machine (used to bind to the correct GPU).
- `WORLD_SIZE`: The total number of processes participating in training.

These environment variables are used internally by PyTorch’s DDP to coordinate communication and gradient synchronization among all processes.

### Datasets Used In GPT-3
In the GPT-3 paper, the authors explain that they created a new web scraper focused on high-quality documents. Instead of scraping the entire web indiscriminately, they filtered for pages that had already been curated by humans. As a practical heuristic, they scraped all outbound links from Reddit posts that had received at least 3 karma - under the assumption that links upvoted by users were likely to be interesting, informative, or entertaining. This dataset, however, was not publicly released. An open-source effort to replicate it is called OpenWebText.

For his GPT-2 (124M) reproduction tutorial, Andrej Karpathy uses a dataset called FineWeb, which is based on Common Crawl data. FineWeb is filtered and preprocessed to emphasize high-quality content. Hugging Face released versions of FineWeb containing up to 1.3 trillion tokens of educational content and 5.4 trillion tokens of “highly educational” content. In Andrej’s implementation, he uses a specific subset of FinWeb named `sample-10BT`, which is a randomly sampled corpus containing around 10 billion GPT-2 tokens from the larger dataset.

### Validation Data Split
A `val_loader` is created by passing `split=“val”` to the data loader, which gives us access to the validation shard of the dataset. To support this, a `reset(self)` method is added to the `DataLoaderLite` class and called within its `__init__`. This method reinitializes the loader’s internal state, ensuring it starts fresh for each validation pass.  

During training, at every 100th iteration (including the 0th), the model is switched to evaluation mode. The `val_loader` is reset and validation loss is computed without gradient tracking. The loss is accumulated over, say, 20 steps and then averaged. This logic mirrors the training loop, except it skips `loss.backward()` - it’s pure inference, used solely to track validation performance. 

In PyTorch (and most deep learning frameworks), the typical training loop has three main steps:
1. Forward pass - Compute the model’s output (logits) from the input.
2. Loss computation - Calculate how wrong the predictions are.
3. Backward pass - Call `loss.backward()` to compute gradients.
4. Optimizer step - Update model weights based on gradients

When you’re doing inference (i.e., just evaluating the model, not training it), you only care about steps 1 and 2 - you want to know how well the model is doing and you do not want to compute gradients or update weights. By skipping `loss.backward()` you’re skipping the gradient computation step, which is a hallmark of training. That’s why it’s called inference - you’re just running the model to observe its predictions or performance, not trying to make it better at anything in that moment. To optimize this further during validation/inference, we also wrap the code in: `with torch.no_grad():` which tells PyTorch: don:t build the computation graph, don’t track operations for gradients, and use less memory and compute.

### Sampling Revive
The sampling code should look familiar, but there’s a key change: a dedicated PyTorch `Generator` object is now used for sampling. This allows for direct control over the random number generation process, ensuring that sampling does not interfere with the global RNG state used during training. To achieve this separation, a special RNG is created solely for sampling, and it is explicitly seeded so that each process (or rank) receives a different seed. This generator is then passed to the `torch.multinomial` function, which handles the actual sampling.

One caveat: the model runs slightly slower because this setup is currently incompatible with `torch.compile`. Attempting to use `torch.compile` may result in a PyTorch error, though this issue might have been resolved by the time Andrej committed his code to GitHub.

### HellaSwag
HellaSwag is a multiple-choice sentence completion dataset designed to test a model’s ability to reason about everyday situations. Each example provides a short context, followed by four possible continuations. For instance, given a context like:

“A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She…”
A. Rinses the bucket of the soap and blow dries the dog’s head.
B. Uses a hose to keep it from getting soapy.
C. Gets the dog wet, then it runs away again.
D. Gets into a bathtub with the dog.

Only one of these options is the correct, natural continuation. The remaining choices are adversarially generated - they are syntactically plausible and often grammatically correct, but semantically or logically off. This makes them deceptively difficult for language models to distinguish, even though humans can easily pick the right answer. 

The dataset pulls its sentence contexts from sources like ActivityNet and WikiHow, covering a broad range of everyday domains such as “Home and Garden”, “Computers and Electronics” etc. The paper includes a helpful diagram showing the diversity of topics from WikiHow.

The key idea is that well-trained models with strong world knowledge should perform well, while weaker models will struggle with the subtle distinctions. HellaSwag is therefore a useful benchmark for evaluating a model’s common sense reasoning abilities. 

### Token Completion
Andrej evaluates the model using a token completion approach rather than a traditional multiple-choice format. The problem with multiple-choice for this small model is that it likely doesn’t understand the concept of associating a label (like A, B, C, or D) with one of the options. Instead, the model is tested in its native form: predicting token sequences.

To do this, a batch is constructed with 4 rows (one for each option) and T tokens (the length varies). The shared context - the initial prompt common to all options - is repeated across all rows. Then, each row appends one of the four candidate completions, with exactly one correct option (e.g., option 3). Since the options might be of different lengths, the batch length is set to the longest option, and shorter options are padded. A mask is created to indicate which tokens are valid (mask=1) and which are padding (mask=0).

During evaluation, the language model computes the probabilities of each token in each option. We then calculate the average token probability for each option and select the one with the highest average probability as the model’s predicted completion. This effectively turns the multiple-choice problem into a token completion scoring task. 

Andrej believes this is similar to how GPT-3 handles HellaSwag. However, some implementations of HellaSwag treat it as a classic multiple-choice task, where the model sees the context once and all options simultaneously-allowing it to compare choices directly. This is an easier task, but requires larger models with greater capacity. Since Andrej’s model is small, it evaluates each option independently without access to the others, making the task more challenging.

### Checkpointing
After evaluating the validation loss, the master process logs the result and saves a checkpoint every 5000 steps (for Andrej’s implementation). A checkpoint includes the model’s state dictionary, which allows you to save and later resume training or perform evaluations.

In addition to the model state, the optimizer state dictionary must also be saved. This is important because optimizers like Adam maintain internal buffers (e.g., the first and second moment estimates, `m` and `v`), which are essential for correct resumption of training. In the context of the Adam optimizer, `m` and `v` refer to moving averages of the gradients, which help the optimizer adaptively adjust the learning rate for each parameter. `m` represents the first moment estimate, which is essentially the exponentially decaying average of past gradients. It captures the direction the gradients are pointing on average. `v` represents the second moment estimate, the exponentially decaying average of the squared gradients. It captures how large the gradients are on average (their variance).

Care should also be taken to preserve RNG seeds to ensure reproducibility when resuming training.

Checkpointing is useful not only for continuing training but also for performing more rigorous evaluations. For instance, instead of the quick HellaSwag evaluation used in the tutorial, you could use a more comprehensive evaluation framework like the Eleuthera Evaluation Harness.

### llm.c
LLMs in simple, pure C/CUDA with no need for 245MB of PyTorch of 107MB of cPython. Current focus is on retraining, in particular reproducing the GPT-2 and GPT-3 miniseries, along with a parallel PyTorch reference implementation in `train_gpt2.py`. 

`llm.c` is faster than our implementation in regards to start up and getting to stepping and it is faster per step, so overall it is faster.


