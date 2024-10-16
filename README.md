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

