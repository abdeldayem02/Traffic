# an AI to identify which traffic sign appears in a photograph, using a tensorflow convolutional neural network.

## Background:

As research continues in the development of self-driving cars, one of the key challenges is computer vision, allowing these cars to develop an understanding of their environment from digital images. In particular, this involves the ability to recognize and distinguish road signs – stop signs, speed limit signs, yield signs, and more.

In this project, you’ll use TensorFlow to build a neural network to classify road signs based on an image of those signs. To do so, you’ll need a labeled dataset: a collection of images that have already been categorized by the road sign represented in them.

Several such data sets exist, but for this project, we’ll use the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains thousands of images of 43 different kinds of road signs.


## Model Experimentation Process:

To build this model, I started with very simple models (models with small 'capacity' i.e. small number of learnable parameters), and then gradually added in more layers, increasing the complexity/capacity of the model. Each model was trained against the training data, and then evaluated using the testing data, each set of data randomly selected using Scikit-learn train_test_split (test size = 40%). I could then compare the accuracy of each model on the training set and the testing set. An ideal model would have high and similar accuracy on both the training and testing data sets.

Where a model has a higher loss on the testing data than the training data, this may suggest that the model is overfitting the training data, and so not generalising well onto the test data. When overfitting is severe, a model may be highly accurate (low loss) on the training data but have very poor accuracy (high loss) on the test data. Strategies to reduce overfitting of a model include reducing the capacity (complexity) of the model, adding 'dropout' to layers of the model, or adding weight regularization (penalizing large weights) [1](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#strategies_to_prevent_overfitting). However, while a simple model may reduce the risk of overfitting the training data, a model with insufficient capacity may suffer from higher loss for both the training and testing data. The capacity of the model must be tweaked to get the best results without overfitting.

There are many different model parameters that can be specified and tuned, e.g.:
* Different numbers of convolutional and pooling layers (learn features, and reduce image size/complexity)
* Different numbers and sizes of filters for convolution layers (the number of kernel matrices to train on and the size of the matrices)
* Different pool sizes for pooling layers (bigger pool size will reduce image size more)
* Different numbers and sizes of hidden layers (model complexity/capacity)
* Additional parameters for the model layers such as dropout, weight regularization, activation functions.
* Other model settings such as the optimizer algorithm, loss function and metric used to monitor the training and testing steps
* etc....!

To limit some of these choices, all models used the Adam optimisation algorithm, with categorical crossentropy for the loss function. Accuracy is a suitable metric to use for all models as we want to know the percentage of labels that were correctly predicted by the model. The output layer uses the "softmax" activation function, such that the output from the network is a normalised probability distribution (i.e. the predicted probability for the given image being each type of road sign). In addition, all hidden layers and convolutional layers will use the 'reLU' activation function.

First part of the model:

```python
tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

    # Max-pooling layer, using 3x3 pool size
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
```

**Conv2D** layer: This layer performs convolution on the input image with 32 filters, each of size 3x3, and using the ReLU activation function.

**MaxPooling2D** layer: This layer performs max-pooling on the output of the previous layer, using a 3x3 pool size. Here I Noticed that when I used 3x3 pool size the speed or the preformance of the model increased without lowering the accuracy of it.

Then I added another conviution layer to increase the accuracy of the model with another Max-pooling layer using 2x2 pool size.

**Flatten** layer flattens the output of the previous layer into a 1D array.

```python
 # Add another convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    
    # Add Another Max-pooling layer, using 2x2 pool size
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten units
    tf.keras.layers.Flatten(),
```
The Second part I added 2 hidden layers with 128 units each, using the ReLU activation function with a dropout layer is added after these layers to prevent overfitting.

```python
# Add 2 hidden layers with dropout
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.33),
```
The Last part was the output layer, this layer has NUM_CATEGORIES units and uses the softmax activation function to output the predicted probability distribution over the categories.

```python
# Add an output layer with output units
    tf.keras.layers.Dense(NUM_CATEGORIES,activation="softmax")
```
After defining the model architecture, the function compiles the model using the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.

```python
# Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
        )
    
    return model
    

```
Finally, the function returns the compiled model object. This function can be called to create a new instance of the model with the same architecture and parameters.

And The Final and overall model is here:

```python
model=tf.keras.models.Sequential([
        
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

    # Max-pooling layer, using 3x3 pool size
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

    # Add another convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    
    # Add Another Max-pooling layer, using 2x2 pool size
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten units
    tf.keras.layers.Flatten(),


    # Add 2 hidden layers with dropout
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.33),

    # Add an output layer with output units
    tf.keras.layers.Dense(NUM_CATEGORIES,activation="softmax")
    
    
    ])

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
        )
    
    return model

```
This model had a training and testing accuracy of 95% , and the training and testing loss were similar during trainig runs. The model appears to fit the training data well without overfitting, and generalises well to the testing data.
