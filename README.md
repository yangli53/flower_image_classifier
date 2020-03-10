# Developing an Image Classifier

In this project, I trained an image classifier to recognize different species of flowers. To train this classifier, I used the dataset of 
102 flower categories. 

The project is broken down into multiple steps:

- Load and preprocess the image dataset
- Train the image classifier on my dataset
- Use the trained classifier to predict image content

I used one of the pretrained models from `torchvision.models` to get the image features and then built and trained a new feed-forward 
classifier using those features.

Things I did:

- Load a pre-trained network, vgg16
- Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
- Train the classifier layers using backpropagation using the pre-trained network to get the features
- Track the loss and accuracy on the validation set to determine the best hyperparameters

I ran the test images using the model and got an accuracy of 83.9% indicating that the model had been trained well.

I saved the trained model with essential information and loaded the checkpoint and rebuilt the model.

I wrote a function that uses my model to predict the top 5 most probable flower classes with class probabilities.
