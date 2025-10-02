# Neural Network CIFAR 10 Training Project

<img width="543" height="176" alt="image" src="https://github.com/user-attachments/assets/f3beda1e-994e-419d-accd-491c4ef57c7e" />
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/b08c737e-1318-4d3e-ae42-dfdac903bf56" />




# Dataset and Preprocessing
For training, the CIFAR10 dataset was used. The images were converted to tensors, but a series
of randomisation techniques were applied first. The images were randomly cropped to 32x32
images with a padding of 4, then flipped horizontally and rotated randomly between 0 degrees
and 15 degrees. Lastly, colour jitter was applied, and the image was converted to a tensor. The
tensor was then normalised with the dataset’s mean and standard deviation.
For testing, no modifications were made to the images, and they were converted to tensors as
well as normalised.
# Model Architecture
### Stem
The stem acts as the model’s initial feature extractor, transforming raw image input into a matrix
suitable for deeper processing in the backbone. The stem layer is responsible for ensuring that
the first feature map is rich with information so that the block can accurately extract relevant
features. At first, only a convolutional layer was implemented, but adding multiple
convolutional layers with batch normalisation and ReLU significantly improved the
performance.
### Backbone Blocks
The backbone blocks contain to branches: the expert branch and the main branch. The expert
branch applies average pooling followed by two fully connected layers with ReLU in between
and a SoftMax operation is applied to it. The main branch contains K convolutional layers which
are combined with the expert branch, creating a single output.
### Classifier
The classifier consists of average pooling, flattening, and a multi-layer perceptron (MLP) with
two hidden layers (1000 and 500 neurons). It outputs logits for 10 classes as the CIFAR10
dataset consists of 10 classes.
The model architecture is then joined altogether in the Net class. When the model is created, it
is moved to the GPU for a more efficient training and testing process. Cross entropy loss is set
and an Adam optimiser is also set. Originally, an SGD optimiser was used, but was switched
with Adam as Adam performs better.
# Hyper Parameters
The following hyperparameters were used:
- num_epochs = 120: Originally started with 50 but the model was under fitting.
Similarly, 100 epochs didn’t reach the 90% accuracy requirement and the model
was still improving. 120 was chosen instead to prevent overfitting while allowing the
model to continue improving
- batch_size = 64: Smaller batches affected training time and larger batches reduces
the frequency of changes per epoch. 64 was chosen as the sweet spot.
- in_channels = 3: CIFAR10 has 3 channels for RGB.
- out_channels = 64: The convolutional layers in the stem and backbone are projected
to 64 feature maps providing the model to identify expressive features without taking
up too much time
- block_number = 8: Lower values caused the model to under-perform and larger
values were very computationally expensive
- reduction_factor = 4: Reducing the r factor improved the accuracy of the model
through the first few epochs. A reduction factor of 2 affected performance and so 4
was chosen instead.
- k = 3: Large values of k significantly affected performance without improving
accuracy
- image_size = 32: CIFAR10 images are 32x32 images. This was set up as a modifiable
hyper-parameter due to experimenting with MaxPool. MaxPool didn’t improve
accuracy significantly and so was removed.
- learning_rate = 0.0001: I’m not too sure what this does, but I experimented with
0.0001 and 0.0005 and 0.0001 performed slightly better.
