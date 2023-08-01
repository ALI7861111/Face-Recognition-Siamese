
# Face Recognition Dataset

Face Recognition Dataset is a collection of face photographs designed for the creation of face detection and recognition models. This dataset has been derived from the Labeled Faces in the Wild (LFW) Dataset. The original images are collected from the internet and feature pictures of famous people. The dataset provides a valuable resource for researchers and developers working on facial recognition tasks.

# Dataset Details

Each picture in the dataset is centered on a single face and encoded in RGB format. The original images are of size 250 x 250 pixels.
The dataset consists of 1680 directories, each representing a different celebrity.
Each celebrity directory contains 2 to 50 images of the individual.

##### Extracted Faces

-For ease of use in training a classifier, the dataset includes a folder named "Extracted Faces." In this folder, faces have been extracted from the original images using the Haar-Cascade 

-Classifier from the OpenCV library. The extracted faces are encoded in RGB format and have a size of 128 x 128 pixels.

We will be using the "Extracted Faces" folder in our work, as working with zoomed-in face pictures can enhance the classifier's performance.

## Challenges with Large-Class Datasets
The facial recognition dataset poses unique challenges due to its large number of classes (1678 in this case). Training traditional machine learning or deep learning models on such datasets can be computationally expensive and challenging due to limited data available per class.

## Few-Shot Learning and Siamese Models
To address these challenges and improve the classifier's performance, we can explore two promising approaches:

## Few-Shot Learning
Few-shot learning is a machine learning approach that enables models to recognize new classes with very few labeled examples. It leverages knowledge learned from previously seen classes to adapt quickly to new classes with limited examples. One popular few-shot learning method is meta-learning, where models are trained on multiple related tasks to learn how to learn efficiently and generalize to new tasks with few examples.

## Siamese Models
Siamese models are deep neural networks that learn a similarity function between pairs of inputs. These models consist of two identical subnetworks that share weights and are trained to minimize the distance between the representations of similar pairs and maximize the distance between dissimilar pairs. Siamese models are particularly useful for facial recognition tasks, as they can compare the similarity between two faces even when they have different orientations, lighting conditions, or facial expressions.

## Choosing the Best Model
To determine the most suitable model for our facial recognition task, we will experiment with different pretrained models in the Siamese Network architecture. The model that performs best in terms of accuracy and generalization will be selected for this task.

## Siamese Network with Pretrained Models
We will proceed with building a Siamese Network using various pretrained models. By evaluating the performance of each model, we will identify the one that demonstrates superior performance and is well-suited for our facial recognition task.

For more details and the latest updates, please visit the official website of the dataset: http://vis-www.cs.umass.edu/lfw/

If you use this dataset in your research or projects, kindly acknowledge the Labeled Faces in the Wild (LFW) dataset as the original source.

## Citation
The original LFW Dataset and related works have been mentioned in various academic papers. If you use this dataset, please refer to the relevant citations to give proper credit to the creators and contributors of the dataset.