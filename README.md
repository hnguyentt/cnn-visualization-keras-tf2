# CNN Visualization and Explanation
This work aims to:
* Visualize filters and feature maps of all pre-trained models on ImageNet in [`tf.keras.applications`](https://github.com/conan7882/CNN-Visualization) with `Tensorflow` verion 2.3.0. The visualization methods include simply plotting filters of the model, plotting the feature maps of convolutional layers, DeConvNet and Guided Backpropagation
* Explain for the top 5 predictions of these models by GradCAM and Guided-GradCAM

With the current version, there are 26 pre-trained models.
## Briefs
| **Method**                | **Brief**                                                                                                                                                                                                                                                                                               | **Example** |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------:|
| Filter visualization      | Simply plot the learned filters.<br>* Step 1: Find a convolutional layer.<br>* Step 2: Get weights at a convolution layer, they are filters at this layer.<br>* Step 3: Plot filter with the values from step 2.<br>This method does not requre an input image.                                         |             |
| Feature map visualization | Plot the feature maps obtained when fitting an image to the network.<br>* Step 1: Find a convolutional layer.<br>* Step 2: Build a feature model from the input up to that convolutional layer.<br>* Step 3: Fit the image to the feature model to get feature maps.<br>* Step 4: Plot the feature map. |             |
| Guided Backpropagation    | Backpropagate from a particular convolution layer to input image with modificaton of the gradient of ReLU.                                                                                                                                                                                              |             |
## How to use
### Run with your resource
* Clone this repo:
```bash
git clone https://github.com/nguyenhoa93/cnn-visualization-keras-tf2
cd cnn-visualization-keras-tf2
```
* Create virtualev:
```bash
conda create -n cnn-vis python=3.6
conda activate cnn-vs
bash requirements.txt
```
* Run demo with the file `visualization.ipynb`

### Run on Google Colab
(to be updated)

## References
1. [How to Visualize Filters and Feature Maps in Convolutional Neural Networks](https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/) by Machine Learning Mastery
2. Pytorch CNN visualzaton by [utkuozbulak](https://github.com/utkuozbulak): https://github.com/utkuozbulak
3. CNN visualization with TF 1.3 by [conan7882](https://github.com/conan7882): https://github.com/conan7882/CNN-Visualization