# CNN Visualization and Explanation
This work aims to:
* Visualize filters and feature maps of all pre-trained models on ImageNet in [`tf.keras.applications`](https://github.com/conan7882/CNN-Visualization) with `Tensorflow` verion 2.3.0. The visualization methods include simply plotting filters of the model, plotting the feature maps of convolutional layers, DeConvNet and Guided Backpropagation
* Explain for the top 5 predictions of these models by GradCAM and Guided-GradCAM

With the current version, there are 26 pre-trained models.
## Briefs

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-fymr{border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg" style="undefined;table-layout: fixed; width: 908px">
<colgroup>
<col style="width: 188px">
<col style="width: 259px">
<col style="width: 461px">
</colgroup>
<thead>
  <tr>
    <th class="tg-c3ow"><span style="font-weight:bold">Method</span></th>
    <th class="tg-c3ow"><span style="font-weight:bold">Brief</span></th>
    <th class="tg-c3ow"><span style="font-weight:bold">**Example**</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-fymr">Filter visualization</td>
    <td class="tg-0pky">Simply plot the learned filters.<br>* Step 1: Find a convolutional layer.<br>* Step 2: Get weights at a convolution layer, they are filters at this layer.<br>* Step 3: Plot filter with the values from step 2.<br>This method does not requre an input image.</td>
    <td class="tg-c3ow">VGG16, <span style="color:#905;background-color:#DDD">`block1_conv1`</span><br><br><img src="https://raw.githubusercontent.com/nguyenhoa93/cnn-visualization-keras-tf2/master/images/filtervisVGG16_block1_conv1.png" alt="Image" width="400" height="142"></td>
  </tr>
  <tr>
    <td class="tg-fymr">Feature map visualization</td>
    <td class="tg-0pky">Plot the feature maps obtained when fitting an image to the network.<br>* Step 1: Find a convolutional layer.<br>* Step 2: Build a feature model from the input up to that convolutional layer.<br>* Step 3: Fit the image to the feature model to get feature maps.<br>* Step 4: Plot the feature map.</td>
    <td class="tg-c3ow">VGG16, <span style="color:#905;background-color:#DDD">`block1_conv1`</span><br><img src="https://raw.githubusercontent.com/nguyenhoa93/cnn-visualization-keras-tf2/master/images/featurevisVGG16_block1_conv1.png" alt="Image" width="400" height="315"><br><br><span style="font-weight:400;font-style:normal">VGG16, </span>`block5_conv3`<br><br><img src="https://raw.githubusercontent.com/nguyenhoa93/cnn-visualization-keras-tf2/master/images/featurevisVGG16_block5_conv3.png" alt="Image" width="400" height="315"></td>
  </tr>
  <tr>
    <td class="tg-fymr">Guided Backpropagation</td>
    <td class="tg-0pky">Backpropagate from a particular convolution layer to input image with modificaton of the gradient of ReLU.</td>
    <td class="tg-c3ow">VGG16, <span style="color:#905;background-color:#DDD">`block1_conv1`</span><br><img src="https://raw.githubusercontent.com/nguyenhoa93/cnn-visualization-keras-tf2/master/images/guidedbackpropVGG16_block1_conv1.png" alt="Image" width="231" height="231"><br><br>VGG16, `block5_conv3`<br><img src="https://raw.githubusercontent.com/nguyenhoa93/cnn-visualization-keras-tf2/master/images/backguidedVGG16_block5_conv3.png" alt="Image" width="231" height="231"></td>
  </tr>
  <tr>
    <td class="tg-fymr">GradCAM</td>
    <td class="tg-0pky">* Step 1: Determine the last convolutional layer<br>* Step 2: Perform gradient from `pre-softmax` layer to last convolutional layer and the apply global average pooling to obtain weights for neurons' importance.<br>* Step 3: Linearly combinate feature map of last convolutional layer and weights, then apply ReLu on that linear combination.</td>
    <td class="tg-c3ow">GradCAM &amp; Guided GradCAM for class l<span style="font-weight:bold">akeside</span>, InceptionV3<br><img src="https://raw.githubusercontent.com/nguyenhoa93/cnn-visualization-keras-tf2/master/images/lakesideInceptionV3.png" width="400" height="202"></td>
  </tr>
  <tr>
    <td class="tg-fymr">Guided-GradCAM</td>
    <td class="tg-0pky">* Step 1: Calculate guided backpropagation from last convolutional layer to input.<br>* Step 2: Upsample GradCAM to the size of input<br>* Step 3: Apply element-wise multiplication of guided backpropagation and GradCAM</td>
    <td class="tg-c3ow">GradCAM &amp; Guided GradCAM for class <span style="font-weight:bold">**boathouse**</span>, InceptionV3<br><br><img src="https://raw.githubusercontent.com/nguyenhoa93/cnn-visualization-keras-tf2/master/images/boathouseInceptionv3.png" width="400" height="202"></td>
  </tr>
  <tr>
    <td class="tg-fymr">Deep Dream</td>
    <td class="tg-0pky">See more in this excellent tutorial from François Chollet: <a href="https://keras.io/examples/generative/deep_dream/"><span style="color:#905">https://keras.io/examples/generative/deep_dream/</span></a></td>
    <td class="tg-c3ow">InceptionV3<br><br><img src="https://raw.githubusercontent.com/nguyenhoa93/cnn-visualization-keras-tf2/master/images/deepdreamInceptionv3.png" width="400" height="266"></td>
  </tr>
  <tr>
    <td class="tg-baqh" colspan="3">Original image<br><br><img src="https://raw.githubusercontent.com/nguyenhoa93/cnn-visualization-keras-tf2/master/images/lapan.jpg" width="700" height="466"></td>
  </tr>
</tbody>
</table>

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
4. Deep Dream Tutorial from François Chollet: https://keras.io/examples/generative/deep_dream/ 