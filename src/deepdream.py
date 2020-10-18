import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
import cv2

# Hyperparameters
"""
step = 0.01  # Gradient ascent step size
num_octave = 3  # Number of scales at which to run gradient ascent
octave_scale = 1.4  # Size ratio between scales
iterations = 20  # Number of ascent steps per scale
max_loss = 15.0
"""

def preprocess_deep_dream(x, preprocess_input):
    # Check if image size is to large?
    if x.shape[0]*x.shape[1] > 512*512:
        coef = np.sqrt(x.shape[0]*x.shape[1])/512
        h = int(x.shape[0]/coef)
        w = int(x.shape[1]/coef)
        x = cv2.resize(x,(w,h))
    x = preprocess_input(x)
    return np.expand_dims(x,axis=0)
    
def deprocess_deep_dream(x):
    # Util function to convert a NumPy array into a valid image.
    x = x.reshape((x.shape[1], x.shape[2], 3))
    # Undo inception v3 preprocessing
    x /= 2.0
    x += 0.5
    x *= 255.0
    # Convert to uint8 and clip to the valid range [0, 255]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

class DeepDream:
    def __init__(self,model, conv_layers):
        self.model = model
        self.conv_layers = conv_layers
        self.layer_settings = self.rand_layer_settings()
        
        outputs_dict = dict(
            [
                (layer.name, layer.output)
                for layer in [model.get_layer(name) for name in self.layer_settings.keys()]
            ]
        )
        self.feature_extractor = Model(inputs=model.inputs, outputs=outputs_dict)
        
    def rand_layer_settings(self):
        idx = random.sample(range(0,len(self.conv_layers)),4)
        chosen_layers = [self.conv_layers[i] for i in idx]
        values = [2.0,2.5,3.0,3.5]

        return dict(zip(chosen_layers,values))

    def compute_loss(self,input_image):
        features = self.feature_extractor(input_image)
        # Initialize the loss
        loss = tf.zeros(shape=())
        for name in features.keys():
            coeff = self.layer_settings[name]
            activation = features[name]
            # We avoid border artifacts by only involving non-border pixels in the loss.
            scaling = tf.reduce_prod(tf.cast(tf.shape(activation), "float32"))
            loss += coeff * tf.reduce_sum(tf.square(activation[:, 2:-2, 2:-2, :])) / scaling
        return loss

    @tf.function
    def gradient_ascent_step(self,img, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = self.compute_loss(img)
        # Compute gradients.
        grads = tape.gradient(loss, img)
        # Normalize gradients.
        grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
        img += learning_rate * grads
        return loss, img


    def gradient_ascent_loop(self, img, iterations, learning_rate, max_loss=None):
        for i in range(iterations):
            loss, img = self.gradient_ascent_step(img, learning_rate)
            if max_loss is not None and loss > max_loss:
                break
            print("... Loss value at step %d: %.2f" % (i, loss))
        return img


    def generate_deep_dream(self,original_img,iterations=20,step=0.01,max_loss=15.0,num_octave=3, octave_scale=1.14):
        original_shape = original_img.shape[1:3]

        successive_shapes = [original_shape]
        for i in range(1, num_octave):
            shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
            successive_shapes.append(shape)
        successive_shapes = successive_shapes[::-1]
        shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])

        img = tf.identity(original_img)  # Make a copy
        for i, shape in enumerate(successive_shapes):
            print("Processing octave %d with shape %s" % (i, shape))
            img = tf.image.resize(img, shape)
            img = self.gradient_ascent_loop(
                img, iterations=iterations, learning_rate=step, max_loss=max_loss
            )
            upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
            same_size_original = tf.image.resize(original_img, shape)
            lost_detail = same_size_original - upscaled_shrunk_original_img

            img += lost_detail

        return img
