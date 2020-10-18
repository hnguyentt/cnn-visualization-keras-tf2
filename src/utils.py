import tensorflow as tf
from PIL import Image
import io

# Constants
model_dicts = {
    "Xception": "xception",
    "VGG16": "vgg16",
    "VGG19": "vgg19",
    "ResNet50": "resnet50",
    "ResNet101": "resnet",
    "ResNet152": "resnet",
    "ResNet50V2": "resnet_v2",
    "ResNet101V2": "resnet_v2",
    "ResNet152V2": "resnet_v2",
    "InceptionV3": "inception_v3",
    "InceptionResNetV2": "inception_resnet_v2",
    "MobileMet": "mobilenet",
    "MobileNetV2": "mobilenet_v2",
    "DenseNet121": "densenet",
    "DenseNet169": "densenet",
    "DenseNet201": "densenet",
    "NASNetMobile": "nasnet",
    "NASNetLarge": "nasnet",
    "EfficientNetB0": "efficientnet",
    "EfficientNetB1": "efficientnet",
    "EfficientNetB2": "efficientnet",
    "EfficientNetB3": "efficientnet",
    "EfficientNetB4": "efficientnet",
    "EfficientNetB5": "efficientnet",
    "EfficientNetB6": "efficientnet",
    "EfficientNetB7": "efficientnet"
}

excluded_layers = ["bn","act","relu","out","add","pad","concat"]


# FUNCTIONS
def array2bytes(im_arr, fmt='png'):
    img = Image.fromarray(im_arr, mode='RGB')
    f = io.BytesIO()
    img.save(f, fmt)

    return f.getvalue()

def create_model(model_name, include_top=True):
    return getattr(tf.keras.applications,model_name)(weights="imagenet",include_top=include_top)

def get_conv_layers(model):
    layers = []
    for l in model.layers:
        if "conv" in l.name and all(x not in l.name for x in excluded_layers):
            layers.append(l.name)
    return layers

