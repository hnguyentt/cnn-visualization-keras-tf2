import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model

class FeaturesExtraction:
    def __init__(self,model,layername):
        self.model = model
        self.layername = layername
        
        self.feature_model = Model(inputs=model.inputs,outputs=model.get_layer(layername).output)
    
    def extract_features(self, img):
        return self.feature_model.predict(img)
    
def vis_feature_map(feature_maps):
    ncol = min(8,int(np.floor(np.sqrt(feature_maps.shape[3]))))
    fig, ax = plt.subplots(ncol, ncol,figsize=(2*ncol,ncol*1.5))
    if ncol == 1:
        ax.imshow(feature_maps[0,:,:,0],cmap="gray")
    else:
        count = 0
        for i in range(ncol):
            for j in range(ncol):
                ax[j,i].imshow(feature_maps[0,:,:,count],cmap="gray")
                ax[j,i].axis("off")
                count += 1
    plt.tight_layout()
    plt.show()
            