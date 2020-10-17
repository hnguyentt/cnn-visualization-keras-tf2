import matplotlib.pyplot as plt

def extract_filter(model, layer_name):
    filters =  model.get_layer(layer_name).get_weights()[0]
    filters = filters[:,:,:,:6]
    # normalize values to 0-1
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min)/(f_max - f_min)

    return filters

def vis_filter(filters):
    n_filters = min([6,filters.shape[3]])
    fig, ax = plt.subplots(3, n_filters,figsize=(1.5*n_filters,3))
    for i in range(n_filters):
        f = filters[:,:,:,i]
        for j in range(3):
            if n_filters > 1:
                ax[j,i].imshow(f[:,:,j],cmap="gray")
                ax[j,i].axis("off")
            else:
                ax[j].imshow(f[:,:,j],cmap="gray")
                ax[j].axis("off")
    plt.tight_layout()
    plt.show()