from vis.visualization import visualize_cam 
from vis.visualization import visualize_saliency
from matplotlib import pyplot as plot
from skimage.io import imread
import matplotlib.cm as cm
import numpy as np

def load_model_from_JSON(modelpath, weightpath) :
    from keras.models import model_from_json
    json_file = open(modelpath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weightpath)
    return loaded_model

def get_image(filepath) : 
    img = imread(filepath, as_gray = True)
    if(img.dtype != "uint8") :
            img *= 255
            img = img.astype("uint8")
    #plot.imshow(img)
    if len(img[0]) < 499 :
        img = np.reshape(img, (512, img.shape[1], 1))
        index = int((512-len(img[0]))/2) +1
        end_index = 512-index
        fin = np.full((512,512,1), 0)
        for r in range(512) :
            for c in range(index, end_index) :
                fin.itemset((r,c,0), img.item((r,c-index,0)))
    return fin


model = load_model_from_JSON("./networkdata/modelold.json", "./networkdata/weightsold.h5")
img = get_image("./networkdata/testimg.png")
X_test = np.asarray(img)
predicts = model.predict(np.array([X_test,]), verbose=1)
(breakp, nop) = predicts[0]
plot.xlabel("Break prob: {}".format(breakp))
cam_map = visualize_cam(model, layer_idx=17, filter_indices=0, seed_input=img, backprop_modifier=None, grad_modifier="absolute")
print(img.shape)
#plot.imshow(visualize_cam(model, layer_idx=17, filter_indices=0, seed_input=get_image("./networkdata/testimg.png"), penultimate_layer_idx=None, backprop_modifier=None, grad_modifier=None))
plot.imshow(img[:,:,0])
plot.imshow(cam_map, alpha=0.5)