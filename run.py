import argparse
from conv_model import conv_model
from trainer import trainer
from data_loader import data_loader
from PIL import Image
import numpy as np
import tensorflow as tf
from skimage import transform

#Dataset URL: https://www.kaggle.com/puneet6060/intel-image-classification

#Training params
epochs = 4

#Load images batch size
batch_size = 32

#Dataset PATH
datapath = './dataset'

#Checpoint where NN is saved
checkpoint_path = "checkpoint/cp.ckpt"

#CLI args
parser = argparse.ArgumentParser()

parser.add_argument(
    "-t", "--train",
    help="Train and fit the Network.",
    action="store_true"
)

parser.add_argument(
    "-p", "--predict",
    type=str,
    help="Predict an image class.",
    action="store"
)

args = parser.parse_args()


def predict(img_path):

    class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    
    my_conv_model = conv_model.ConvModel()
    
    print("Loading NN weights from checkpoint...")
    my_conv_model.model.load_weights(checkpoint_path)
    print("Done.")
    
    np_img = load_image(img_path)
    
    print("Running prediction...")
    predictions= my_conv_model.model.predict(np_img)
    print(predictions)
    score = tf.nn.softmax(predictions[0])
    print(score)
    
    print(
        "==> This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


def load_image(img_path):
    print("Loading img...")
    np_image = Image.open(img_path)
    np_image = np.array(np_image).astype('uint8')/255
    np_image = transform.resize(np_image, (160, 160, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

    
def train_network():
    my_conv_model = conv_model.ConvModel()
    dl = data_loader.MyDataLoader(datapath=datapath, batchsize=batch_size)
    dl.load()
    my_trainer = trainer.ConvTrainer(my_conv_model.model, dl.train_ds, dl.valid_ds) 
    history = my_trainer.do_training_cycle(epochs)
    return history


if args.train:
    train_network()

if args.predict:
    predict(args.predict)
