# Generate Chairs paper code
Generating 3D Chairs in Tensorflow2 using CNNs

Code is written according to [this paper](https://lmb.informatik.uni-freiburg.de/Publications/2015/DB15/Generate_Chairs_arxiv.pdf) using [this dataset](https://www.di.ens.fr/willow/research/seeing3Dchairs/). Two main differences are:

__1-__ Segmantation network (in Fig.2 network shown below which have Euclidean error x1) is not used.

__2-__ Embeddings are used to encode chair id's rather than one-hot encodings. Embeddings are generally preferred due to their statistical efficiency.

You can use quick_data.py to download the dataset. Running this script with the h string for the dataset argument will process the chairs data to a TFRecords file. Then, you can use the parse_chairs function from utils/data to read the images.

Resize training images as like as you want using __parse_chairs(x, resize=64)__ function.

There is two different jupyter notebooks: training and experiment. In the experiment notebook, model structure is re-created manually and proper weights has been loaded for experiments such as morphing between images and fixing other parameters like rotation and elevation.

# Experiment outputs

Morphing between two chair (fixed elevation and rotation)

![Morphing Image](https://i.imgur.com/qWO7iyS.png)

Rotating a chair for a specific chair

![Rotating chair](https://i.imgur.com/meTBLxN.png)
