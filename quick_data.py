import argparse
import os

from utils.data import (mnist_tfr, svhn_tfr, cifar10_tfr, cifar100_tfr,
                        tfsc_tfr, download_unpack, convert_raw_npy, lpd5_tfr,
                        chair_tfr)


parser = argparse.ArgumentParser("Despite the name, this might take a while.")
parser.add_argument("dir",
                    help="Directory to download and process the data to. Will "
                         "be created if non-existent.")
parser.add_argument("datasets",
                    help="Which datasets to download/process. One letter "
                         "encodes one set.")
args = parser.parse_args()
sets = set(args.datasets)
rawdir = os.path.join(args.dir, "raw")
tfrdir = os.path.join(args.dir, "tfrs")

if not os.path.isdir(rawdir):
    os.makedirs(rawdir)
if not os.path.isdir(tfrdir):
    os.makedirs(tfrdir)

if "m" in sets:
    mnist_dir = os.path.join(rawdir, "mnist")
    if not os.path.isdir(mnist_dir):
        os.mkdir(mnist_dir)
    download_unpack("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                    mnist_dir)
    download_unpack("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                    mnist_dir)
    download_unpack("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                    mnist_dir)
    download_unpack("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                    mnist_dir)
    convert_raw_npy(mnist_dir)
    mnist_tfr(mnist_dir, os.path.join(tfrdir, "mnist"))

if "f" in sets:
    fashion_dir = os.path.join(rawdir, "fashion")
    if not os.path.isdir(fashion_dir):
        os.mkdir(fashion_dir)
    download_unpack("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
                    fashion_dir)
    download_unpack("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
                    fashion_dir)
    download_unpack("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
                    fashion_dir)
    download_unpack("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
                    fashion_dir)
    convert_raw_npy(fashion_dir)
    mnist_tfr(fashion_dir, os.path.join(tfrdir, "fashion"))

if "C" in sets:
    download_unpack("http://www.cs.utoronto.ca/~kriz/cifar-100-python.tar.gz",
                    rawdir)
    cifar100_tfr(os.path.join(rawdir, "cifar-100-python"),
                 os.path.join(tfrdir, "cifar100"))
if "c" in sets:
    download_unpack("http://www.cs.utoronto.ca/~kriz/cifar-10-python.tar.gz",
                    rawdir)
    cifar10_tfr(os.path.join(rawdir, "cifar-10-batches-py"),
                os.path.join(tfrdir, "cifar10"))
if "s" in sets:
    svhndir = os.path.join(rawdir, "svhn")
    if not os.path.isdir(svhndir):
        os.mkdir(svhndir)
    download_unpack("http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                    svhndir)
    download_unpack("http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                    svhndir)
    download_unpack("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                    svhndir)
    svhn_tfr(svhndir, os.path.join(tfrdir, "svhn"))

if "S" in sets:
    scdir = os.path.join(rawdir, "speech_commands_v0.02")
    if not os.path.isdir(scdir):
        os.mkdir(scdir)
    download_unpack("http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                    scdir)
    tfsc_tfr(scdir, os.path.join(tfrdir, "tfsc"))

if "l" in sets:
    lpddir = os.path.join(rawdir, "lpd_5")
    if not os.path.isdir(lpddir):
        raise ValueError("Downloading LPD-5 is not supported. Please download "
                         "and unpack it manually into {}/lpd_5.".format(rawdir))
    lpd5_tfr(lpddir, os.path.join(tfrdir, "lpd5"), beats_per_datum=16,
             downsample=4)

if "h" in sets:
    chairdir = os.path.join(rawdir, "rendered_chairs")
    if not os.path.isdir(chairdir):
        os.mkdir(chairdir)
    download_unpack("https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar",
                    chairdir)
    chair_tfr(os.path.join(chairdir, "rendered_chairs"), os.path.join(tfrdir, "chairs"))
