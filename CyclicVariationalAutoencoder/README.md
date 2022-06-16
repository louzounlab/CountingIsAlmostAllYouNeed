The code that trains a new TCR autoencoder.

You may edit the parameters used in the training in the parameters.json file. The parameters file also contains the path to your data.

In order to run the training code write in the commandline:

python3 cyclic_vae.py

make sure 'parameters.json' is in the same directory as the code.
The output of the code is an autoencoder dict that contains the weights and other parameter on the autoencoder. This dict is used to train the other ML models.
