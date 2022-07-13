# CountingIsAlmostAllYouNeed
Code for the models described in the paper "Counting is Almost All You Need". 
This repeository contains code for the counting model, attTCR, and gTCR, as well as preprocessing code for the Emerson dataset, and the dataset that was used in the paper, after preprocessing.


To run the models run the command:

python run.py --config <config_file> -- model <model_type> --params <param_file>

**model_type** : The model you wish to run. Can be either 'gTCR', 'attTCR' or 'count'.
**config_file** : The data config json file. See example file.
**param_file** : The parameter json file. The defualt parameters used in the paper are provided in the example files, but you may edit them.
