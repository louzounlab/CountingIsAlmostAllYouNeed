The code runs both the counting model and attTCR on a given dataset. 
To run the code write in the commadline:

python3 run.py --data <data_path> --ae <autoencoder_dictionary> --model <model_type>

data_path - The path to your dataset. You may use the preprocessed datset provided in the repository with the split used in the paper. 
            You also may create you own dataset, as long as it contains the 'Train' and 'Test' directories and contains the necessery columns.
            
autoencoder_dictionary - The path to your pretrained autoencoder dict. You may use the provided, pretrained autoencoder dict, or you may train one yourself using the code provided. 

model_type - The type of model you with to train. Recieves either 'count' or 'attTCR'
            
