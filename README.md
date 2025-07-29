# Analysis Code for Learned Representations of Biofilm Lifecycle Videos of _Vibrio cholerae_
## 1 - Embeddings Generation
### 1.1 - Dataloading
1.1.1 - Modifying the config file

Example config for loading mutant training data (config/train_config.json)
```
{
    "data_loc": "../../data",
    "dataloader_loc": "../../dataloaders",
    "weights_loc": "../../weights",
    "mutants_or_transposons": "mutants",
    "dataloader": "mutant_train_dataset.pth",
    "weights": "final_weights.pth"
}
```
Example config for loading mutant evaluation data (config/eval_config.json)
```
{
    "data_loc": "../../data",
    "dataloader_loc": "../../dataloaders",
    "weights_loc": "../../weights",
    "mutants_or_transposons": "mutants",
    "dataloader": "mutant_eval_dataset.pth",
    "weights": "final_weights.pth"
}
```
1.1.2 - Loading the data 

- Dataloader saved in ./dataloaders

```
cd src/single_proc
python3 load_data.py
```
## 2 - Training 

- Skip this step if only interested in zero-shot evaluation, weights saved in ./weights

### 2.1 - Multi-process training (recommended)

- For use on computing cluster setup, adjust BATCH_SIZE as needed
- Submits SLURM job using ```encode``` script
- Adjust requested resources in ```encode``` as needed
```
cd src/multi_proc
sbatch encode
```
### 2.2 - Single-process training 

- For use on single GPU/CPU, only when VRAM/RAM is sufficient to load entire dataset, adjust BATCH_SIZE as needed

```
cd src/single_proc
python3 train.py
```
## 3 - Evaluation 

- Embeddings and labels saved in ./processed_data
- Embeddings saved in ```embeddings.npy```
- Labels saved in ```labels.npy```
- Labels for mutants are strain identifiers
- Labels for transposons are plate and well identifiers

- If weights file in config does not exist in ./weights, will use default pretrained weights from OpenAI CLIP model
- Ensure that enough memory is available to load entire dataset
    
```
cd src/single_proc
python3 eval.py
```
## 4 - Analysis 
### 4.1 - Plotting
### 4.2 - Classification
### 4.3 - Outlier Detection
### 4.4 - Bioinformatics (Functional Enrichment  Analysis and Clustering)




