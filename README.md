# Kaggle Vesuvius Challenge - Ink Detection 10th Place Solution

### Hardware
gpu: RTX4090
cpu: 13th Gen Intel® Core™ i9-13900K × 32
ram: 32gb

### Environment
Ubuntu 22.04.2 LTS 64 bit

### How to reproduce
1. Download data from [here](https://www.kaggle.com/datasets/fengqilong/vesuvius-split) and extract the folders within the downloaded folder into the /data folder, it should follow a structure like this:
```
├── data
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── 4
│   ├── 5
```
2. Run the following in the terminal:
```
python sample.py
python pretrain.py
python main.py
```
3. The trained weights will be stored in weights/trained
4. Upload the weights to the [notebook](https://www.kaggle.com/code/fengqilong/vesuvius-inference), set the dataset name as "trained"
```
├── trained
│   ├── weight fold 1
│   ├── weight fold 2
│   ├── weight fold 3
│   ├── weight fold 4
```
5. Change `CFG.exp_name` and `TH` to `"trained"` and the cv threshold shown in output respectively.
6. Submit to competition
