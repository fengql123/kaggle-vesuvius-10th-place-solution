# Kaggle Vesuvius Challenge - Ink Detection 10th Place Solution

### Hardware
gpu: RTX4090
cpu: 13th Gen Intel® Core™ i9-13900K × 32
ram: 32gb

### Environment
Ubuntu 22.04.2 LTS 64 bit

### How to reproduce
1. Download data from kaggle into the /data folder
2. Split fragment 2 along the y-axis(height) into 3 equal parts, store the top, middle and bottom in data/train/2, data/train/3 and data/train/4
3. Store fragment 3 as data/train/5
4. Run the following in the terminal:
```
python sample.py
python pretrain.py
python main.py
```
4. The trained weights will be stored in weights/trained
5. Upload the weights to the [notebook](https://www.kaggle.com/code/fengqilong/vesuvius-inference), set the dataset name as "trained"
```
├── trained
│   ├── weight fold 1
│   ├── weight fold 2
│   ├── weight fold 3
│   ├── weight fold 4
```
6. Change `CFG.exp_name` to `"trained"`
7. Submit to competition
