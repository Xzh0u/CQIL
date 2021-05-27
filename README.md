# CQIL
## Environment Preparation
```bash
conda create -n cqil python=3.7
conda activate cqil
pip install -r requirements.txt

```
## Dataset
There is a data demo in `data/example`. `config.py` defines the dataset path.
To preprocess the data, you should run
```bash
python pipeline.py
# or contains detailed data path
python pipeline.py --data_dir './data/example/' --train_file "train.data_origin.json" --valid_file "valid.data_origin.json" --eval_file "eval.json" > log/pipeline.log
```
Input: `train.data.json` and `valid.data.json`

## Train & Evaluate
  To train our model:
   ```bash
   python main.py --mode train 
   # or contains detailed data path
   python main.py --mode train --data_dir './data/example/' --train_file "train.data_origin.json" --valid_file "valid.data_origin.json" --eval_file "eval.json"
   ```   
   
  To evaluate our model:
   ```bash
   python main.py --mode eval
   ``` 
