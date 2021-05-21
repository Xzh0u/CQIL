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
```


## Train & Evaluate
  To train our model:
   ```bash
   python main.py --mode train
   ```   
   
  To evaluate our model:
   ```bash
   python main.py --mode eval
   ``` 