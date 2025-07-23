# Align-LoRA

### Environment

```sh
conda create -n Align-LoRA python=3.10
conda activate Align-LoRA
pip install -r requirements.txt
```

### Project Structure

```sh
data_load.py # download the data
train.py # train with R-LoRA
test_fc # function for the evaluation
test.py # evluation
eval_bbh.py # Big Bench Hard
```

### Train

```shell
python train.py
```

### Test

```shel
python eval_bbh.py
```

### 
