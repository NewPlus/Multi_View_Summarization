# Multi_View_Summarization
- 화자 및 주제 대조 학습을 통한 대화 요약 성능 향상
- 2023.05.02
- PyTorch, Huggingface

## Directory
```
.
|-- README.md
|-- bart_trainer.py
|-- bartmodel.py
|-- data
|   |-- dialogue.json
|   |-- test.csv
|   |-- train.csv
|   `-- valid.csv
`-- requirements.txt
```

# Tutorial
## Setting
- docker container
```
docker run -it --name multi_view_yh -p [port num]:[port num] -v [current root] --shm-size=[memory size] --gpus '"device=[gpu num]"' pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
```
- directory
```
cd bart_customize/
```
- pip install requirements
```
pip install -r requirements.txt
```

## Experiments
- Baseline Experiments
- Speaker-Aware Experiments
- Topic-Aware Experiments
- Multi-Aware Experiments

### Baseline Experiments
```
CUDA_VISIBLE_DEVICES=0 python bart_trainer.py \
--model_name "facebook/bart-large" \
--data_name "samsum" \
--ctr_mode 0 \
--lamda 0.08 \
--batch_size 8 \
--set_seed 100 \
--cluster_mode 1 \
--output_dir "/root/bart_customize/test_save"
```

### Speaker-Aware Experiments
```
CUDA_VISIBLE_DEVICES=0 python bart_trainer.py \
--model_name "facebook/bart-large" \
--data_name "samsum" \
--ctr_mode 1 \
--lamda 0.08 \
--batch_size 8 \
--set_seed 100 \
--cluster_mode 1 \
--output_dir "/root/bart_customize/test_save"
```

### Topic-Aware Experiments
```
CUDA_VISIBLE_DEVICES=0 python bart_trainer.py \
--model_name "facebook/bart-large" \
--data_name "samsum" \
--ctr_mode 2 \
--lamda 0.08 \
--batch_size 8 \
--set_seed 100 \
--cluster_mode 1 \
--output_dir "/root/bart_customize/test_save"
```

### Multi-Aware Experiments
```
CUDA_VISIBLE_DEVICES=0 python bart_trainer.py \
--model_name "facebook/bart-large" \
--data_name "samsum" \
--ctr_mode 3 \
--lamda 0.08 \
--batch_size 8 \
--set_seed 100 \
--cluster_mode 1 \
--output_dir "/root/bart_customize/test_save"
```

# Process
