# Improving Dialogue Summarization with Speaker-aware and Topic-aware Contrastive Learning
- Authors: Yonghwan Lee, Jinhyeong Lim, Hyeon-Je Song
- 대화문(Dialogue)의 화자(Speaker)와 주제(Topic) 정보를 동시에 고려하는 다중 관점 대조 학습(Multi-Aware Contrastive Learning)을 추가하여 대화 요약 모델(Dialogue Summarization Model)의 성능을 향상
- 한국정보과학회 제출 : [KCC2023](https://www.kiise.or.kr/conference/kcc/2023/)
- 2023.05.01
- [PyTorch](https://pytorch.org/), [Huggingface](https://huggingface.co/)

## Model Architecture
![architecture](experimental_img/model_architecture.png)
- BART 모델을 Summarization Task로 Fine-tuning
- Encoder의 Representation을 Contrastive Learning으로 조정하는 Auxiliary task를 추가
    - Speaker-Aware : Dialogue의 Speaker Token Representation들을 Speaker가 같으면 Representation이 유사하도록, 다르면 Representation을 조정
    - Topic-Aware : Dialogue의 Utterance Token Representation들을 K-Means Algorithms으로 Clustering ->  Cluster가 같으면 Representation이 유사하도록, 다르면 Representation을 조정
    - Multi-Aware : 위 Speaker-Aware와 Topic-Aware를 모두 진행하여 Encoder Representation들을 조정

## Directory
```
.
|-- PROCESS.md
|-- README.md
|-- bart_trainer.py
|-- bartmodel.py
|-- data
|   |-- dialogue.json
|   |-- test.csv
|   |-- train.csv
|   `-- valid.csv
|-- requirements.txt
`-- results
    |-- results_eng_experiments.md
    `-- results_kor_experiments.md
```

# Tutorial
## Installation
- pip install requirements
```
pip install -r requirements.txt
```

## Experiments
- Baseline Experiments
    - 기존 BART 모델을 Fine-tuning
    - ctr_mode = "baseline"
- Speaker-Aware Experiments
    - BART + Speaker-Aware
    - ctr_mode = "speaker"
- Topic-Aware Experiments
    - BART + Topic-Aware
    - ctr_mode = "topic"
- Multi-Aware Experiments
    - BART + Speaker-Aware + Topic-Aware
    - ctr_mode = "multi"

- Example of Baseline
```
CUDA_VISIBLE_DEVICES=0 python bart_trainer.py \
--model_name "facebook/bart-large" \
--data_name "samsum" \
--ctr_mode "baseline" \
--lamda 0.08 \
--batch_size 8 \
--set_seed 100 \
--cluster_mode 1 \
--output_dir "/root/bart_customize/test_save"
```

# Results
