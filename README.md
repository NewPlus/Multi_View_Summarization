# Multi_View_Summarization
- 논문 준비 기록 git
- 2023

## Process
### Baseline
- HuggingFace Transformers 라이브러리의 BartModel 클래스 불러와서 Encoder Output만 코딩할 수 있도록 코드 가져오기(2023.03.30)
    - BartModel만 수정하고 나머지는 원래 Transformers 라이브러리 그대로 똑같이 사용
- Speaker-Aware()와 Topic-Aware() 멤버 함수로 추가해두고 bart_test.py로 테스트 세팅(2023.03.31)
- Speaker-Aware()에서 Speaker P01을 기준으로 나머지 Speaker Token의 Representation과의 Similarity를 Contrastive Learning(2023.04.01)
- Topic-Aware()에서 Utterance의 표현들 중 padding을 제외한 표현들을 K-Means 및 DBSCAN으로 Clustering하여 Topic 추출(2023.04.02)
- Topic-Aware()에서 Clustering 시각화(PCA 이용)(2023.04.02)
- 잘못 이해한 내용 수정 -> 한 Dialogue를 통째로 넣어서 Speaker-Aware와 Topic-Aware진행하는 것으로 수정(2023.04.03 ~ 2023.04.04)
- Seq2SeqModelOutput, Seq2SeqLMOutput 상속 받아서 CustomSeq2SeqModelOutput, CustomSeq2SeqLMOutput DataClass 추가(2023.04.06)
- BartModel, BartForConditionalGeneration에 special token의 index 전달(2023.04.06)
- BartTrainer를 Customize(Preprocessing, Filtering 등등) (2023.04.06)
- 기존 Bart로 Fine-tuning 후, 실험 진행(1차) (2023.04.07)
- bart_trainer 오류 수정 (2023.04.08)
- Speaker-Aware 잘못 이해한 부분 수정 (2023.04.08)
- Speaker-Aware 1차 실험 (2023.04.09) : koBART보다 ROUGE1 기준 0.92 상승
- Topic-Aware 실험 코드 (2023.04.09)
- Topic-Aware 1차 실험 (2023.04.09) : koBART보다 ROUGE1 기준 0.8 상승

### Data_PreProcessing
- SAMSum Dataset으로 Test준비(2023.03.31)
- SAMSum 대신 한국어 데이터셋으로 전환(2023.04.01)
- Speaker Token들을 Random하게 샘플링하여 교체하는 전처리 작업(2023.04.01)
- Speaker 전처리 미적용 수정(2023.04.02)
- 잘못 이해한 내용 수정 -> 한 Dialogue를 통째로 넣어서 Speaker-Aware와 Topic-Aware진행하는 것으로 수정(2023.04.03 ~ 2023.04.04)
- 전처리 부분 Class화 -> DataPreprocess Class(data_preprocess.py)로 객체화(2023.04.04)
- 전치리 부분에서 랜덤 샘플링 중 이상 수정(2023.04.05)
