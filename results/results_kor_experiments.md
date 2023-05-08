# 2차 Baseline 실험 전반 정리(Kor)

목차

---

---

# Baseline Expermental

- Baseline 실험 결과 정리
- [koBART(gogamza/kobart-base-v2)](https://huggingface.co/gogamza/kobart-base-v2) 사용

# Experimental Plan

- [x]  koBART Fine-tuning
- [x]  koBART + Speaker-Aware Fine-tuning
- [x]  koBART + Topic-Aware Fine-tuning
- [x]  koBART + Speaker-Aware + Topic-Aware Fine-tuning

# Experimental Analysis

## Test Raw Data

## Test Raw Data

| Dialogue 1 |
| --- |
| P01: 오호? 토너 아까워서 그렇게 안쓰고살은 1인 걍 적실정도로 칙칙칙 하고 마는데 |
| P02: 마자 막 좋은거 아갑다고 들쓰자나 |
| P01: 응응 |
| P02: 근데 저렴한건 막써 팍팍팍 해서 저렵한게 더 효과조아보임대 |

| Dialogue 2 |
| --- |
| P01: 나 요즘 장활동 개미친이야 9시 눈떠서 운동가기 전 45분동안 세번 똥 싸 |
| P02: 아니 먹은것도 없는디 왤케잘싸..? ㅋㅋㅋ더먹는 나보다 잘싸는듯,, 운동하면 원래 장활동 활발해지지않나 ㅋㅋㅋ 그래서그런가 |
| P01: 모르겠엌ㅋㅋㅋ 진짜 먹는양에 비해 똥 왱케 잘싸짘ㅋㅋ3번 다 어ㅉㅣ됬든 쾌변함 |
| P02: ㅋㅋㅋ좋은거지 머 살빠질듯 응아 무게빠질테니!! |
| P01: ㅋㅋㅋㅋㅋㅋ그건 살이 아니자나요 그냥무게가 빠지는거자나욬ㅋㅋ |
| P02: 무게가 주는것 = 행복 갠춘 똥은 계속 쌀거니까 늘진않겟지 무게가 |

| Dialogue 3 |
| --- |
| P01: 토여일에 약속없댓나? |
| P02: 토욜에 약속업지 #@이름#는 잇고 #@이름# 초딩친구들이랑 놀잖아 |
| P01: 잘알고있군 그럼 금요일밤에 #@이름#랑 새벽까지 놀래? 한우양도먹고 심야로 기생충도 먹고 24시카페도가고 |

| Dialogue 4 |
| --- |
| P01: 승우 센터 못해서 나 지금 기분 너무 안 좋네 제일 가생이에 서있네 어쩌지 빡치네 |
| P02: 아니 진전해 진정해 분량 없어? |
| P01: 김요한만 와...... 가 미쳤네 한숭우... 쟨 끼가.... 몬스타엑스 감임 한승우 사랑해.. |

| Dialogue 5 |
| --- |
| P01: 나오늘 손님이준 에그타르트 세개 처묵했으 |
| P02: 에그타르트 맛있었겠다... 진짜 맛있어.. |
| P01: 존맛이더라 전문점건 다르군 했어... |

---

## koBART Fine-tuning

- 기존 Pretrained Model인 koBART를 Dialogue Summarization에 맞게 Fine-tuning

### Results

![https://user-images.githubusercontent.com/32642002/230756721-2bc6106f-2a1e-4634-887c-3269b04d2a3d.png](https://user-images.githubusercontent.com/32642002/230756721-2bc6106f-2a1e-4634-887c-3269b04d2a3d.png)

| Rouge 1 | Rouge 2 | Rouge L | Rouge Lsum |
| --- | --- | --- | --- |
| 0.3134001998905277 | 0.1782477263484106 | 0.27797291413303815 | 0.2779665165772779 |

| Generated Summary | Label Summary |
| --- | --- |
| 토너가 아까워서 안 쓰고 살은 적실 정도로 칙칙하고 좋은 거 아갑다고 들쓰는데 저렴한 건 팍팍팍 해서 저렵한 게 더 효과적일 것 같다고 한다. | 토너가 아까워서 아껴서 사용했다고 하니 좋은 건 아깝다고 덜 쓰지만 저렴한 건 막 써서 저렴한 게 더 효과가 좋은 것 같다고 말한다. |
| 요즘 장활동이 좋아서 운동하기 전 45분 동안 세 번 똥을 싸서 살이 빠질 것 같다고 이야기하고 있다. 똥은 계속 쌀 거니까 늘지는 않을 것 같지만 살이 빠지는 건 아니라고 한다. 똥이 계속 쌀 거라고 한다. 살이 빠지니까 똥도 늘진 않을 거 같다. | 요즘 운동을 해서 그런지 9시 눈 떠서 운동 가기 전 45분 동안 세 번이나 쾌변을 본다. |
| 금요일에에 약속이 없어서 초딩 친구들이랑 새벽까지 놀기로 했다. 한우와 기생충도 먹고 24시 카페에 간다. 금요일 밤에 약속이 없는지 물어본다. 한우는 있고 초딩 친구들은 있다고 한다. 한우는 없고 심야에 카페를 간다고 한다. 24시 약속이 있다. 한우는 없다. 금요일 밤에 새벽까지 같이 놀자고 한다. 한우를 먹고 심야 | 금요일 밤에 만나서 새벽까지 놀기로 했다. |
| 승우 센터를 못해서 기분이 안 좋다며 김요한만 오라고 하고 한승우를 사랑한다고 한다. | 한승우가 센터를 못해서 기분이 안 좋다. |
| 오늘 손님이 준 에그타르트 세 개를 처묵하게 먹었는데 맛있었다. 전문점 건 다르지만 맛있었다. | 손님이 준 에그타르트를 세 개 먹었는데 너무 맛있어서 전문점 건 다르다고 생각했다. |

## koBART + Speaker-Aware Fine-tuning

### Method

- Speaker-Aware Function에 Speaker Token들의 Encoder Representation을 전달
- Positivie Sample : 기준 Speaker와 같은 Speaker Token들의 Encoder Representation
- Negativie Sample : 기준 Speaker와 다른 Speaker Token들의 Encoder Representation
- Contrastive Learning을 통해 Positive Samples와 Negative Samples의 Representation을 각각 비슷해지도록 학습

### Results

![https://user-images.githubusercontent.com/32642002/230756796-f89599c9-7d9e-4f81-a35f-a43b9fb75ebe.png](https://user-images.githubusercontent.com/32642002/230756796-f89599c9-7d9e-4f81-a35f-a43b9fb75ebe.png)

| Rouge 1 | Rouge 2 | Rouge L | Rouge Lsum |
| --- | --- | --- | --- |
| 0.3226975922434263 | 0.18266904865135086 | 0.28801242656619996 | 0.28788674056581637 |

| Generated Summary | Label Summary |
| --- | --- |
| 토너가 아까워서 안 쓰고 살은 적실 정도로 칙칙칙하고 좋은 거 아갑다고 들쓰는데 저렴한 건 안 쓰고 팍팍팍 해서 저렵한 게 더 효과적이라고 한다. | 토너가 아까워서 아껴서 사용했다고 하니 좋은 건 아깝다고 덜 쓰지만 저렴한 건 막 써서 저렴한 게 더 효과가 좋은 것 같다고 말한다. |
| 요즘 장활동이 좋아서 운동하기 전에 45분 동안 세 번 똥을 싸고 똥이 잘 싸는 것 같다. 똥은 계속 쌀 거니까 늘진 않을 것 같지만 살이 빠질 것 같다고 이야기한다. 똥도 잘 싸고 살이 빠지는 것 같고 행복하다. 씹는 것도 행복이다. 씹으면 살이 빠진다고 한다. 똥 | 요즘 운동을 해서 그런지 9시 눈 떠서 운동 가기 전 45분 동안 세 번이나 쾌변을 본다. |
| 금요일에에 약속이 없냐고 물으니 초딩 친구들이랑 놀니까 새벽까지 놀자고 한다. 24시 카페에 가서 기생충도 먹고 24시도 카페에 가자고 한다. | 금요일 밤에 만나서 새벽까지 놀기로 했다. |
| 승우 센터를 못해서 기분이 안 좋다며 김요한만 오라고 한다. 한승우는 끼가 있고 몬스타엑스(몬스타) 감이라고 한다. | 한승우가 센터를 못해서 기분이 안 좋다. |
| 오늘 손님이 준 에그타르트 세 개를 처묵하게 먹었는데 맛있었다. 전문점 건 다르지만 맛있었다. | 손님이 준 에그타르트를 세 개 먹었는데 너무 맛있어서 전문점 건 다르다고 생각했다. |

## koBART + Topic-Aware Fine-tuning

### Method

- Encoder로부터 나온 Encoder Representation 중에서 utterance token들의 Encoder Representation Mean Pooling 값을 계산
- Topic-Aware Function에 Utterance Tokens의 Encoder Representation Mean Pooling을 전달
- Kmeans Algorithm으로 각 Utterance Token들의 Topic을 Clustering으로 예측(# of Cluster : 2)
- 각 Cluster들의 Centeroid와 Centeroid로부터 가장 가까이 있는 다른 Cluster의 표현과의 L2 Distance를 Positive와 Negative로 추출
- 예) Bench(기준)가 Clsuter 0에 속할때,
- Min(Cluster 0번의 Centeroid <-> Cluster 1번의 Utterance 표현들) = Negative Sample
- Min(Cluster 1번의 Centeroid <-> Cluster 0번의 Utterance 표현들) = Positive Sample
- Contrastive Learning을 통해 Positive Samples와 Negative Samples의 Representation을 각각 비슷해지도록 학습

### Results

![https://user-images.githubusercontent.com/32642002/230780485-9492aaf1-dfc7-46c9-989a-0ef7c0145fa9.png](https://user-images.githubusercontent.com/32642002/230780485-9492aaf1-dfc7-46c9-989a-0ef7c0145fa9.png)

| Rouge 1 | Rouge 2 | Rouge L | Rouge Lsum |
| --- | --- | --- | --- |
| 0.32003445288683796 | 0.18135920225142368 | 0.28603414195900134 | 0.28600778235137236 |

| Generated Summary | Label Summary |
| --- | --- |
| 토너 아까워서 안 쓰고 살은 적실 정도로 칙칙칙하고 좋은 거 아갑다고 들쓰는데 저렴한 건 팍팍팍 해서 저렵한 게 더 효과적일 것 같다. | 토너가 아까워서 아껴서 사용했다고 하니 좋은 건 아깝다고 덜 쓰지만 저렴한 건 막 써서 저렴한 게 더 효과가 좋은 것 같다고 말한다. |
| 요즘 장활동이 좋아서 운동하기 전에 45분 동안 세 번 똥을 싸고 있다. 똥은 계속 쌀 거니까 늘진 않을 것 같지만 살이 빠질 것 같다. | 요즘 운동을 해서 그런지 9시 눈 떠서 운동 가기 전 45분 동안 세 번이나 쾌변을 본다. |
| 금요일에에 약속이 없냐고 물으니 초딩 친구들이랑 놀고 24시 카페에 가서 기생충도 먹고 24시도 카페에 가자고 한다. | 금요일 밤에 만나서 새벽까지 놀기로 했다. |
| 승우 센터를 못해서 기분이 안 좋다며 김요한만 온다고 하고 한승우를 사랑한다고 한다. 몬스타엑스 감이라고 한다. | 한승우가 센터를 못해서 기분이 안 좋다. |
| 오늘 손님이 준 에그타르트 세 개가 맛있었다고 하자 맛있다고 한다. 전문점건 다르다고 한다. 맛있다고 하자 전문점 건 다르다고 말한다. | 손님이 준 에그타르트를 세 개 먹었는데 너무 맛있어서 전문점 건 다르다고 생각했다. |

## koBART + Multi-Aware(Speaker-Aware + Topic-Aware) Fine-tuning

### Method

- 위에서 실험한 Speaker-Aware와 Topic-Aware를 함께 진행한 경우
- Speaker-Aware의 Contrastive

### Results

![https://user-images.githubusercontent.com/32642002/230848840-23822b83-d55c-4273-b275-a78ddfe8c0f8.png](https://user-images.githubusercontent.com/32642002/230848840-23822b83-d55c-4273-b275-a78ddfe8c0f8.png)

| Rouge 1 | Rouge 2 | Rouge L | Rouge Lsum |
| --- | --- | --- | --- |
| 0.32304120182091534 | 0.18292308583671243 | 0.28819565682547654 | 0.2879850996973683 |

| Generated Summary | Label Summary |
| --- | --- |
| 토너가 아까워서 안 쓰고 살은 적실 정도로 칙칙칙해서 좋은 거 아갑다고 들쓰는데 저렴한 건 팍팍팍 해서 저렵한 게 더 효과적이라고 한다. | 토너가 아까워서 아껴서 사용했다고 하니 좋은 건 아깝다고 덜 쓰지만 저렴한 건 막 써서 저렴한 게 더 효과가 좋은 것 같다고 말한다. |
| 요즘 장활동이 좋아서 운동하기 전에 45분 동안 세 번 똥을 싸고 있다. 똥은 계속 쌀 거니까 늘진 않을 것 같지만 살이 빠질 것 같다. | 요즘 운동을 해서 그런지 9시 눈 떠서 운동 가기 전 45분 동안 세 번이나 쾌변을 본다. |
| 금요일에에 약속이 없냐고 하자 초딩 친구들이랑 놀고 있다고 한다. 금요일 밤에 한우 양도 먹고 심야로 기생충도 먹고 24시 카페에 간다고 한다. | 금요일 밤에 만나서 새벽까지 놀기로 했다. |
| 승우 센터를 못해서 기분이 안 좋다며 김요한만 오라고 하고 한승우를 사랑한다고 한다. 몬스타엑스(몬스타엑스) 감이라고 한다. | 한승우가 센터를 못해서 기분이 안 좋다. |
| 오늘 손님이 준 에그타르트 세 개를 처묵하게 먹었는데 맛있었다. 전문점 건 다르지만 맛있었다. | 손님이 준 에그타르트를 세 개 먹었는데 너무 맛있어서 전문점 건 다르다고 생각했다. |

---

### 실험 변경 사항(Update Code) 001

- 2023.04.08
- bartmodel.py > speaker_aware Function
- Speaker-Aware의 Speaker Token Sampling 방식을 잘못 이해하여 다시 수정
- 기존 : Speaker Tokens을 전부 Softmax 취한 후, max sampling으로 positive와 negative 값을 추출 -> 이 값을 Contrastive Learning 식에 적용
- 수정 : Speaker Tokens을 positive와 negative로 나누고 그 중 random sampling한 값을 Softmax 취한 후, positive와 negative 값으로 Contrastive Learning 식에 적용

---

### 실험 변경 사항(Update Code) 002

- 2023.04.09
- bartmodel.py > speaker_aware Function
- Speaker-Aware의 Similarity를 L2 distance(거리) + Cosine Similarity(방향)의 값으로 수정
- 거리와 방향 둘 다 고려하여야 좀 더 정확하게 표현을 가깝게 조정할 수 있다고 생각하여 수정
- L2 distance는 작은 값일 수록 좋고, Cosine Similarity는 큰 값일 수록 좋으므로, Cosine Similarity에서 L2 distance 값을 빼준 값을 Similarity로 두고 학습 수행
- 기존 : Speaker-Aware의 Similarity로 Cosine Similarity(방향)를 사용
- 수정 : Speaker-Aware의 Similarity로 L2 distance(거리) + Cosine Similarity(방향)를 사용
- 결과 : L2와 Similarity가 큰 차이 없어서 일반적으로 사용한다는 L2 Distance로 결정 -> baseline보다 성능 1 상승

---

### 실험 변경 사항(Update Code) 003

- 2023.04.10
- bartmodel.py > topic_aware Function
- Topic-Aware의 Utterance Token Sampling 방식을 잘못 이해하여 다시 수정
- 기존 : Utterance Tokens의 Mean Pooling 값을 min, max를 이용하여 positive와 negative 값을 추출 -> 이 값을 Contrastive Learning 식에 적용
- 수정 : Utterance Tokens의 Mean Pooling 값을 KMeans로 Clustering한 후, 각 Cluster의 Centroid와 반대편의 Cluster에 속한 Utterance Token간 L2 Distance를 구하고 그 중 min값을 positive와 negative로 결정 -> Softmax 취한 후, Contrastive Learning 식에 적용
- 결과 : ROUGE1 기준으로 이전보다 0.1 저하는 되었지만 운 좋게 데이터셋이 적절하게 나뉜것으로 판단 -> 수정된 방식이 이전 방식보다 논리적으로 완성도 있는 방식이므로 수정 방식을 사용

---