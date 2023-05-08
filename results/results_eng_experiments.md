# eng_experiments

# Baseline Expermental - English Dataset

- Baseline 실험 결과 정리
- 영어 데이터셋 실험 결과 정리
- Dataset : [SAMSum](https://huggingface.co/datasets/samsum) 사용
- Model : [BART-Large(facebook/bart-large)](https://huggingface.co/facebook/bart-large) 사용

# Experimental Plan

- [x]  BART Fine-tuning
- [x]  BART + Speaker-Aware Fine-tuning
- [x]  BART + Topic-Aware Fine-tuning
- [x]  BART + Speaker-Aware + Topic-Aware Fine-tuning

# Experimental Analysis

## Test Raw Data

- [SAMSum](https://huggingface.co/datasets/samsum) 사용

| Dialogue 1 |
| Hannah: Hey, do you have Betty's number?
Amanda: Lemme check
Hannah: <file_gif>
Amanda: Sorry, can't find it.
Amanda: Ask Larry
Amanda: He called her last time we were at the park together
Hannah: I don't know him well
Hannah: <file_gif>
Amanda: Don't be shy, he's very nice
Hannah: If you say so..
Hannah: I'd rather you texted him
Amanda: Just text him 🙂
Hannah: Urgh.. Alright
Hannah: Bye
Amanda: Bye bye |

| Dialogue 2 |
| --- |
| Eric: MACHINE!
Rob: That's so gr8!
Eric: I know! And shows how Americans see Russian ;)
Rob: And it's really funny!
Eric: I know! I especially like the train part!
Rob: Hahaha! No one talks to the machine like that!
Eric: Is this his only stand-up?
Rob: Idk. I'll check.
Eric: Sure.
Rob: Turns out no! There are some of his stand-ups on youtube.
Eric: Gr8! I'll watch them now!
Rob: Me too!
Eric: MACHINE!
Rob: MACHINE!
Eric: TTYL?
Rob: Sure :) |

| Dialogue 3 |
| --- |
| Lenny: Babe, can you help me with something?
Bob: Sure, what's up?
Lenny: Which one should I pick?
Bob: Send me photos
Lenny:  <file_photo>
Lenny:  <file_photo>
Lenny:  <file_photo>
Bob: I like the first ones best
Lenny: But I already have purple trousers. Does it make sense to have two pairs?
Bob: I have four black pairs :D :D
Lenny: yeah, but shouldn't I pick a different color?
Bob: what matters is what you'll give you the most outfit options
Lenny: So I guess I'll buy the first or the third pair then
Bob: Pick the best quality then
Lenny: ur right, thx
Bob: no prob :) |

| Dialogue 4 |
| --- |
| Will: hey babe, what do you want for dinner tonight?
Emma:  gah, don't even worry about it tonight
Will: what do you mean? everything ok?
Emma: not really, but it's ok, don't worry about cooking though, I'm not hungry
Will: Well what time will you be home?
Emma: soon, hopefully
Will: you sure? Maybe you want me to pick you up?
Emma: no no it's alright. I'll be home soon, i'll tell you when I get home.
Will: Alright, love you.
Emma: love you too. |

| Dialogue 5 |
| --- |
| Ollie: Hi , are you in Warsaw
Jane: yes, just back! Btw are you free for diner the 19th?
Ollie: nope!
Jane: and the  18th?
Ollie: nope, we have this party and you must be there, remember?
Jane: oh right! i lost my calendar..  thanks for reminding me
Ollie: we have lunch this week?
Jane: with pleasure!
Ollie: friday?
Jane: ok
Jane: what do you mean " we don't have any more whisky!" lol..
Ollie: what!!!
Jane: you just call me and the all thing i heard was that sentence about whisky... what's wrong with you?
Ollie: oh oh... very strange! i have to be carefull may be there is some spy in my mobile! lol
Jane: dont' worry, we'll check on friday.
Ollie: don't forget to bring some sun with you
Jane: I can't wait to be in Morocco..
Ollie: enjoy and see you friday
Jane: sorry Ollie, i'm very busy, i won't have time for lunch  tomorrow, but may be at 6pm after my courses?this trip to Morocco was so nice, but time consuming!
Ollie: ok for tea!
Jane: I'm on my way..
Ollie: tea is ready, did you bring the pastries?
Jane: I already ate them all... see you in a minute
Ollie: ok |

## BART Fine-tuning

- 기존 Pretrained Model인 BART를 Dialogue Summarization에 맞게 Fine-tuning

### Results

![https://user-images.githubusercontent.com/32642002/231643650-6c7d90cd-4b4f-4aac-87c7-f2f122ac1d4c.png](https://user-images.githubusercontent.com/32642002/231643650-6c7d90cd-4b4f-4aac-87c7-f2f122ac1d4c.png)

| Rouge 1 | Rouge 2 | Rouge L | Rouge Lsum |
| --- | --- | --- | --- |
| 0.5302506885049254 | 0.3097367281995699 | 0.46174891712362565 | 0.46216103700216515 |

| Generated Summary | Label Summary |
| --- | --- |
| Amanda can't find Betty's number. Larry called her last time they were at the park. Hannah suggests Amanda to text him. | Hannah needs Betty's number but Amanda doesn't have it. She needs to contact Larry. |
| Eric and Rob are laughing at a Russian stand-up comedian's joke about a machine. | Eric and Rob are going to watch a stand-up on youtube. |
| Lenny will buy the first or the third pair of purple trousers. | Lenny can't decide which trousers to buy. Bob advised Lenny on that topic. Lenny goes with Bob's advice to pick the trousers that are of best quality. |
| Emma will be home soon and will let Will know what she wants for dinner. | Emma will be home soon and she will let Will know. |
| Jane is back from her trip to Morocco. Ollie and Jane will meet for lunch on Friday at 6 pm after her courses. | Jane is in Warsaw. Ollie and Jane has a party. Jane lost her calendar. They will get a lunch this week on Friday. Ollie accidentally called Jane and talked about whisky. Jane cancels lunch. They'll meet for a tea at 6 pm. |

## BART + Speaker-Aware Fine-tuning

### Method

- Speaker-Aware Function에 Speaker Token들의 Encoder Representation을 전달
- Speaker Tokens : Turn에서 '<sep>' 다음 Token부터 ':' 이전 Token까지를 Speaker Token으로 간주하고 Mean Pooling
- Positivie Sample : 기준 Speaker와 같은 Speaker Token들의 Encoder Representation
- Negativie Sample : 기준 Speaker와 다른 Speaker Token들의 Encoder Representation
- Contrastive Learning을 통해 Positive Samples와 Negative Samples의 Representation을 각각 비슷해지도록 학습

### Results

![https://user-images.githubusercontent.com/32642002/234998854-1e4f6d86-884b-4058-8a1f-0cb7a0a7d323.png](https://user-images.githubusercontent.com/32642002/234998854-1e4f6d86-884b-4058-8a1f-0cb7a0a7d323.png)

| Rouge 1 | Rouge 2 | Rouge L | Rouge Lsum |
| --- | --- | --- | --- |
| 0.5305 | 0.3105 | 0.4608 | 0.4608 |

| Generated Summary | Label Summary |
| --- | --- |
| Amanda can't find Betty's number. Larry called her last time they were at the park. Hannah doesn't know Betty. Amanda will text Larry. | Hannah needs Betty's number but Amanda doesn't have it. She needs to contact Larry. |
| Eric and Rob are laughing at a Russian stand-up comedian's joke about a machine. They will watch some of his stand-ups on youtube. | Eric and Rob are going to watch a stand-up on youtube. |
| Lenny will buy the first or the third pair of black trousers. | Lenny can't decide which trousers to buy. Bob advised Lenny on that topic. Lenny goes with Bob's advice to pick the trousers that are of best quality. |
| Emma doesn't want to cook dinner tonight. She will tell Will when she gets home. | Emma will be home soon and she will let Will know. |
| Jane is back from Morocco. Ollie has a party on the 18th. They will have lunch on Friday at 6 pm. Jane has already eaten all the pastries. | Jane is in Warsaw. Ollie and Jane has a party. Jane lost her calendar. They will get a lunch this week on Friday. Ollie accidentally called Jane and talked about whisky. Jane cancels lunch. They'll meet for a tea at 6 pm. |

## BART + Topic-Aware Fine-tuning

### Method

- Encoder로부터 나온 Encoder Representation 중에서 utterance token들의 Encoder Representation Mean Pooling 값을 계산
- Topic-Aware Function에 Utterance Tokens의 Encoder Representation Mean Pooling을 전달
- Kmeans Algorithm으로 각 Utterance Token들의 Topic을 Clustering으로 예측(# of Cluster : 2)
- 각 Cluster들의 Centeroid와 같은 Cluster의 표현과의 L2 Distance는 Positivie, 다른 Cluster의 표현과의 L2 Distance는 Negative로 추출
- Contrastive Learning을 통해 Positive Samples와 Negative Samples의 Representation을 각각 비슷해지도록 학습

### Results

![https://user-images.githubusercontent.com/32642002/234999065-6784be53-31bf-4e49-858a-db023028381b.png](https://user-images.githubusercontent.com/32642002/234999065-6784be53-31bf-4e49-858a-db023028381b.png)

| Rouge 1 | Rouge 2 | Rouge L | Rouge Lsum |
| --- | --- | --- | --- |
| 52.96 | 31.18 | 46.04 | 46.11 |

| Generated Summary | Label Summary |
| --- | --- |
| Amanda can't find Betty's number. She will ask Larry for it. Hannah doesn't know him well, but she will text him. | Hannah needs Betty's number but Amanda doesn't have it. She needs to contact Larry. |
| Eric and Rob like his stand-up. | Eric and Rob are going to watch a stand-up on youtube. |
| Lenny will buy the first or the third pair of black trousers. | Lenny can't decide which trousers to buy. Bob advised Lenny on that topic. Lenny goes with Bob's advice to pick the trousers that are of best quality. |
| Emma doesn't want to cook dinner tonight. She will be home soon. Will will pick her up. | Emma will be home soon and she will let Will know. |
| Jane is back from her trip to Morocco. Ollie and Jane will meet for lunch on Friday at 6 pm. Jane will bring the pastries. | Jane is in Warsaw. Ollie and Jane has a party. Jane lost her calendar. They will get a lunch this week on Friday. Ollie accidentally called Jane and talked about whisky. Jane cancels lunch. They'll meet for a tea at 6 pm. |

## BART + Multi-Aware(Speaker-Aware + Topic-Aware) Fine-tuning

### Method

- 위에서 실험한 Speaker-Aware와 Topic-Aware를 함께 진행한 경우
- Speaker-Aware의 Contrastive

### Results

![https://user-images.githubusercontent.com/32642002/234999172-e18c015f-4d8d-468f-8e1d-14cab089706d.png](https://user-images.githubusercontent.com/32642002/234999172-e18c015f-4d8d-468f-8e1d-14cab089706d.png)

| Rouge 1 | Rouge 2 | Rouge L | Rouge Lsum |
| --- | --- | --- | --- |
| 53.18 | 31.17 | 46.20 | 46.20 |

| Generated Summary | Label Summary |
| --- | --- |
| Amanda can't find Betty's number. She will ask Larry, who called her last time they were at the park. | Hannah needs Betty's number but Amanda doesn't have it. She needs to contact Larry. |
| Eric and Rob like his Russian stand-up. They will watch some of his stand-ups on youtube. | Eric and Rob are going to watch a stand-up on youtube. |
| Lenny will buy the first or the third pair of black trousers. | Lenny can't decide which trousers to buy. Bob advised Lenny on that topic. Lenny goes with Bob's advice to pick the trousers that are of best quality. |
| Emma will be home soon and will let Will know what she wants for dinner. | Emma will be home soon and she will let Will know. |
| Jane is back from Morocco. Ollie has a party on the 18th. They will meet for lunch on Friday at 6 pm. Jane has already eaten the pastries. | Jane is in Warsaw. Ollie and Jane has a party. Jane lost her calendar. They will get a lunch this week on Friday. Ollie accidentally called Jane and talked about whisky. Jane cancels lunch. They'll meet for a tea at 6 pm. |