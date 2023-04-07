from bartmodel import BartModel
from data_preprocess import DataPreProcess

# # Tokenizer와 Model 불러오기(gogamza/kobart-base-v2)
model_name = "gogamza/kobart-base-v2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartModel.from_pretrained(model_name)

dialogue_turn = []
num_dialogue = 5

datas = DataPreProcess(train_valid_test="train", data_file_type="csv", data_files="data/train.csv", model_name="gogamza/kobart-base-v2")
print(f"datas : {datas}")
input_ids, all_special_ids, vocab_size, raw_data = datas(num_dialogue=5)

# Model special_token 추가한 수로 vocab_size 맞춰주기
model.resize_token_embeddings(vocab_size)
for id in input_ids:
    # model에 tokenizer를 거친 Dialogue를 입력, all_special_ids에 special token들의 id List도 함께 전달
    last_hidden_states = model(id, all_special_ids=all_special_ids, raw_data=raw_data).last_hidden_state

# # dataset 불러오기(csv)
# datasets = load_dataset("csv", data_files="data/train.csv")
# print(f"datasets : {type(datasets)}")
# # Speaker Token List
# speaker_tokens = ["P01:", "P02:", "P03:", "P04:", "P05:", "P06:", "P07:", "P08:", "P09:"]


# dialogue_turn = []
# num_dialogue = 5



# # P01, P02, P03 등 작은 숫자 Speaker가 더 자주 등장 -> Speaker 토큰을 랜덤하게 전처리(P01, P02, P03 -> P02, P05, P09 등으로)
# for i in datasets['train']['dialogue'][:num_dialogue]:
#     # <sep> 기준으로 turn 나누기
#     turn = i.split("<sep>")
#     # old_speakers_token : 바꿔줄 기존 Speaker Token 목록
#     old_speakers_token = list(set([re.sub("[^P0-9]", "", t) for t in turn]))
#     # new_speakers_token : 바꿀 랜덤한  Speaker Token 목록
#     new_speakers_token = [re.sub("[:]", "", j) for j in random.sample(speaker_tokens, len(old_speakers_token))]
#     # Turn에 Speaker Token 랜덤하게 바꿔 넣기
#     dialogue_turn = [re.sub(old, new, t) for t in turn for old, new in zip(old_speakers_token, new_speakers_token) if old in t]

#     # Speaker Token을 섞고 <sep> Token 붙이기
#     after_dialogue = ""
#     for t in dialogue_turn:
#         after_dialogue += "<sep>"+t

#     # encoded_sequence : 전처리한 Dialogue를 tokenizing -> padding은 최대 길이 기준, return 값은 pytorch로 input_ids만
#     encoded_sequence = tokenizer(after_dialogue, padding='max_length', return_tensors='pt')['input_ids']
#     # all_special_ids : tokenizer에 등록된 special token들의 id List
#     all_special_ids = tokenizer.all_special_ids
#     # last_hidden_states : model로부터 나온 마지막 hidden states 값
#     # model에 tokenizer를 거친 Dialogue를 입력, all_special_ids에 special token들의 id List도 함께 전달
#     last_hidden_states = model(encoded_sequence, all_special_ids=all_special_ids).last_hidden_state
