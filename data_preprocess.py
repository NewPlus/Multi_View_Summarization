import random
from transformers import AutoTokenizer
from datasets import load_dataset
import re

class DataPreProcess:
    def __init__(self,
                 data_file_type="csv",
                 train_files="",
                 valid_files="",
                 test_files="",
                 model_name=""
                 ):
        self.data_file_type = data_file_type
        self.train_files = train_files
        self.valid_files = valid_files
        self.test_files = test_files
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Tokenizer에 Special Token으로 <sep>, Speaker(P01~P09), : 추가
        self.num_add_token = self.tokenizer.add_special_tokens({"additional_special_tokens":["<sep>", "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", ":"]})
    
    def preprocessing(self,
                      datasets,
                      train_valid_test="",
                      num_dialogue=1,
                      speaker_tokens=["P01:", "P02:", "P03:", "P04:", "P05:", "P06:", "P07:", "P08:", "P09:"]
                      ):
        encoded_sequence, raw_data = [], []
        # P01, P02, P03 등 작은 숫자 Speaker가 더 자주 등장 -> Speaker 토큰을 랜덤하게 전처리(P01, P02, P03 -> P02, P05, P09 등으로)
        for i in datasets[train_valid_test]['dialogue'][:num_dialogue]:
            # <sep> 기준으로 turn 나누기
            turn = i.split("<sep>")
            # old_speakers_token : 바꿔줄 기존 Speaker Token 목록
            old_speakers_token = list(set([re.sub("[^P0-9]", "", t) for t in turn]))
            # new_speakers_token : 바꿀 랜덤한  Speaker Token 목록
            new_speakers_token = [re.sub("[:]", "", j) for j in random.sample(speaker_tokens, len(old_speakers_token))]
            # Turn에 Speaker Token 랜덤하게 바꿔 넣기
            dialogue_turn = [re.sub(old, new, t) for t in turn for old, new in zip(old_speakers_token, new_speakers_token) if old in t]

            # Speaker Token을 섞고 <sep> Token 붙이기
            after_dialogue = ""
            for t in dialogue_turn:
                after_dialogue += "<sep>"+t
            print(f"after_dialogue : {after_dialogue}")

            # encoded_sequence : 전처리한 Dialogue를 tokenizing -> padding은 최대 길이 기준, return 값은 pytorch로 input_ids만
            encoded_sequence.append(self.tokenizer(after_dialogue, padding='max_length', return_tensors='pt')['input_ids'])
            raw_data.append(after_dialogue)

        # model에 업데이트할 최종 vocab_size
        vocab_size = self.tokenizer.vocab_size + self.num_add_token
        
        # all_special_ids : tokenizer에 등록된 special token들의 id List
        all_special_ids = self.tokenizer.all_special_ids

        return encoded_sequence, all_special_ids, vocab_size, raw_data

    def __call__(self,
                 num_dialogue=1,
                 speaker_tokens_list=["P01:", "P02:", "P03:", "P04:", "P05:", "P06:", "P07:", "P08:", "P09:"]
                ):
        # dataset 불러오기(csv)
        datasets = load_dataset(self.data_file_type, data_files={"train": self.train_files, "valid": self.valid_files, "test": self.test_files})
        print(f"datasets : {datasets}")
        # speaker_tokens : Speaker Token List
        speaker_tokens = speaker_tokens_list

        train = self.preprocessing(datasets, train_valid_test="train", num_dialogue=num_dialogue, speaker_tokens=speaker_tokens)
        valid = self.preprocessing(datasets, train_valid_test="valid", num_dialogue=num_dialogue, speaker_tokens=speaker_tokens)
        test = self.preprocessing(datasets, train_valid_test="test", num_dialogue=num_dialogue, speaker_tokens=speaker_tokens)

        print(f"train : {train[1]}\n valid : {valid[1]}\n test : {test[1]}")
        return train, valid, test, self.tokenizer
