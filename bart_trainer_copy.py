from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
import evaluate
import numpy as np
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from bartmodel import BartModel, BartForConditionalGeneration
from data_preprocess import DataPreProcess
import numpy as np
import re
import random

model_name = "gogamza/kobart-base-v2"

datasets = load_dataset("csv", data_files={"train": "data/train.csv", "valid": "data/valid.csv", "test": "data/test.csv"})
# tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

preprocess_datas = DataPreProcess(data_file_type="csv",
                             train_files="data/train.csv",
                             valid_files="data/valid.csv",
                             test_files="data/test.csv",
                             model_name="gogamza/kobart-base-v2")
train, valid, test, tokenizer = preprocess_datas(num_dialogue=5)
# Model special_token 추가한 수로 vocab_size 맞춰주기
model.resize_token_embeddings(train[2])

def preprocess_function(examples):
    # inputs, summary = [], []
    # speaker_tokens = ["P01:", "P02:", "P03:", "P04:", "P05:", "P06:", "P07:", "P08:", "P09:"]

    # # print(f"x['dialogue'].split('<sep>') : {examples['dialogue'].split('<sep>')}")

    # # P01, P02, P03 등 작은 숫자 Speaker가 더 자주 등장 -> Speaker 토큰을 랜덤하게 전처리(P01, P02, P03 -> P02, P05, P09 등으로)
    # for i, j in zip(examples['dialogue'], examples["summary"]):

    #     # <sep> 기준으로 turn 나누기
    #     turn = i.split("<sep>")
    #     # old_speakers_token : 바꿔줄 기존 Speaker Token 목록
    #     print(f"turn : {turn}")
    #     old_speakers_token = list(set([re.sub("[^P01]|[^P02]|[^P03]|[^P04]|[^P05]|[^P06]|[^P07]|[^P08]|[^P09]", "", t) for t in turn])) # P02|P03|P04|P05|P06|P07|P08|P09
    #     print(f"old_speakers_token : {old_speakers_token}")
    #     # new_speakers_token : 바꿀 랜덤한  Speaker Token 목록
    #     new_speakers_token = [re.sub("[:]", "", j) for j in random.sample(speaker_tokens, k=(len(old_speakers_token)%10))]
    #     # Turn에 Speaker Token 랜덤하게 바꿔 넣기
    #     dialogue_turn = [re.sub(old, new, t) for t in turn for old, new in zip(old_speakers_token, new_speakers_token) if old in t]

    #     # Speaker Token을 섞고 <sep> Token 붙이기
    #     after_dialogue = ""
    #     for t in dialogue_turn:
    #         after_dialogue += "<sep>"+t
        
    #     assert len(after_dialogue) > 0, f"after_dialogue : {after_dialogue}, turn : {turn}, old_speakers_token : {old_speakers_token}"
    #     print(f"after_dialogue : {after_dialogue}")
    #     # print(f"after_dialogue : {len(after_dialogue)}")
    #     # print(f"summary : {len(j)}")

    #     # inputs : 전처리한 Dialogue를 tokenizing -> max_length=1024, truncation=True
    #     inputs.append(after_dialogue)
    #     summary.append(j)
    # model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    # labels = tokenizer(text_target=summary, max_length=128, truncation=True)
    # model_inputs["labels"] = labels["input_ids"]
    # return model_inputs
    inputs = [doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def filter_by_num_words(example):
    return len(example["dialogue"].split("<sep>")) > 7

filter_datasets = datasets.filter(filter_by_num_words)
tokenized_data = filter_datasets.map(preprocess_function, batched=True)
print(f"tokenized_data : {tokenized_data}")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

training_args = Seq2SeqTrainingArguments(
    output_dir="test_save",
    per_device_train_batch_size= 8,
    per_device_eval_batch_size= 8,
    save_total_limit=3,
    evaluation_strategy="steps",
    gradient_accumulation_steps= 1,
    learning_rate= 2e-5,
    max_steps=10000,
    eval_steps=1000,
    save_steps=1000,
    weight_decay= 0.1,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    fp16=True,
    seed=1
)

class BartTrainer(Seq2SeqTrainer):
    def __init__(self, all_special_ids, raw_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_special_ids = all_special_ids
        self.raw_data = raw_data

    def compute_loss(self, model, inputs, return_outputs=False):
        # implement custom logic here
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        print(f"!!!?")
        outputs = model(**inputs, all_special_ids=self.all_special_ids, raw_data=self.raw_data)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            #if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            #    loss = self.label_smoother(outputs, labels, shift_labels=True)
            #else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss += outputs["ctr_loss"]

        return (loss, outputs) if return_outputs else loss

trainer = BartTrainer(
    model=model, # all_special_ids = train[1]
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['valid'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    all_special_ids=train[1],
    raw_data=tokenized_data['train']
)
trainer.train()

predict_results = trainer.predict(
            tokenized_data['test'],
            metric_key_prefix=" ",
            max_length=80,
            num_beams=6,
            length_penalty=1.0,
            no_repeat_ngram_size=3
        )

metrics = predict_results.metrics
trainer.log_metrics("predict", metrics)
trainer.save_metrics("predict", metrics)