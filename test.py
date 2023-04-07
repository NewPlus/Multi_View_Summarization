import numpy as np
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

model_name = "/root/bart_customize/test_save/checkpoint-8000"

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
    inputs = [doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = datasets.map(preprocess_function, batched=True)

# print(f"input : {tokenized_data['valid']}")

with open("predictions.json", "a") as f_predictions:
    for t, s in zip(tokenized_data['valid']['dialogue'], tokenized_data['valid']['summary']):
        turn_inputs = tokenizer(t, max_length=1024, return_tensors="pt")
        print(f"predict_inputss : {turn_inputs}")
        predictions = model.generate(turn_inputs['input_ids'])
        decode_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        print(f"decode_predict : {decode_predictions}")
        print(f"summary : {s}")
        rouge = evaluate.load("rouge")
        rouge_result = rouge.compute(predictions=decode_predictions, references=[s], tokenizer=lambda x: tokenizer.tokenize(x), use_stemmer=True)
        print(f"rouge : {rouge_result}")
        save = {"turn_text" : t, "predict" : decode_predictions, "label" : s, "rouge" : rouge_result}
        f_predictions.write(str({"turn_text" : t})+"\n")
        f_predictions.write(str({"predict" : decode_predictions})+"\n")
        f_predictions.write(str({"label" : s})+"\n")
        f_predictions.write(str({"rouge" : rouge_result})+"\n")
        f_predictions.write(str("=================================================")+"\n")

# with open("labels.json", "a") as f_labels:
#     for t in tokenized_data['valid']['summary'][:2]:
#         inputs = tokenizer(s, max_length=1024, return_tensors="pt")
#         print(f"label_inputs : {inputs}")
#         labels = model.generate(inputs['input_ids'])
#         print(f"label : {labels}")
#         decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#         save = {"labels" : decode_labels}
        # f_labels.write(str(labels)+"\n")


# print(f"input : {inputs}")









# def preprocess_function(examples):
#     inputs = [doc for doc in examples["dialogue"]]
#     model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
#     labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

# # filter_datasets = datasets.filter(filter_by_num_words)
# tokenized_data = datasets.map(preprocess_function, batched=True)
# # print(f"tokenized_data : {tokenized_data}")
# data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
# rouge = evaluate.load("rouge")

# inputs = tokenizer(tokenized_data, max_length=1024, return_tensors="pt")
# print(inputs)

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred

#     f_predictions = open("predictions.json", "a")
#     f_labels = open("labels.json", "a")
#     f_predictions.write(str(predictions)+"\n")
#     f_labels.write(str(labels)+"\n")

#     # print(f"predictions : {predictions[:5]}, labels : {labels[:5]}")
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    

#     result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
#     print(f"decoded_preds : {decoded_preds[:5]}")
#     print(f"decoded_labels : {decoded_labels[:5]}")
#     breakpoint()
#     result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
#     print(f"roudge : {result}")
#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
#     result["gen_len"] = np.mean(prediction_lens)
#     f_predictions.close()
#     f_labels.close()

#     return {k: round(v, 4) for k, v in result.items()}

