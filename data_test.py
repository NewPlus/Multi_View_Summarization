import datasets
import tqdm
import re
from datasets import load_dataset
import pandas as pd
import transformers
from transformers import AutoTokenizer, BartForConditionalGeneration

model_name = "facebook/bart-large"

# df = pd.DataFrame()
# data = load_dataset("csv", data_files={"train": "data/ami/train.csv", "valid": "data/ami/valid.csv", "test": "data/ami/test.csv"})
# # print(f"data : {data['train']['dialogue'][6]}")
# speakers_tokens = ["Speaker A", "Speaker B", "Speaker C", "Speaker D", "Speaker E", "Speaker F", "Speaker G", "Speaker H", "Speaker I"]
# preprocess_tokens = ["<sep>Speaker A", "<sep>Speaker B", "<sep>Speaker C", "<sep>Speaker D", "<sep>Speaker E", "<sep>Speaker F", "<sep>Speaker G", "<sep>Speaker H", "<sep>Speaker I"]

# dialog_non_enters = [dialog.split("\n") for dialog in data['train']['dialogue']]
# # print("==================================")
# # print(f"{dialog_non_enters[0]}")
# # print("==================================")

# train_ppdialog = []
# for dialog in dialog_non_enters:
#     str = ""
#     for turn in dialog:
#         # print("==================================")
#         # print(turn)
#         # print("==================================")
#         for i, j in zip(speakers_tokens, preprocess_tokens):
#             if turn.find(i) != -1:
#                 # print(f"{re.sub(i, j, turn)}")
#                 str += re.sub(i, j, turn)
#     train_ppdialog.append(str)

# df = pd.DataFrame()
# df['dialogue'] = train_ppdialog
# df['summary'] = data['train']['summary']

# print(df.head())

data = load_dataset("samsum")

# speakers_tokens = ["Speaker A", "Speaker B", "Speaker C", "Speaker D", "Speaker E", "Speaker F", "Speaker G", "Speaker H", "Speaker I"]

model = BartForConditionalGeneration.from_pretrained("philschmid/bart-large-cnn-samsum")
tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")

print("====================================")
s=0
for i, j in zip(data["test"]["dialogue"][:5], data["test"]["summary"][:5]):
    s+=1
    print(f"==============dialogue {s}================")
    print(i)
    print(f"==============label summary {s}================")
    print(j)
    inputs = tokenizer(i, max_length=1024, return_tensors="pt")        
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
    out = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"==============BART summary {s}================")
    print(out)
print("====================================")



# for i in zip(data["validation"]["dialogue"][:10]):
#     inputs = tokenizer(i, max_length=1024, return_tensors="pt")        
#     summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
#     out = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#     print(f"out : {out}")

# Generate Summary
# summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
# tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# preprocessing_dialog = []
# for dialog in dialogue:
#     preprocessing_dialog.append(re.sub("\r\n", "<sep>", dialog))

# print(preprocessing_dialog[0])

# for dialog in preprocessing_dialog:
#     sep_turn = dialog.split("<sep>")
#     print(f"sep_turn : {sep_turn}")
#     speaker_tok = []
#     for sep in sep_turn:
#         speaker_utter = sep.split(": ")
#         speaker_tok.append(speaker_utter[0])
#     speakers = list(set(speaker_tok))
#     num_speaker = len(speakers)
#     ppdialog = []
#     for sep in sep_turn:
#         sep = re.sub("[\(\)-=\<\>]", "", sep)
#         for i, j in zip(speakers, ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09"]):
#             if sep.find(i) != -1:
#                 ppdialog.append(re.sub(i, "<sep>"+j, sep))
#     print(f"ppdialog : {ppdialog}")
