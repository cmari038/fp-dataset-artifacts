import json

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
#ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")
#ds = ds["train"].train_test_split(0.30)
# model = AutoModel.from_pretrained("/path/to/model_name", use_safetensors=True)
#ds["train"].to_json("train.json")

errors = []
success = []
with open("eval_predictions_copy.jsonl", "r") as file:
    for prediction in file:
        pred = json.loads(prediction)
        if pred["label"] != pred["predicted_label"]:
            errors.append(pred)
        else:
            success.append(prediction)
print(errors)
print(success)

