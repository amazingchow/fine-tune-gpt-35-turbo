# fine-tune-gpt-35-turbo

### Training Steps

```bash
# STEP 1: 
python prepare_data.py --raw_data=./test/raw_data/qa.txt --base_system_instruction=./test/raw_data/fine_tune_instructions_base.json --output=./data

# STEP 2: 
python json2jsonl.py --input=./data --output=./data

# STEP 3: 
python fine_tune.py --action=check --json_dir=./data

# STEP 4: 
python fine_tune.py --action=upload --jsonl_file=./data/fine_tune_instructions.jsonl

# STEP 5: 
python fine_tune.py --action=start

# STEP 6: 
python fine_tune.py --action=status
```

### Limitations && Warnings

* Right now we can only fine-tune gpt-3.5-turbo (gpt-3.5-turbo-0613 specifically) which has 4K context.
* The cost of fine-tuning itself is quite low ($0.008 for 1K tokens of the dataset), but the main problem is the inference cost - because the fine-tuned model will be only used by you, the inference will cost 8 times more compared to normal 4K Turbo, which makes it almost half as expensive as GPT-4.
* The fine-tune model cannot be shared between different OpenAI accounts, so the only way to have the "same" fine-tune is to run the fine-tune job on all the separate accounts you want to use.
* The dataset for the fine-tune has to be 100% SFW, because, to quote OpenAI - "fine-tuning training data is passed through our Moderation API and a GPT-4 powered moderation system to detect unsafe training data that conflict with our safety standards". The Moderation API is quite strict, so even things like "sucking on a finger" won't pass.
* The owner of the account will get an email when a fine-tune finishes.

### References

* [Fine-Tuning Doc by OpenAI](https://platform.openai.com/docs/guides/fine-tuning/use-a-fine-tuned-model)
