# -*- coding: utf-8 -*-
import glob
import numpy as np
import openai
import os
import tiktoken
import ujson as json

from collections import defaultdict
from colorama import Fore, Style
from typing import Dict, List


def num_tokens_from_messages(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo-0613") -> int:
    """
    Return the number of tokens used by a list of messages.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"{Fore.YELLOW}-> Warning: Model not found. Using cl100k_base encoding.\n{Style.RESET_ALL}")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613"
        }:

        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4
        tokens_per_name = -1
    elif "gpt-3.5-turbo" in model:
        print(f"{Fore.YELLOW}-> Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.\n{Style.RESET_ALL}")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(f"{Fore.YELLOW}-> Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.\n{Style.RESET_ALL}")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    # every reply is primed with <|start|>assistant<|message|>
    num_tokens += 3
    return num_tokens


def num_assistant_tokens_from_messages(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo-0613") -> int:
    """
    Return the number of tokens used by a list of assistant-messages.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"{Fore.YELLOW}-> Warning: model not found. Using cl100k_base encoding.\n{Style.RESET_ALL}")
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens


def check_moderation(data_file: str, messages: List[Dict[str, str]]):
    """
    To check whether content complies with the OpenAI content policy, https://openai.com/policies/usage-policies.
    """
    batch_size = 32

    for batch_start in range(0, len(messages), batch_size):
        batches = messages[batch_start:batch_start + batch_size]
        contents = [message["content"] for message in batches]
        
        mod_check = openai.Moderation.create(input=contents)["results"]
        for i, result in enumerate(mod_check):
            # 'flagged' will be set to true if the model classifies the content as violating OpenAI's usage policies.
            if result["flagged"]:
                global_index = batch_start + i
                print(f"{Fore.RED}-> Message {global_index} in file {data_file} got flagged!\n{Style.RESET_ALL}")
                print(f"Message contents (truncated): {messages[global_index]['content'][:50]}...")
                for name, flagged in result["categories"].items():
                    if flagged:
                        print(f"{Fore.RED}-> Flagged category {name} with the score {result['category_scores'][name]}.\n{Style.RESET_ALL}")


def check_format_errors(data_file: str, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo-0613") -> Dict[str, int]:
    """
    To check to make sure the formatting is correct and matches the Chat completions message structure.
    Also need to ensure that the length does not exceed the 4096 token limit.

    SUPPLEMENTARY INSTRUCTION
    -------------------------

    * Each training example is limited to 4096 tokens.
      Examples longer than this will be truncated to the first 4096 tokens when training.
      To be sure that your entire training example fits in context,
      consider checking that the total token counts in the message contents are under 4,000.
      Each file is currently limited to 50 MB.
    """
    format_errors = defaultdict(int)
    
    for message in messages:
        if not isinstance(message, dict):
            format_errors["message_data_type"] += 1

        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1

        if any(k not in ("role", "content", "name") for k in message):
            format_errors["message_unrecognized_key"] += 1

        if message.get("role", None) not in ("system", "user", "assistant"):
            format_errors["message_unrecognized_role"] += 1
        
        content = message.get("content", None)
        if not content or not isinstance(content, str):
            format_errors["message_missing_content"] += 1

    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["messages_missing_assistant_message"] += 1

    convo_len = num_tokens_from_messages(messages, model)
    if convo_len > 4096:
        format_errors["messages_token_limit"] = 1

    check_moderation(data_file, messages)

    return format_errors


def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}\n")


def check_data_formatting(data_dir: str, model: str = "gpt-3.5-turbo-0613") -> int:
    """
    Once you have compiled a dataset and before you create a fine-tuning job,
    it is important to check the data formatting.

    SUPPLEMENTARY INSTRUCTION
    -------------------------

    * Pricing
      
      Fine-tuning costs are broken down into two buckets: the initial training cost and usage cost:

      - Training: $0.008 / 1K Tokens
      - Usage input: $0.012 / 1K Tokens
      - Usage output: $0.016 / 1K Tokens
      
      For example, a gpt-3.5-turbo fine-tuning job with a training file of 100,000 tokens that is trained for 3 epochs would have an expected cost of $2.40.
    """
    print(f"{Fore.GREEN}---------- ST DATA FORMATTING CHECK ----------\n{Style.RESET_ALL}")

    data_files = glob.glob(os.path.join(data_dir, "*.json"))
    data_files = sorted(data_files)
    
    # Warnings and tokens counts
    total_tokens = 0
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []
    for data_file in data_files:
        with open(data_file, "r") as fin:
            messages = json.load(fin).get("messages", [])
        
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        num_tokens = num_tokens_from_messages(messages, model)
        total_tokens += num_tokens
        convo_lens.append(num_tokens)
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages, model))

        format_errors = check_format_errors(data_file, messages, model)
        if format_errors:
            print(f"{Fore.RED}-> Found errors in data file {data_file}:\n{Style.RESET_ALL}")
            for k, v in format_errors.items():
                if k == "messages_token_limit":
                    print(f"{Fore.RED}-> The data file may be over the 4096 token limit, it will be truncated during fine-tuning.\n{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}-> {k}: {v}\n{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}-> No errors found in data file {data_file}.\n{Style.RESET_ALL}")

    if n_missing_system > 0:
        print(f"{Fore.RED}-> {n_missing_system} data files missing system message.\n{Style.RESET_ALL}")
    if n_missing_user > 0:
        print(f"{Fore.RED}-> {n_missing_user} data files missing user message.\n{Style.RESET_ALL}")
    print_distribution(n_messages, "num_messages_per_data_file")
    print_distribution(convo_lens, "num_total_tokens_per_data_file")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_data_file")

    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = 4096
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    TARGET_EPOCHS = 3
    MIN_EPOCHS = 1
    MAX_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(data_files)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)
    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    print(f"{Fore.GREEN}-> Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training.\n{Style.RESET_ALL}")
    print(f"{Fore.GREEN}-> By default, you'll train for {n_epochs} epochs on this dataset.\n{Style.RESET_ALL}")
    print(f"{Fore.GREEN}-> By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens.\n{Style.RESET_ALL}")

    token_cost_1k = 0.008
    training_cost = (total_tokens / 1000) * token_cost_1k * n_epochs
    print(f"{Fore.GREEN}-> Fine-Tune will cost ~${training_cost:.2f} (epochs = {n_epochs}).\n{Style.RESET_ALL}")

    print(f"{Fore.GREEN}---------- ED DATA FORMATTING CHECK ----------\n{Style.RESET_ALL}")
    return n_epochs
