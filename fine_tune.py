# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.curdir), "modules"))

import argparse
import asyncio
# More info: https://github.com/aio-libs/aiohttp/discussions/6044.
setattr(asyncio.sslproto._SSLProtocolTransport, "_start_tls_compatible", True)
import datetime
import openai

from colorama import Fore, Style
from modules import data_check, config
config.success()

tmp_dir = "./.tmp"
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)


def upload_data(data_file: str):
    try:
        response = openai.File.create(
            file=open(data_file, "rb"),
            purpose="fine-tune"
        )
        if "status" in response and response["status"] == "uploaded":
            print(f"{Fore.GREEN}-> Uploaded a data file <fid: {response['id']}> for fine-tune!\n{Style.RESET_ALL}")
            with open(os.path.join(tmp_dir, "file_id.txt"), "w") as f:
                f.write(response["id"])
        else:
            print(f"{Fore.RED}-> Failed to upload the data file <fn: {data_file}> for fine-tune, err:{response}.\n{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}-> Failed to upload the data file <fn: {data_file}> for fine-tune, err:{e}.\n{Style.RESET_ALL}")


def start_fine_tune(n_epochs: int, model: str = "gpt-3.5-turbo-0613"):
    with open(os.path.join(tmp_dir, "file_id.txt"), "r") as f:
        fid = f.read()
    print(f"{Fore.GREEN}-> Use uploaded data file <fid: {fid}> to start a fine-tune job...\n{Style.RESET_ALL}")

    try:
        response = openai.FineTuningJob.create(
            training_file=fid,
            model=model,
            hyperparameters={"n_epochs": n_epochs}
        )
        if "status" in response and response["status"] == "created":
            print(f"{Fore.GREEN}-> Started a fine-tune job <job_id:{response['id']}>!\n{Style.RESET_ALL}")
            with open(os.path.join(tmp_dir, "ft_id.txt"), "w") as f:
                f.write(response["id"])
        else:
            print(f"{Fore.RED}-> Failed to start the fine-tune job, err:{response}.\n{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}-> Failed to start the fine-tune job, err:{e}.\n{Style.RESET_ALL}")


def check_fine_tune():
    try:
        ft_id = open(os.path.join(tmp_dir, "ft_id.txt"), "r").read()
        if len(ft_id) < 5:
            raise
    except Exception:
        print(f"{Fore.YELLOW}-> Failed to get job-id from local file {os.path.join(tmp_dir, 'ft_id.txt')}, try to get the newest one from OpenAI instead.\n{Style.RESET_ALL}")
        ft_id = openai.FineTuningJob.list(limit=1)["data"][0]["id"]
    
    try:
        job = openai.FineTuningJob.retrieve(ft_id)
        status = job["status"]
        if status == "succeeded":
            print(f"{Fore.GREEN}-> The fine-tune job <job_id:{ft_id}> is done! The created fine-tune model name is {job['fine_tuned_model']}.\n{Style.RESET_ALL}")
        elif status in ["failed", "cancelled"]:
            print(f"{Fore.RED}-> The fine-tune job <job_id:{ft_id}> failed or got cancelled, no clue why.\n{Style.RESET_ALL}")
        else:
            print(f"{Fore.BLUE}-> The fine-tuning job <job_id:{ft_id}> is {status}.\n{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}-> Last 10 events from current fine-tuning job <job_id:{ft_id}>:\n{Style.RESET_ALL}")
        events = openai.FineTuningJob.list_events(id=ft_id, limit=10)
        sorted_events = sorted(events["data"], key=lambda x: x["created_at"])
        for event in sorted_events:
            date = datetime.datetime.fromtimestamp(event["created_at"])
            print(f"{date} - {event['message']}")
    except Exception as e:
        print(f"{Fore.RED}-> Failed to check the fine-tune job <job_id:{ft_id}>, err:{e}.\n{Style.RESET_ALL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT Model Fine-Tuning Utility")
    parser.add_argument("--action", type=str, help="action to perform: check | upload | start | status", required=True)
    parser.add_argument("--json_dir", type=str, help="dir to store JSON-structured example files", default="./data")
    parser.add_argument("--jsonl_file", type=str, help="JSONL-structured file for fine-tuning", default="./data/fine_tune_instructions.jsonl")
    args = parser.parse_args()
    action = args.action
    json_dir = args.json_dir
    jsonl_file = args.jsonl_file
    
    if action == "check":
        print(f"{Fore.GREEN}-> Performing action: {action}\n{Style.RESET_ALL}")
        n_epochs = data_check.check_data_formatting(json_dir)
        with open(os.path.join(tmp_dir, "n_epochs.txt"), "w") as f:
            f.write(f"{n_epochs}")
        print(f"{Fore.GREEN}-> Done action: {action}\n{Style.RESET_ALL}")
    elif action == "upload":
        print(f"{Fore.GREEN}-> Performing action: {action}\n{Style.RESET_ALL}")
        upload_data(jsonl_file)
        print(f"{Fore.GREEN}-> Done action: {action}\n{Style.RESET_ALL}")
    elif action == "start":
        print(f"{Fore.GREEN}-> Performing action: {action}\n{Style.RESET_ALL}")
        with open(os.path.join(tmp_dir, "n_epochs.txt"), "r") as f:
            n_epochs = int(f.read())
        start_fine_tune(n_epochs)
        print(f"{Fore.GREEN}-> Done action: {action}\n{Style.RESET_ALL}")
    elif action == "status":
        print(f"{Fore.GREEN}-> Performing action: {action}\n{Style.RESET_ALL}")
        check_fine_tune()
        print(f"{Fore.GREEN}-> Done action: {action}\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}-> Unknown action: {action}\n{Style.RESET_ALL}")
