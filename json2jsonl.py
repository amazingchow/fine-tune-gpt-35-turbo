# -*- coding: utf-8 -*-
import argparse
import glob
import os
import ujson as json

from colorama import just_fix_windows_console, Fore, Style
just_fix_windows_console()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tuning-Data-Post-Processing Utility")
    parser.add_argument("--input", type=str, help="dir to store JSON-structured data files", default="./data")
    parser.add_argument("--output", type=str, help="dir to store final training file", default="./data")
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "fine_tune_instructions.jsonl"), "w", encoding="utf-8") as jsonl_wt:
        data_files = glob.glob(os.path.join(input_dir, "*.json"))
        data_files = sorted(data_files)
        
        for data_file in data_files:
            with open(data_file, "r", encoding="utf-8") as fin:
                entry = json.load(fin)
                json.dump(entry, jsonl_wt, ensure_ascii=False)
                jsonl_wt.write('\n')

        print(f"{Fore.GREEN}-> Generated {os.path.join(output_dir, 'fine_tune_instructions.jsonl')}.{Style.RESET_ALL}")
