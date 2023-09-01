# -*- coding: utf-8 -*-
import argparse
import os
import ujson as json

from colorama import just_fix_windows_console, Fore, Style
just_fix_windows_console()
from copy import deepcopy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tuning-Data-Preparing Utility")
    parser.add_argument("--raw_data", type=str, help="raw data file", default="./test/raw_data/qa.txt")
    parser.add_argument("--base_system_instruction", type=str, help="system instruction should appear in every example", default="./test/raw_data/fine_tune_instructions_base.json")
    parser.add_argument("--output", type=str, help="output dir to store structured data files", default="./data")
    args = parser.parse_args()
    raw_data_file = args.raw_data
    base_system_instruction_file = args.base_system_instruction
    output_dir = args.output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(raw_data_file, "r", encoding="utf-8") as f1, open(base_system_instruction_file, "r", encoding="utf-8") as f2:

        instructions = []
        idx = -1
        match = False
        for line in f1.readlines():
            if match and line.startswith("A:"):
                instructions[idx]["A"] = line[2:].rstrip()
                match = False
            elif not match and line.startswith("Q:"):
                instructions.append({"Q": line[2:].rstrip()})
                idx += 1
                match = True
        print(f"{Fore.GREEN}-> Loaded {len(instructions)} QA pairs.{Style.RESET_ALL}")

        base_instructions = json.loads(f2.read())

        feed_instructions = 0
        idx = 0
        batch_size = len(instructions) // 10
        for batch_start in range(0, len(instructions), batch_size):
            idx += 1

            _base_instructions = deepcopy(base_instructions)
            batched_instructions = instructions[batch_start:batch_start + batch_size]
            feed_instructions += len(batched_instructions)
            with open(os.path.join(output_dir, f"fine_tune_instructions_{idx:04d}.json"), "w", encoding="utf-8") as f3:
                for x in batched_instructions:
                    _base_instructions["messages"].append(
                        {
                            "role": "user",
                            "content": x["Q"]
                        }
                    )
                    _base_instructions["messages"].append(
                        {
                            "role": "assistant",
                            "content": x["A"]
                        }
                    )
                json.dump(_base_instructions, f3, indent=2, ensure_ascii=False)

            print(f"{Fore.GREEN}-> Generated {os.path.join(output_dir, f'fine_tune_instructions_{idx:04d}.json')}.{Style.RESET_ALL}")

        assert feed_instructions == len(instructions), "Check Your Shit Code!"
