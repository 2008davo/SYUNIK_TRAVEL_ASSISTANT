import os
import json

data = []
with open("places.json", "r", encoding="utf-8") as f:
    data = json.load(f)

output_file = "all.txt"

def save_context_answer(results, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in results:
            f.write(f"{item['context']}. {item['answer']}.\n")

    print(f"File {filename} created successfully.")

def add_existing(results, filename):
    with open(filename, "r", encoding="utf-8") as f:
        existing_content = f.read()
    with open(results, "a", encoding="utf-8") as f:
        f.write(existing_content)
    print(f"File {results} updated successfully.")


# save_context_answer(data, output_file)
txt_files = ['Kapan.txt', 'Meghri.txt', 'Sisian.txt','Goris.txt','Kajaran.txt','Dastakert.txt','Agarak.txt']
for txt_file in txt_files:
    print(f"Adding content from {txt_file} to {output_file}...")
    add_existing(output_file, txt_file)

