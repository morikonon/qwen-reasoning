# Preparing Chain-of-Thoughts function
def prepare_cot_dataset(row):
    prompt = row["question"] + "\n\nThink step-by-step. Wrap your reasoning in <think></think> tags and put your final exact answer inside <answer></answer> tags."
    assistant_target = "<think>{}</think>\n<answer>{}</answer>".format(row["long_answer"], row["short_answer"])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_target}
            ]
        }
    ]

    return messages
