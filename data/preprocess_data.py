# Preparing Chain-of-Thougts function
def prepare_cot_dataset(row):
  prompt = row["question"] + "\n\nThink step-by-step. Put your final exact answer inside <answer> and </answer> tags."
  assistant_target = f"{row["long_answer"]}\n\n<answer>{row["short_answer"]}</answer>."

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