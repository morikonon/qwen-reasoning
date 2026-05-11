## Qwen reasoning

In this work I implemented fine tuning Qwen3.5-0.8B model in Math Reasoning dataset.
Model must learn how to answer like "thinking" model. This thinking model helps to model understand task more deeper and solve it correctly.

# To run this project in your laptop
''' bash
git clone https://github.com/morikonon/qwen-reasoning.git

python3 -m venv venv

source venv venv

pip install -r requirements.txt

uvicorn app.app:app --reload

streamit run app.ui.py
'''