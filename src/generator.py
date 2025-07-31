from openai import OpenAI
import pandas as pd
import os
import json
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load env variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(" currently in use API Key:", "Loaded" if api_key else "Not Found ")
client = OpenAI(api_key=api_key)

# Prompt
PROMPT_TEMPLATE = (
    "Please generate a fictional diabetes patient record as a JSON object with the following fields: "
    "'Pregnancies' (int), 'Glucose' (int), 'BloodPressure' (int), 'SkinThickness' (int), 'Insulin' (int), "
    "'BMI' (float), 'DiabetesPedigreeFunction' (float), 'Age' (int), 'Outcome' (0 or 1). "
    "Return only the JSON object, no explanation."
)

def extract_json(text: str):
    """
    Try to extract the JSON object string from the output of the large model
    """
    try:
        json_str = re.search(r"\{.*\}", text, re.DOTALL).group()
        return json.loads(json_str)
    except Exception as e:
        print("JSON parsing failed:", e)
        print("Original returned content:\n", text)
        return None

def generate_synthetic_data(prompt=PROMPT_TEMPLATE, num_samples=5, save=True) -> pd.DataFrame:
    data_rows = []

    for i in range(num_samples):
        print(f"\n🌀 In the generation of item {i + 1}...")
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            text = response.choices[0].message.content
            row = extract_json(text)
            if row:
                data_rows.append(row)
            else:
                print(" JSON structural error. Skip")
        except Exception as e:
            print("Generation failed", e)

    df = pd.DataFrame(data_rows)

    if save and not df.empty:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_path = Path("data/synthetic")
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / f"synthetic_{timestamp}.csv"
        df.to_csv(filepath, index=False)
        print(f"save successfully:{filepath}")
    else:
        print("All sample generation failed and no content was saved")

    print("\nPreview of the first 5 pieces of data")
    print(df.head())
    return df

if __name__ == "__main__":
    generate_synthetic_data()
