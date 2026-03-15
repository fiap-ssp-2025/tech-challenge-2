import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def explain_case(probability, top_features):

    prompt = f"""
    Um modelo de risco de recorrência de violência contra a mulher produziu:

    Probabilidade de recorrência: {probability:.2f}

    Fatores mais importantes:
    {top_features}

    Explique de forma clara para um profissional da assistência social.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
