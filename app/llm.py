from groq import Groq

from app.config import get_settings


SYSTEM_PROMPT = """You are a careful retrieval-augmented assistant.
Answer only from the supplied context.
If the context does not contain enough evidence, say you do not know.
Cite supporting context items using [1], [2], etc.
Do not invent sources or facts."""


def generate_answer(question: str, context: str) -> str:
    settings = get_settings()

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    if not settings.groq_model:
        raise RuntimeError("GROQ_MODEL is not configured.")

    client = Groq(api_key=settings.groq_api_key)

    response = client.chat.completions.create(
        model=settings.groq_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Question:
{question}

Retrieved context:
{context}

Answer the question using only the retrieved context and include citations.""",
            },
        ],
    )

    return response.choices[0].message.content or ""
