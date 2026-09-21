# from langchain_ollama import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
# from vector import retriever

# model = OllamaLLM(model = "llama3.2")

# template = """
# You are an expert in answering questions about a pizza restaurant"

# Here are some relevant reviews: {reviews}

# Here is the question to answer: {question}

# """

# prompt = ChatPromptTemplate.from_template(template)

# chain = prompt | model

# while True:
#     print("\n\n -----------------------------------------")
#     question = input("Ask your questions (type 'q' to quit): ")
#     print("\n\n")
#     if question == "q":
#         break

#     reviews = retriever.invoke(question)
#     result = chain.invoke({"reviews": reviews, "question": question})
#     print(result)





































from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from vector import retriever

import pandas as pd
import os


# -----------------------------
# Load CSV for data analysis
# -----------------------------

DATA_FOLDER = "./data"


def load_csv():
    files = os.listdir(DATA_FOLDER)

    csv_files = [
        file for file in files
        if file.lower().endswith(".csv")
    ]

    if len(csv_files) == 1:
        csv_path = os.path.join(DATA_FOLDER, csv_files[0])
        return pd.read_csv(csv_path)

    return None


df = load_csv()


# -----------------------------
# LLM
# -----------------------------

model = OllamaLLM(
    model="llama3.2"
)


# -----------------------------
# RAG prompt
# -----------------------------

template = """
You are an AI assistant that answers questions using the provided document information.

Here is the relevant information from the document:

{reviews}

Here is the question:

{question}

Answer the question using the provided information.

If the answer cannot be found in the provided information,
say that you don't have enough information.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


# -----------------------------
# Detect data-analysis questions
# -----------------------------

def is_data_analysis_question(question):

    keywords = [
        "percentage",
        "percent",
        "%",
        "average",
        "mean",
        "median",
        "total",
        "sum",
        "count",
        "how many",
        "number of",
        "maximum",
        "minimum",
        "highest",
        "lowest",
        "most",
        "least"
    ]

    question = question.lower()

    return any(keyword in question for keyword in keywords)


# -----------------------------
# Handle CSV analysis
# -----------------------------

def analyze_csv(question):

    if df is None:
        return None

    question = question.lower()

    # --------------------------------
    # Positive reviews
    # --------------------------------

    if "positive" in question:

        positive_reviews = df[df["Rating"] >= 4]

        percentage = (
            len(positive_reviews) / len(df)
        ) * 100

        return (
            f"There are {len(df)} reviews in total. "
            f"{len(positive_reviews)} of them are positive "
            f"(rating 4 or 5). "
            f"Therefore, {percentage:.2f}% of the reviews are positive."
        )


    # --------------------------------
    # Negative reviews
    # --------------------------------

    if "negative" in question:

        negative_reviews = df[df["Rating"] <= 2]

        percentage = (
            len(negative_reviews) / len(df)
        ) * 100

        return (
            f"There are {len(df)} reviews in total. "
            f"{len(negative_reviews)} of them are negative "
            f"(rating 1 or 2). "
            f"Therefore, {percentage:.2f}% of the reviews are negative."
        )


    # --------------------------------
    # Average rating
    # --------------------------------

    if "average rating" in question or "mean rating" in question:

        average_rating = df["Rating"].mean()

        return (
            f"The average rating is {average_rating:.2f}."
        )


    # --------------------------------
    # Total number of reviews
    # --------------------------------

    if (
        "how many reviews" in question
        or "number of reviews" in question
        or "total reviews" in question
        or "total number of reviews" in question
    ):

        return (
            f"There are {len(df)} reviews in total."
        )


    # --------------------------------
    # 5-star reviews
    # --------------------------------

    if "5 star" in question or "5-star" in question:

        five_star_reviews = df[df["Rating"] == 5]

        percentage = (
            len(five_star_reviews) / len(df)
        ) * 100

        return (
            f"There are {len(five_star_reviews)} five-star reviews "
            f"out of {len(df)} total reviews. "
            f"That is {percentage:.2f}%."
        )


    # --------------------------------
    # 1-star reviews
    # --------------------------------

    if "1 star" in question or "1-star" in question:

        one_star_reviews = df[df["Rating"] == 1]

        percentage = (
            len(one_star_reviews) / len(df)
        ) * 100

        return (
            f"There are {len(one_star_reviews)} one-star reviews "
            f"out of {len(df)} total reviews. "
            f"That is {percentage:.2f}%."
        )


    return None


# -----------------------------
# Main loop
# -----------------------------

while True:

    print("\n\n-----------------------------------------")

    question = input(
        "Ask your question (type 'q' to quit): "
    )

    print("\n\n")

    if question.lower() == "q":
        break


    # --------------------------------
    # First try data analysis
    # --------------------------------

    if is_data_analysis_question(question):

        result = analyze_csv(question)

        if result is not None:
            print(result)
            continue


    # --------------------------------
    # Otherwise use RAG
    # --------------------------------

    reviews = retriever.invoke(question)

    result = chain.invoke({
        "reviews": reviews,
        "question": question
    })

    print(result)