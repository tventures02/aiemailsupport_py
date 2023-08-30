# modified from https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/chat_prompts.py

from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.prompts.base import ChatPromptTemplate

SYSTEM_PROMPT = ("You are the expert Q&A email customer support system for a company.\n"
        "Answer the query using the provided context information, "
        "and not prior knowledge.\n"
        "Some rules to follow:\n"
        "1. Never directly reference the given context in your answer.\n"
        "2. Avoid statements like 'Based on the context, ...' or "
        "'The context information does not provide ...' or anything along those lines.\n"
        "3. Never make up an answer. If you are unsure, ask the customer to clarify.\n"
        "4. Avoid statments like 'send us an email' or 'contact our support team'.\n\n"
        "If you cannot form an exact response from the context information, "
        # "if there is large uncertainty in forming the answer or "
        # "if you cannot do what the customer is requesting, "
        "respond with exactly thank you for reaching out, "
        "that you will look into the inquiry, "
        "contact them back as soon as you can, "
        "and ask them provide more information if possible.\n")

# text qa prompt
EMAIL_SUPPORT_QA_SYSTEM_PROMPT = ChatMessage(
    content=SYSTEM_PROMPT,
    role=MessageRole.SYSTEM,
)

EMAIL_SUPPORT_QA_PROMPT_TMPL_MSGS = [
    EMAIL_SUPPORT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "The context information is delimited by triple backticks:\n"
            "```{context_str}```\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

EMAIL_SUPPORT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=EMAIL_SUPPORT_QA_PROMPT_TMPL_MSGS)

UNWANTED_SENTENCE_PHRASES = [
    "email support system",
    "customer support system",
    "contact our support team",
    "contacting our support team",
    "provided context information",
    "context information provided"
    ]