from langchain.llms import OpenAI
from llama_index.llms import OpenAI
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage,
    set_global_service_context,
)
from llama_index.node_parser import SimpleNodeParser
import sys
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from email_support_QA_prompts import EMAIL_SUPPORT_TEXT_QA_PROMPT
from nlp_functions import contains_question

def load_index_from_disk(index_path):
    try:
        storage_context = StorageContext.from_defaults(persist_dir=f"{index_path}")
        # load index
        index = load_index_from_storage(storage_context)
        print(f"Loaded index from {index_path}: {index}")
        return index
    except FileNotFoundError:
        print(f"Index {index_path} not found!")
        return None

def main(userPrompt):
    print(f"You passed the prompt:\n {userPrompt} \n\n")
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    # llm = OpenAI(temperature=0, model="gpt-4")
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo")

    user_prompt_contains_questions = contains_question(userPrompt)

    augPrompt = ""
    userPromptQAugment = userPrompt
    if user_prompt_contains_questions == False:
        augPrompt = "Here is a customer issue: \n\n" + f"'{userPrompt}'\n\n" + "Act as this customer. What are you trying to ask? Be concise. Generate the response as a question."    
        print(f"{augPrompt}")
        resp = llm.complete(f"{augPrompt}")
        print(f"{resp}")
        userPromptQAugment = userPrompt + "\n\n" + resp.text
        print(f"{userPromptQAugment}")
    else:
        userPromptQAugment = userPrompt

    service_context = ServiceContext.from_defaults(
        llm=llm,
        callback_manager=callback_manager
        )
    set_global_service_context(service_context)
    index_path = "./storage"
    index = load_index_from_disk(index_path)

    if index is None:
        documents = SimpleDirectoryReader('data').load_data()
        parser = SimpleNodeParser.from_defaults() # default chunk_size=1024, chunk_overlap=20
        nodes = parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes, service_context=service_context)
        index.storage_context.persist()

    query_engine = index.as_query_engine(
        text_qa_template=EMAIL_SUPPORT_TEXT_QA_PROMPT
    )
    response = query_engine.query(userPromptQAugment)

    # Print info on llm inputs/outputs
    event_pairs = llama_debug.get_llm_inputs_outputs()
    print(event_pairs[0][0]) # Show what was sent to LLM
    print("\n\nAnswer (from gpt-3.5-turbo): ")
    print(response)
    
    # substring = "will look into"

    # if substring in response.response:
    #     llm = OpenAI(temperature=0, model="gpt-4")
    #     service_context = ServiceContext.from_defaults(
    #         llm=llm,
    #         callback_manager=callback_manager
    #     )
    #     documents = SimpleDirectoryReader('data').load_data()
    #     parser = SimpleNodeParser.from_defaults() # default chunk_size=1024, chunk_overlap=20
    #     nodes = parser.get_nodes_from_documents(documents)
    #     index = VectorStoreIndex(nodes, service_context=service_context)
    #     query_engine = index.as_query_engine(
    #         text_qa_template=EMAIL_SUPPORT_TEXT_QA_PROMPT
    #     )
        
    #     response = query_engine.query(userPromptQuestionAug)
    #     event_pairs = llama_debug.get_llm_inputs_outputs()
    #     print(event_pairs[0][0]) # Show what was sent to LLM
    #     print(event_pairs[0][1].payload["response"]) # Shows the LLM response it generated.
    #     print("\n\nAnswer (from gpt-4): ")
    #     print(response)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python myscript.py <your_argument>")
    else:
        main(sys.argv[1])


