import sys
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
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.node_parser import SimpleNodeParser
from llama_index.evaluation import ResponseEvaluator
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from email_support_QA_prompts import EMAIL_SUPPORT_TEXT_QA_PROMPT
# from nlp_functions import contains_question

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

def main(userPrompt, saveResults):
    print(f"You passed the prompt:\n {userPrompt} \n\n")
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    # llm = OpenAI(temperature=0, model="gpt-4")
    model = "gpt-3.5-turbo"
    llm = OpenAI(temperature=0, model=model)

    augPrompt = "Here is a customer issue: \n\n" + f"'{userPrompt}'\n\n" + "Act as this customer. What are you trying to ask? Be concise. Generate the response as a question."
    resp = llm.complete(f"{augPrompt}")
    userPromptQAugment = userPrompt + "\n\n" + resp.text
    print(f"{userPromptQAugment}")

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

    #https://wandb.ai/ayush-thakur/llama-index-report/reports/Building-Advanced-Query-Engine-and-Evaluation-with-LlamaIndex-and-W-B--Vmlldzo0OTIzMjMy#setting-up-evaluation-using-llamaindex
    rerank = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=2
    )

    query_engine = index.as_query_engine(
        text_qa_template=EMAIL_SUPPORT_TEXT_QA_PROMPT,
        similarity_top_k=4,
        node_postprocessors=[rerank],
    )
    response = query_engine.query(userPromptQAugment)

    # Print info on llm inputs/outputs
    event_pairs = llama_debug.get_llm_inputs_outputs()
    print(event_pairs[0][0]) # Show what was sent to LLM
    print("\n\nAnswer (from gpt-3.5-turbo): ")
    print(response)

    # evaluator = ResponseEvaluator(service_context=service_context)
    # eval_result = evaluator.evaluate(response)
    # print("Does response match context?")
    # print(str(eval_result))

    if saveResults == '1':
        # Open the file in append mode ('a') and write the text
        with open("testResults.txt", "a") as file:
            file.write("Prompt:\n" + userPromptQAugment + "\n\n")
            file.write("Answer (" + model + "):\n" + response.response + "\n\n")
            # file.write("Does response match context?\n" + str(eval_result))
            file.write("\n\n---------------------------\n\n")

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
    if len(sys.argv) != 3:
        print("Usage: python callAI.py <prompt> <save results to txt file = 1, else = 0>")
    else:
        main(sys.argv[1], sys.argv[2])


