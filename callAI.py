import sys
import json
import nltk
nltk.data.path.append("./nltk_data")
from llama_index.llms import OpenAI
from llama_index import (
    ServiceContext,
    set_global_service_context,
)
# from llama_index.indices.postprocessor import SentenceTransformerRerank
from aiemailsupport_vectorstore import createAndSaveIndex
from aiemailsupport_vectorstore import loadIndex
from llama_index.node_parser import SimpleNodeParser
from llama_index.evaluation import ResponseEvaluator
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from email_support_QA_prompts import (
    EMAIL_SUPPORT_TEXT_QA_PROMPT,
    UNWANTED_SENTENCE_PHRASES
)
from nlp_functions import (
    find_matches,
    contains_question,
    )

def main(userPrompt, saveResults):
    print(f"You passed the prompt:")
    print(userPrompt + "\n\n")
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])

    # Settings
    rerankTopN = 2
    similarityTopK = 2
    loadDataPath = 'data'
    index_path = './storage'
    evaluateResponse = True

    llm_gpt35 = OpenAI(temperature=0, model='gpt-3.5-turbo')
    # llm_gpt4 = OpenAI(temperature=0, model='gpt-3.5-turbo')
    llm_gpt4 = OpenAI(temperature=0, model='gpt-4')

    # Augment the user's prompt with a question if there are no questions in the prompt
    user_prompt_contains_question = contains_question(userPrompt)
    userPromptQAugment = userPrompt
    if user_prompt_contains_question == False:
        augPrompt = "Here is a customer issue: \n\n" + f"'{userPrompt}'\n\n" + "Act as this customer. What are you trying to ask? Be concise. Generate the response as a question. Avoid 'why' questions."
        resp = llm_gpt35.complete(f"{augPrompt}")
        userPromptQAugment = userPrompt + "\n\n" + resp.text
        print(f"Question augmented user prompt:")
        print(f"{userPromptQAugment}\n\n")

    service_context = ServiceContext.from_defaults(
        llm=llm_gpt4,
        callback_manager=callback_manager
        )
    set_global_service_context(service_context)

    # Load index from chromadb
    index = loadIndex.main("./aiemailsupport_vectorstore/chromaDB", "bptm")
    print(index)

    if index is None:
        # Create and save vector store to chromadb and persist on disk volume
        index = createAndSaveIndex.main(loadDataPath, "./aiemailsupport_vectorstore/chromaDB","bptm")

    ## https://wandb.ai/ayush-thakur/llama-index-report/reports/Building-Advanced-Query-Engine-and-Evaluation-with-LlamaIndex-and-W-B--Vmlldzo0OTIzMjMy#setting-up-evaluation-using-llamaindex
    # rerank = SentenceTransformerRerank(
    #     model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=rerankTopN
    # )

    query_engine = index.as_query_engine(
        text_qa_template=EMAIL_SUPPORT_TEXT_QA_PROMPT,
        similarity_top_k=similarityTopK,
        # node_postprocessors=[rerank],
    )
    response = query_engine.query(userPromptQAugment)

    # Print info on llm inputs/outputs
    event_pairs = llama_debug.get_llm_inputs_outputs()
    print(event_pairs[0][0]) # Show what was sent to LLM
    print(f"\n\nAnswer: ")
    print(response)

    doesAnswerMatchSource = "NO"
    if (evaluateResponse):
        service_context35 = ServiceContext.from_defaults(llm=llm_gpt35)
        evaluator = ResponseEvaluator(service_context=service_context35)
        eval_result = evaluator.evaluate(response)
        print(str(eval_result)) # YES indicates the response was contructed from the source context well. NO indicates otherwise or it hallucinated 
        doesAnswerMatchSource = str(eval_result)

    # Refine answer if the LLM deviated from guardrails
    matches = find_matches(response.response, UNWANTED_SENTENCE_PHRASES)
    if matches:
        improveRespPrompt = f"Remove any text mentioning \"{matches}\" or similar:\n\n" + f"\"{response.response}\"\n\n Then, rewrite a sensible response." 
        improvedResp = llm_gpt35.complete(improveRespPrompt)
        print(f"\n\nImproved answer: ")
        print(improvedResp.text)
        improvedResp = improvedResp.text.strip('\'"')
        response.response = improvedResp

    if saveResults == '1':
        # Open the file in append mode ('a') and write the text
        with open("testResults.txt", "a") as file:
            file.write("Prompt:\n" + userPromptQAugment + "\n\n")
            file.write("Answer:\n" + response.response + "\n\n")
            file.write("Settings:\n" + "rerank top n: " + str(rerankTopN) + ", similarity top K:" + str(similarityTopK))
            file.write("\n\n---------------------------\n\n")

    return {
        'responseText': response.response,
        'originalPrompt': userPrompt,
        'userPromptQAugment': userPromptQAugment,
        'doesAnswerMatchSource': doesAnswerMatchSource == 'YES',
        # TODO return how many tokens were used, source vector(s), actual, complete prompt to chatgpt
    }

def lambda_handler(event, context):
    arg1 = event.get("prompt")
    print(arg1)
    output = main(arg1,"0")
    
    return {
        'statusCode': 200,
        'body': {
            'responseText': output['responseText'],
            'originalPrompt': output['originalPrompt'],
            'userPromptQAugment': output['userPromptQAugment'],
            'doesAnswerMatchSource': output['doesAnswerMatchSource'],
        }
    }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python callAI.py <prompt> <save results to txt file = 1, else = 0>")
    else:
        main(sys.argv[1], sys.argv[2])


