import sys
import json
import nltk
import time
nltk.data.path.append("./nltk_data")
from llama_index.llms import OpenAI
from llama_index import (
    ServiceContext,
    set_global_service_context,
)
# from llama_index.indices.postprocessor import SentenceTransformerRerank
# Use "git submodule update --init --recursive" to update submodule to latest commit in aiemailsupport_vectorstore repo
from aiemailsupport_vectorstore import loadIndex
from llama_index.evaluation import FaithfulnessEvaluator
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from email_support_QA_prompts import (
    EMAIL_SUPPORT_TEXT_QA_PROMPT,
    UNWANTED_SENTENCE_PHRASES
)
from nlp_functions import (
    find_matches,
    contains_question,
    process_email_body,
    )

def main(userPrompt, saveResults, collectionName):
    try:
        # print(sys.path)
        # print(f"You passed the prompt:")
        # print(userPrompt + "\n\n")
        # llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        # callback_manager = CallbackManager([llama_debug])

        # Settings
        rerankTopN = 2
        similarityTopK = 2
        # collectionName = 'bptm'
        evaluateResponse = False
        printOutTime = False

        t0 = time.time();

        llm_gpt35 = OpenAI(temperature=0, model='gpt-3.5-turbo')
        llm_gpt4 = OpenAI(temperature=0, model='gpt-4-1106-preview')
        #llm_gpt4 = OpenAI(temperature=0, model='gpt-4')
        if printOutTime:
            t1 = time.time();
            print(f"Done defining openai models: {t1-t0}\n");

        # Try to remove signatures from email body
        try:
            userPrompt = process_email_body(userPrompt) # userPrompt can be a string or a list of strings
        except Exception:
            pass

        # Augment the user's prompt with a question if there are no questions in the prompt
        if printOutTime:
            t1_1 = time.time();
            print(f"Start augmenting prompt: {t1_1-t0}\n");
        user_prompt_contains_question = contains_question(userPrompt)
        userPromptQAugment = userPrompt
        if user_prompt_contains_question == False:
            augPrompt = "Here is a customer issue: \n\n" + f"'{userPrompt}'\n\n" + "Act as this customer. What are you trying to ask? Be concise. Generate the response as a question. Avoid 'why' questions."
            resp = llm_gpt35.complete(f"{augPrompt}")
            userPromptQAugment = userPrompt + "\n\n" + resp.text
            # print(f"Question augmented user prompt:")
            # print(f"{userPromptQAugment}\n\n")
        if printOutTime:
            t2 = time.time();
            print(f"Done augmenting prompt: {t2-t0}\n");

        service_context = ServiceContext.from_defaults(
            llm=llm_gpt4,
            # callback_manager=callback_manager
            )
        set_global_service_context(service_context)

        # Load index from chromadb
        if printOutTime:
            t2_1 = time.time();
            print(f"Start loading index: {t2_1-t0}\n");
        index = loadIndex.main(collectionName)
        if printOutTime:
            t3 = time.time();
            print(f"Done loading index: {t3-t0}\n");
        # print(index)

        if index is None:
            raise ValueError('No index was found. Please upload a document first.')
        # https://wandb.ai/ayush-thakur/llama-index-report/reports/Building-Advanced-Query-Engine-and-Evaluation-with-LlamaIndex-and-W-B--Vmlldzo0OTIzMjMy#setting-up-evaluation-using-llamaindex
        # rerank = SentenceTransformerRerank(
        #     model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=rerankTopN
        # )
        if printOutTime:
            t4 = time.time();
            print(f"Start query engine def: {t4-t0}\n");
        query_engine = index.as_query_engine(
            text_qa_template=EMAIL_SUPPORT_TEXT_QA_PROMPT,
            similarity_top_k=similarityTopK,
            # node_postprocessors=[rerank],
        )

        if printOutTime:
            t4_1 = time.time();
            print(f"Done query engine def: {t4_1-t0}\n");
            t4_2 = time.time();
            print(f"Start getting response: {t4_2-t0}\n");
        response = query_engine.query(userPromptQAugment)
        if printOutTime:
            t5 = time.time();
            print(f"Done getting response: {t5-t0}\n");

        # Print info on llm inputs/outputs
        # event_pairs = llama_debug.get_llm_inputs_outputs()
        # print(event_pairs[0][0]) # Show what was sent to LLM
        # print(f"\n\nAnswer: ")
        # print(response)

        if printOutTime:
            t5_1 = time.time();
            print(f"Start processing response: {t5_1-t0}\n");
        doesAnswerMatchSource = "NO"
        if (evaluateResponse):
            service_context35 = ServiceContext.from_defaults(llm=llm_gpt35)
            evaluator = FaithfulnessEvaluator(service_context=service_context35)
            eval_result = evaluator.evaluate_response(response=response)
            # print(str(eval_result.passing)) # YES indicates the response was contructed from the source context well. NO indicates otherwise or it hallucinated 
            doesAnswerMatchSource = str(eval_result)

        # Refine answer if the LLM deviated from guardrails
        matches = find_matches(response.response, UNWANTED_SENTENCE_PHRASES)
        if matches:
            improveRespPrompt = f"Remove any text mentioning \"{matches}\" or similar:\n\n" + f"\"{response.response}\"\n\n Then, rewrite a sensible response." 
            improvedResp = llm_gpt35.complete(improveRespPrompt)
            # print(f"\n\nImproved answer: ")
            # print(improvedResp.text)
            improvedResp = improvedResp.text.strip('\'"')
            response.response = improvedResp
        if printOutTime:
            t6 = time.time();
            print(f"Done processing  response: {t6-t0}\n");

        if saveResults == '1':
            # Open the file in append mode ('a') and write the text
            with open("testResults.txt", "a") as file:
                file.write("Prompt:\n" + userPromptQAugment + "\n\n")
                file.write("Answer:\n" + response.response + "\n\n")
                file.write("Settings:\n" + "rerank top n: " + str(rerankTopN) + ", similarity top K:" + str(similarityTopK))
                file.write("\n\n---------------------------\n\n")

        return {
            'success': True,
            'responseText': response.response,
            'originalPrompt': userPrompt,
            'userPromptQAugment': userPromptQAugment,
            'doesAnswerMatchSource': doesAnswerMatchSource == 'YES',
            # TODO return how many tokens were used, source vector(s), actual, complete prompt to chatgpt
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def lambda_handler(event, context):
    prompt = event.get('prompt', '')  # Defaulting to an empty string if 'prompt' key doesn't exist
    collection = event.get('collection', '') 
    # print(prompt)
    output = main(prompt, "0", collection)
    
    return {
        'statusCode': 200,
        'body': {
            'success': True,
            'responseText': output['responseText'],
            'originalPrompt': output['originalPrompt'],
            'userPromptQAugment': output['userPromptQAugment'],
            'doesAnswerMatchSource': output['doesAnswerMatchSource'],
        }
    }

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python callAI.py <prompt> <save results to txt file = 1, else = 0> <collection name>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])


