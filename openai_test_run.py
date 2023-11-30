from openai import OpenAI
import time
import os
import sys

# Example:
# python3.10 openai_test_run.py "Give me all the important points and steps for Setting up an EC2 Instance"
def check_run(client, thread_id, run_id):
    cnt = 0;
    while True:
        # Refresh the run object to get the latest status
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )

        if run.status == "completed":
            messages = client.beta.threads.messages.list(
                thread_id=thread_id
            )
            return {
                "status": "completed",
                "userPrompt": messages.data[1].content[0].text.value,
                "assistantReply": messages.data[0].content[0].text.value
            }
        elif run.status == "expired":
            return {
                "status": "expired"
            }
        else:
            print(f"{cnt} OpenAI: Run is not yet completed. Waiting...")
            cnt = cnt + 1;
            time.sleep(1) 


def main(prompt):
    client = OpenAI()
    t0 = time.time();
    run = client.beta.threads.create_and_run(
        assistant_id="asst_fh5cjjxYy13QyhRux88vVxrQ",
        thread={
            "messages": [{"role": "user", "content": prompt,}]
        }
    )
    print(run)
    resp = check_run(client, run.thread_id, run.id)
    print(resp)
    t1 = time.time();
    print(f"Done with response: {t1-t0}\n");

    # Parse the JSON response
    # response_json = resp.json()

    # # Access the "id" parameter
    # status = response_json.get('status', None)

    # if status == "completed":
    #     print('success!')
    
    print("deleting thread")
    response = client.beta.threads.delete(run.thread_id)
    print(response)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python openai_test_run.py <prompt>")
    else:
        main(sys.argv[1])



# run_steps = client.beta.threads.runs.steps.list(
#     thread_id="thread_w40nlxa9dIeeXtwoPkcaTVcE",
#     run_id="run_Trxu38nhwmjitRbjFY8reKku"
# )
# print(run_steps)


# message = client.beta.threads.messages.retrieve(
#   message_id="msg_83EduEaWyb6yVjdAaHGwAHRc",
#   thread_id="thread_w40nlxa9dIeeXtwoPkcaTVcE",
# )
# print(message)

# response = client.beta.threads.delete("thread_w40nlxa9dIeeXtwoPkcaTVcE")
# print(response)

# client = OpenAI()
# run = client.beta.threads.runs.cancel(
#   thread_id="thread_qQ6sky7kTv7baGCQcxf8WxUf",
#   run_id="run_TwDEI2LR4jWYRvOEQzcb1bWC"
# )
# print(run)