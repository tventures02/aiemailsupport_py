import callAI
import os
from email_support_QA_prompts import SYSTEM_PROMPT

def main():
    # Get user confirmation
    user_input = input("Running this script will cost OpenAI credits proportional to the number of test prompts in 'testPrompts.txt'. Do you want to run this test? Y/N: ").strip().upper()

    if user_input == 'Y':
        # Check if the file exists and then delete it
        if os.path.exists('testResults.txt'):
            os.remove('testResults.txt')

        with open("testPrompts.txt", "r") as file:
            # Read the whole file into a single string
            content = file.read()
            
            # Split the string by the "--" delimiter
            testPrompts = content.split("--")

            # Loop through each testPrompt and process it
            for testPrompt in testPrompts:
                # Strip removes any leading/trailing whitespace
                testPrompt = testPrompt.strip()
                
                # If there's actual content in the testPrompt, process it
                if testPrompt:
                    callAI.main(testPrompt,'1')
            with open("testResults.txt", "a") as file:
                file.write("\n\nSYS PROMPT:\n" + SYSTEM_PROMPT)

    elif user_input == 'N':
        print("Script was not run.")
    else:
        print("Invalid input. Exiting...")

if __name__ == "__main__":
    main()
