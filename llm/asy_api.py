import os
import asyncio
from openai import AsyncOpenAI
from dotenv import dotenv_values
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
config = dotenv_values(env_path)



api_key = config.get("OURS_KEY")
api_url = config.get("OURS_URL")

# Create async client instance
client = AsyncOpenAI(
    # If environment variables are not configured, replace the following line with: api_key="sk-xxx",
    api_key=api_key,
    base_url=api_url
)

# Define async task list
import time

MAX_CONCURRENT_TASKS = 100
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

async def task(question, model, system_prompt=None):
    """
    Single question task: send question, get answer, calculate time taken.

    Args:
        question (str): The question to be answered.
        model (str): The model to be used.
        system_prompt (str): The system prompt.

    Returns:
        str: The answer from the model.
    """
    async with semaphore:
        start_time = time.time()

        # Dynamically construct messages based on whether system_prompt exists
        messages = []
        answer = ""
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        # print(messages)

        response = await client.chat.completions.create(
            messages=messages,
            model=model,
        )

        end_time = time.time()
        elapsed = end_time - start_time

        if model == "deepseek-r1":
            reasoning_output = response.choices[0].message.reasoning_content
            output = response.choices[0].message.content

            if '<answer>' in output:
                answer = f"""
                <think>
                {reasoning_output}
                </think>
                {output}
                """
            else:
                answer = f"""
                <think>
                {reasoning_output}
                </think>
                <answer>
                {output}
                </answer>
                """
        else:
            answer = response.choices[0].message.content
        print(f"Received answer: {answer}...")  # Only show first 100 chars to prevent screen flooding
        print(f"Time taken for this question: {elapsed:.2f} seconds")

    return answer, elapsed



async def async_get_answer(questions, model, system_prompt=None):
    """
    Batch questioning, returns list of all answers and prints overall timing statistics.
    """
    if isinstance(questions, str):
        questions = [questions]

    start_all = time.time()

    tasks = [task(q, model, system_prompt=system_prompt) for q in questions]
    outputs = await asyncio.gather(*tasks,  return_exceptions=True)

    answers = [item[0] for item in outputs]
    elapsed_times = [item[1] for item in outputs]

    end_all = time.time()
    total_time = end_all - start_all
    avg_time = sum(elapsed_times) / len(elapsed_times)

    print("\n==== Timing Summary ====")
    for i, t in enumerate(elapsed_times):
        print(f"Question {i+1}: {t:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per question: {avg_time:.2f} seconds")

    return answers, elapsed_times

    
    
if __name__ == '__main__':
    # # Set event loop policy
    questions = ["Who are you?", "Which model are you?"]
    answers = asyncio.run(async_get_answer(questions, model="llama-3.1-8b-instruct"))

    print(answers)