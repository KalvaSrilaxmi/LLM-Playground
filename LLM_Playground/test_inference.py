# Step 1: Install the required package
from huggingface_hub import InferenceClient

# Step 2: Import required modules
from getpass import getpass
import textwrap  # It makes the Input/Output in a readable format

# Step 3: Get your hugging face token
HF_TOKEN = getpass("Enter your Hugging Face API Token: ")

# Step 4:Create an inference client
Client = InferenceClient(  # InferenceClient - Python class from the huggingface-hub library that lets us send requests to models hosted on the
    model="HuggingFaceH4/zephyr-7b-alpha",
    token=HF_TOKEN,
)

# Step 5:Prepare your conversation (as a chat)
user_question = input("Enter your question: ")

messages = [
    {"role": "system", "content": "You are a helpful and knowledgeble assistant."},  # system - Sets the behaviour or identity of the assistant
    {"role": "user", "content": user_question},  # User - The input/ question from the human
]

# Step 6: Send the chat message
response = Client.chat_completion(
    messages=messages, max_tokens=200
)  # method that takes a chat history and generates a reply
# maximum number of tokens (words/subwords) the model can generate

# Step 7: Display the answer
print("\nAnswer:")
print(textwrap.fill(response.choices[0].message.content.strip(), width=80))


