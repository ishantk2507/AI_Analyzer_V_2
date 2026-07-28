# from llama_cpp import Llama
# from pathlib import Path

# MODEL = Path("models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf").resolve()

# print(MODEL)
# print(MODEL.exists())

# llm = Llama(
#     model_path=str(MODEL),
#     n_ctx=2048,
#     n_gpu_layers=-1,
#     verbose=True,
# )

# print("Loaded!")

# out = llm.create_chat_completion(
#     messages=[
#         {"role": "user", "content": "Say hello"}
#     ],
#     max_tokens=20,
# )

# print(out)

# from llama_cpp import LlamaGrammar

# g = LlamaGrammar.from_string(r'''
# root ::= object
# object ::= "{" "\"action\"" ":" "\"think\"" "}"
# ''')

# print(g)


from llama_cpp import Llama, LlamaGrammar

llm = Llama(
    model_path="models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
    verbose=True,
)

grammar = LlamaGrammar.from_string(r'''
root ::= "hello"
''')

response = llm.create_completion(
    prompt="Repeat exactly hello.",
    grammar=grammar,
    max_tokens=5,
)

print(response)