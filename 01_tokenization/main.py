import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("My name is Tano.")
print("Tokenized string:", tokens)
print("Decoded string:", enc.decode(tokens))