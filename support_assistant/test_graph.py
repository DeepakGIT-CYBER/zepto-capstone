from graph import ask_zepto

response = ask_zepto("What is the delivery fee?")

print("\nANSWER:")
print(response.answer)

print("\nSOURCES:")
print(response.sources)

print("\nCONFIDENCE:")
print(response.confidence)