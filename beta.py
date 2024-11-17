from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=200, min_length=100):
    """
    Summarize the given text using the BART summarization model.

    Parameters:
    - text: str, input text to summarize
    - max_length: int, maximum length of the summary
    - min_length: int, minimum length of the summary

    Returns:
    - str, the summarized text
    """
    try:
        # Generate the summary
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error summarizing text: {e}"

if __name__ == "__main__":
    print("=== Text Summarization ===")
    print("Paste your text below (press Enter twice to submit):")
    user_input = []
    while True:
        line = input()
        if line == "":  # Empty line ends input
            break
        user_input.append(line)

    # Combine all input lines into a single text block
    text = "\n".join(user_input)

    # Ensure the input text is not too long for the model
    max_token_length = 1024
    if len(text.split()) > max_token_length:
        print(f"Warning: The input text exceeds {max_token_length} tokens. Truncating to fit the model.")
        text = " ".join(text.split()[:max_token_length])

    # Summarize the text
    print("\nSummarizing your text...\n")
    summary = summarize_text(text)
    print("=== Summary ===")
    print(summary)
