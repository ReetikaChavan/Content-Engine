import logging
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from document_parser import extract_text_from_pdf
from embeddings import generate_embeddings
from vector_store import VectorStore
from llm import LocalLLM


vector_store = VectorStore(dim=384)  
llm = LocalLLM()


tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")


tokenizer.pad_token = tokenizer.eos_token

# Function to split long documents into smaller chunks
def split_into_chunks(text, max_length=4096):
    """
    Splits the text into chunks that fit within the model's token limit.
    """
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return chunks

def generate_response(query, context):
    """
    Generate a response to the query based on the context.
    """
    chunks = split_into_chunks(context)
    response = ""
    
    for chunk in chunks:
        try:
            chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)

            # Encode the context and query together
            encoding = tokenizer.encode_plus(
                f"Answer the following question based on the context: {chunk_text}\n\nQuestion: {query}",
                return_tensors="pt",
                padding=True,  
                truncation=True,  
                max_length=1024  
            )

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            # Generate output (ensure max tokens and temperature are set appropriately)
            outputs = model.generate(
                input_ids, 
                attention_mask=attention_mask, 
                max_length=1000,  
                do_sample=True, 
                temperature=0.7, 
                max_new_tokens=100  
            )

            response += tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"An error occurred while processing your query: {str(e)}"

    return response

def process_documents(pdf_paths):
    """
    Extract text, generate embeddings, and store in the vector store.
    """
    texts = []
    embeddings = []

    for pdf_path in pdf_paths:
        try:
            # Extract text from PDF
            text = extract_text_from_pdf(pdf_path)
            texts.append(text)

            # Generate embeddings for the extracted text
            embedding = generate_embeddings(text)
            embeddings.append(embedding)
        except Exception as e:
            logging.error(f"Error processing PDF: {pdf_path}, {e}")

    # Add generated embeddings to the vector store
    vector_store.add_vectors(embeddings, texts)

def query_documents(query):
    """
    Process the query by generating embeddings and querying the vector store for relevant documents.
    """
    # Generate embeddings for the query
    query_embedding = generate_embeddings(query)

    # Retrieve top-2 closest documents from the vector store
    results = vector_store.query(query_embedding, k=2)

    # Filter documents based on relevance threshold
    relevance_threshold = 0.7
    filtered_results = [result for result in results if result[1] >= relevance_threshold]

    # Construct context from the relevant documents
    context = ""
    for result in filtered_results:
        # Truncate each document text further to fit model's context
        max_doc_length = 1000  
        truncated_text = result[0][:max_doc_length]
        context += truncated_text + "\n" 

    # Truncate the overall context if necessary
    max_context_length = 512  
    context = context[:max_context_length]

    # Generate response based on the query and the context
    response = generate_response(query, context)
    return response

# Example usage for debugging:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # List of PDF paths to process
    pdf_paths = ["pdfs/goog-10-k-2023.pdf", "pdfs/tsla-20231231-gen.pdf", "pdfs/uber-10-k-2023.pdf"]
    
    # Process the documents to generate embeddings and add to the vector store
    process_documents(pdf_paths)
    print("Documents processed successfully.")
