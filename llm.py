from transformers import GPTNeoForCausalLM, AutoTokenizer

class LocalLLM:
    def __init__(self):
        """
        Initialize the GPT-Neo model and tokenizer.
        Choose the model size based on system resources.
        """
       
        model_name = "EleutherAI/gpt-neo-125M" 
        
        
        print(f"Loading model: {model_name}")
        try:
            self.model = GPTNeoForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("Model and tokenizer loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_response(self, prompt, max_length=150):
        """
        Generate a response to the given prompt using the loaded model.

        Args:
            prompt (str): The input prompt for the model.
            max_length (int): The maximum length of the generated response.

        Returns:
            str: The generated response from the model.
        """
        try:
            # Tokenize the input prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Generate a response using the model
            outputs = self.model.generate(
                inputs.input_ids, 
                max_length=max_length, 
                temperature=0.7, 
                num_return_sequences=1
            )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "An error occurred while generating a response."

# # Test the LocalLLM class
# if __name__ == "__main__":
#     prompt = "What are the advantages of electric vehicles?"
#     llm = LocalLLM()
#     response = llm.generate_response(prompt)
#     print(f"Response: {response}")
