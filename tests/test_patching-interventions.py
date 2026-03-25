import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from glassboxllms.instrumentation.patching import PatchingExperiment # Ensure this matches your filename

def run_verification():
    print("--- Loading Model (GPT-2) ---")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Standardize for the class
    model.tokenizer = tokenizer 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    source = "When Mary and John went to the store, John gave a drink to Mary"
    corrupted = "When Mary and John went to the store, John gave a drink to John"
    
    target = " Mary"

    print(f"--- Running Patching Experiment on Layer 0 MLP ---")
    exp = PatchingExperiment(
        model=model,
        unit="transformer.h.0.mlp", 
        source_prompt=source,
        corrupted_prompt=corrupted,
        target_token_str=target
    )

    try:
        results = exp.run()
        
        print("\n--- Results ---")
        print(f"Unit Tested: {results['unit']}")
        print(f"Indirect Effect Score: {results['Indirect Effect']:.4f}")
        
        if results['Indirect Effect'] != 0:
            print("\nSuccess")
        else:
            print("\nEffect size is 0.")
            
    except NotImplementedError:
        print("\nFailure")

if __name__ == "__main__":
    run_verification()