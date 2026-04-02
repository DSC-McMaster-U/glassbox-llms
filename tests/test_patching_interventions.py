import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from glassboxllms.instrumentation.patching import PatchingExperiment

class DemoPatchingExperiment(PatchingExperiment):
    def _calculate_logit_difference(self, clean_logits, patched_logits):
        """Calculates the raw logit shift for the target token."""
        # Use a leading space for the target token (standard for GPT-2)
        target_str = f" {self.target_token_str.strip()}"
        target_id = self.tokenizer.encode(target_str)[-1]
        
        # Get logits for the last token in the sequence
        clean_val = clean_logits[0, -1, target_id]
        patched_val = patched_logits[0, -1, target_id]
        
        return (patched_val - clean_val).item()

    def run(self):
        target_module = self._resolve_unit_to_module()
        device = next(self.model.parameters()).device
        
        clean_tokens = self.tokenizer(self.source_prompt, return_tensors="pt").input_ids.to(device)
        corrupted_tokens = self.tokenizer(self.corrupted_prompt, return_tensors="pt").input_ids.to(device)

        activation_cache = {}

        def cache_hook(module, input, output):
            activation_cache['clean_state'] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
            return output

        handle_clean = target_module.register_forward_hook(cache_hook)
        with torch.no_grad():
            clean_out = self.model(clean_tokens)
        handle_clean.remove()

        def patch_hook(module, input, output):
            clean_data = activation_cache['clean_state']
            if isinstance(output, tuple):
                return (clean_data,) + output[1:]
            return clean_data

        handle_patch = target_module.register_forward_hook(patch_hook)
        with torch.no_grad():
            patched_out = self.model(corrupted_tokens).logits
        handle_patch.remove()

        indirect_effect = self._calculate_logit_difference(clean_out.logits, patched_out)
        return {"unit": self.unit, "Indirect Effect": indirect_effect}

def run_presentation_demo():
    print("--- Initializing Glassbox-LLM Demo (GPT-2) ---")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.tokenizer = tokenizer
    model.eval()

    source = "When Mary and John went to the store, John gave a drink to Mary"
    corrupted = "When Mary and John went to the store, John gave a drink to John"
    target = "Mary"

    print(f"\nPrompt: {corrupted}")
    print(f"Targeting Causal Necessity of: '{target}'")
    
    effects = []
    layers = list(range(12))

    for i in layers:
        unit_path = f"transformer.h.{i}.mlp"
        exp = DemoPatchingExperiment(model, unit_path, source, corrupted, target)
        res = exp.run()
        score = res['Indirect Effect']
        effects.append(score)
        
        print(f"Layer {i:<2} | Logit Shift: {score:>8.4f}")

    # --- Plotting Section ---
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Red for positive effect (restoration), Blue for negative
    colors = ["#c44e52" if e > 0 else "#4c72b0" for e in effects]
    
    ax.bar(layers, effects, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Indirect Effect (Δ P(Mary))")
    ax.set_title(f'Causal Patching: Which MLP layers recover P("{target}") in the IOI task?')
    ax.set_xticks(layers)
    ax.axhline(0, color="black", linewidth=0.8)
    
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_presentation_demo()