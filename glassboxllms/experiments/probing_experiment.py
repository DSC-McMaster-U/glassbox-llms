import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

from glassboxllms.primitives.probes import LinearProbe, NonLinearProbe
from utils import ActivationStore, evaluate_metrics


class ProbingExperiment:
    def __init__(self, 
                 model_name="gpt2", 
                 layer="transformer.h.6", 
                 dataset_name="truthful_qa", 
                 probe_type="linear", 
                 batch_size=8):
        # Load model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.eval()

        # Load dataset
        dataset = load_dataset(dataset_name)
        self.train_data = dataset["train"]
        self.test_data = dataset["validation"] if "validation" in dataset else dataset["test"]

        # Wrap in DataLoader
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size)

        # Activation store for chosen layer
        self.store = ActivationStore(self.model, layer=layer)

        # Probe type
        if probe_type == "linear":
            self.probe = LinearProbe()
        elif probe_type == "nonlinear":
            self.probe = NonLinearProbe(hidden_dim=128, num_layers=2)
        else:
            raise ValueError("Unsupported probe type")
        self.probe_type = probe_type

    def collect_activations(self, loader):
        activations, labels = [], []
        for batch in loader:
            texts = batch["text"]  # adjust depending on dataset field name
            y = batch["label"]     # adjust depending on dataset field name

            encodings = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**encodings)
                act = self.store.extract(outputs)  # your hook logic

            activations.append(act)
            labels.append(torch.tensor(y))

        return torch.cat(activations), torch.cat(labels)

    def run(self):
        # Collect activations
        X_train, y_train = self.collect_activations(self.train_loader)
        X_test, y_test = self.collect_activations(self.test_loader)

        # Train probe
        self.probe.fit(X_train, y_train)

        # Evaluate
        preds = self.probe.predict(X_test)
        results = evaluate_metrics(y_test, preds)

        # Directional vector (only for linear probe)
        direction = None
        if self.probe_type == "linear":
            direction = self.probe.get_weights()

        return {
            "metrics": results,
            "direction": direction
        }


# Usage
if __name__ == "__main__":
    experiment = ProbingExperiment(
        model_name="gpt2",
        layer="transformer.h.6",
        dataset_name="truthful_qa",
        probe_type="linear"
    )
    results = experiment.run()
    print("Results:", results["metrics"])
    if results["direction"] is not None:
        print("Directional Vector:", results["direction"])
