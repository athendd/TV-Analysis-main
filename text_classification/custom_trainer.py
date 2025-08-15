from torch import nn
from transformers import Trainer 

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,  **kwargs):
        labels = inputs.get("labels").float().to(self.device)
        inputs["labels"] = labels
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def set_device(self, device):
        self.device = device
