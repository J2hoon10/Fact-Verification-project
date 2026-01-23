import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MultiHopVerifier(nn.Module):
    # Changed default model_name to "bert-base-cased"
    def __init__(self, model_name="bert-base-cased", num_labels=3):
        super(MultiHopVerifier, self).__init__()

        # Transformer Encoder: Standard BERT (Case-sensitive)
        self.encoder = AutoModel.from_pretrained(model_name)  # 🔸 Encoder

        # Multi-evidence Aggregation (Self-Attention Layer)
        # This links evidences by calculating attention scores between them
        self.attention_aggregator = nn.MultiheadAttention(    # 🔸 Multi-evidence aggregator
            embed_dim=self.encoder.config.hidden_size,
            num_heads=8,
            batch_first=True
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        # 1. Encode Claim + All Evidence
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)  # 🔸 Encoder

        # 2. Extract [CLS] representing the global relationship
        # Standard BERT uses the first token [CLS] for classification tasks
        cls_output = outputs.last_hidden_state[:, 0, :] # Shape: [Batch, Hidden_Size]

        # 3. Multi-evidence Aggregation logic
        # We treat the CLS output as the query to focus on the reasoning chain
        query = cls_output.unsqueeze(1)
        aggregated_feat, _ = self.attention_aggregator(query, query, query)

        # 4. Final Verdict (SUPPORTS, REFUTES, or NOT ENOUGH INFO)
        logits = self.classifier(aggregated_feat.squeeze(1))
        return logits

# --- IMPORTANT: Tokenizer Update ---
# When using bert-base-cased, you MUST use the matching tokenizer:
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


