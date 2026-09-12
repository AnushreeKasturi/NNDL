# Model Inference Flow

## 1. Model choice
- Contract analysis endpoint selects `payload.model_name` or configured default.
- Supported baseline model IDs include:
  - `bert-base-uncased`
  - `nlpaueb/legal-bert-base-uncased`
  - `allenai/longformer-base-4096`

## 2. Chunking
- Contract text is split into overlapping chunks (`max_length`, `stride`) in `chunk_text`.
- This preserves clauses that cross chunk boundaries.

## 3. Chunk prediction
- Each chunk is tokenized and passed to the multi-label classifier.
- Sigmoid probabilities are produced for the 6 risk labels.

## 4. Document pooling
- Chunk probabilities are max-pooled per label.
- Final label set is thresholded from pooled probabilities.

## 5. Output
- Output JSON includes:
  - `scores` per label
  - `predicted_labels`
  - `chunk_count`
  - `threshold`

