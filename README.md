# Author: Wei Luo(罗威)
# Pre-trained Large Language Models for Financial Sentiment Analysis.
This project contains codes for the research paper title "Pre-trained Large Language Models for Financial Sentiment Analysis". Authors: Wei Luo, Dihong Gong.

## Environment setup
- `hostfile.txt` should contain the IP addresses for distributed training, one line per IP.
- `env_a800.sh` should contain all the additional custom environment variables for running the codes.

## Data preprocessing
- Split the train, val and test sets for the PhraseBank dataset: `process_financial_phrasebank.py`

## Training
- May need to tune some parameters accordingly.
- Execute the training script with `./train.sh`

## Testing
- Execute the testing script with `torchrun test.py`

## Results
Methods | Accuracy
--- | ---
LSTM | 0.71
LSTM with ELMo | 0.75
ULMFit | 0.83
LPS | 0.71
HSC | 0.71
FinBERT | 0.84
`Ours (Few-shots)` | `0.677`
`Ours (Further Pretraining)` | `0.712`
`Ours (SFT)` | `0.894`
