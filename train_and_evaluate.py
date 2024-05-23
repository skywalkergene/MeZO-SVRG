from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from optimizer import MeZO_SVRG
from estimator import ZOGradientEstimator

# module and dataset
model_names = {
    'distilbert': 'distilbert-base-uncased',
    'gpt2': 'gpt2',
    'opt-2.7b': 'facebook/opt-2.7b',
    'opt-6.7b': 'facebook/opt-6.7b'
}

dataset_names = ['mnli', 'qnli', 'sst2', 'cola']


# load dataset and preprocess
def load_and_process_dataset(model_name, dataset_name):
    # load dataset
    dataset = load_dataset('glue', dataset_name)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_names[model_name])

    def tokenize_function(examples):
        if dataset_name == 'mnli':
            return tokenizer(examples['premise'], examples['hypothesis'], padding='max_length', truncation=True)
        elif dataset_name in ['qnli', 'sst2', 'cola']:
            return tokenizer(examples['sentence'], padding='max_length', truncation=True)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    # apply tokenization
    encoded_dataset = dataset.map(tokenize_function, batched=True)

    # transform type to torch.tensor
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # create DataLoader
    data_collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(encoded_dataset['train'], batch_size=32, shuffle=True, collate_fn=data_collator)
    valid_loader = DataLoader(encoded_dataset['validation'], batch_size=32, collate_fn=data_collator)

    return train_loader, valid_loader, tokenizer


# example
model_name = 'distilbert'
dataset_name = 'sst2'

train_loader, valid_loader, tokenizer = load_and_process_dataset(model_name, dataset_name)

# load model
model = AutoModelForSequenceClassification.from_pretrained(model_names[model_name], num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

Gradient_Estimator = ZOGradientEstimator(model, device, train_loader, seed=0)
optimizer = MeZO_SVRG(model.parameters(), Gradient_Estimator, lr1=0.01, lr2=0.001, q=10, mu=0.01, batch_size=32)

# loss function
def compute_loss(model, batch):
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    labels = batch['label'].to(device)
    outputs = model(**inputs, labels=labels)
    return outputs.loss

# train and evaluate
def train_and_evaluate(model, train_loader, valid_loader, optimizer, epochs=10):
    for epoch in range(epochs):
        # train
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            # ban autograd
            with torch.no_grad():

                # must assure param.grad is set correctly
                loss = compute_loss(model, batch)
                # compute gradient and update parameters
                optimizer.step(batch)

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}')

        # evaluate
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                loss = compute_loss(model, batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valid_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}')


# execute
train_and_evaluate(model, train_loader, valid_loader, optimizer)
