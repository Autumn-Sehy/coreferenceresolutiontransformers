import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
import os
from datetime import datetime
from functools import lru_cache
import random
import multiprocessing


def load_data(filename, sample_fewer=0.25):
    """This method loads the data and allows us to sample fewer of the
    larger class the dataset is biased towards"""
    examples = []
    negative_examples = []
    positive_examples = []

    # I read how to do random sampling on some medium article at 2am
    # an issue with this, however, could be reproducibility
    # but it kinda feels lika a backwards monte carlo
    neg_sample_prob = sample_fewer if 'train' in filename else 1.0

    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                sections = line.strip().split('\t')
                label = int(sections[-1])

                should_include = True
                if label == 0 and 'train' in filename:
                    should_include = random.random() <= neg_sample_prob

                if should_include:
                    if '#' in sections[0]:
                        sentence1 = sections[2]
                        trigger_start1 = int(sections[3])
                        trigger_end1 = int(sections[4])
                        sentence2 = sections[13]
                        trigger_start2 = int(sections[14])
                        trigger_end2 = int(sections[15])
                    else:
                        sentence1 = sections[0]
                        trigger_start1 = int(sections[1])
                        trigger_end1 = int(sections[2])
                        sentence2 = sections[11]
                        trigger_start2 = int(sections[12])
                        trigger_end2 = int(sections[13])

                    example = {
                        'sentence1': sentence1,
                        'sentence2': sentence2,
                        'trigger1': (trigger_start1, trigger_end1),
                        'trigger2': (trigger_start2, trigger_end2),
                        'label': label
                    }

                    if label == 0:
                        negative_examples.append(example)
                    else:
                        positive_examples.append(example)

                    examples.append(example)

    except Exception as e:
        print(f"Error loading file {filename}: {e}")

    random.shuffle(examples)
    return examples


# I read about lru caching on another medium article last summer
# Seems to help with memory
@lru_cache(maxsize=1024)
def preprocess_sentence(sentence, trigger_start, trigger_end):
    TRIGGER_START = "<trigger>"
    TRIGGER_END = "</trigger>"

    words = sentence.split()
    words.insert(trigger_start, TRIGGER_START)
    words.insert(trigger_end + 2, TRIGGER_END)

    return ' '.join(words)


def create_features(example, tokenizer, max_length=512):
    """
    This function preprocesses the sentence.
    If I had time, this is likely where I'd have added sentiment analysis
    and emotion analysis.
    """
    sent1 = preprocess_sentence(
        example['sentence1'],
        example['trigger1'][0],
        example['trigger1'][1]
    )

    sent2 = preprocess_sentence(
        example['sentence2'],
        example['trigger2'][0],
        example['trigger2'][1]
    )

    encoded = tokenizer(
        sent1,
        sent2,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors=None
    )

    return {
        'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
        'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
        'labels': torch.tensor(example['label'], dtype=torch.long)
    }


class EventCoreferenceDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = [ex['label'] for ex in examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return create_features(self.examples[idx], self.tokenizer, self.max_length)


def calculate_metrics(preds, labels):
    """
    Evaluation~~~
    """
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    accuracy = accuracy_score(labels, preds)
    return {
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        },
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'micro_avg': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1
        },
        'accuracy': accuracy
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model with detailed metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Evaluating"):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # Forward pass with a limited set of returns
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            return_dict=True,
            output_hidden_states=False,
            output_attentions=False,
        )

        total_loss += outputs.loss.item()
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().tolist())  # .tolist() is faster than .numpy()
        all_labels.extend(batch['labels'].cpu().tolist())

    metrics = calculate_metrics(all_preds, all_labels)
    avg_loss = total_loss / len(loader)
    metrics['loss'] = avg_loss

    return metrics


def save_metrics(metrics, filename):
    """
    I do this out of habit.
    And so I don't have to copy and paste the metrics later.
    """
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)


def print_metrics(metrics):
    """
    Is this necessary? No. Not with the Json.
    But I like extra print statements and immediate results.
    """
    print("\nEvaluation Nation:")
    print("*************************")

    print("\nPer-class metrics:")
    print("**********************")
    for i, (p, r, f1, s) in enumerate(zip(
            metrics['per_class']['precision'],
            metrics['per_class']['recall'],
            metrics['per_class']['f1'],
            metrics['per_class']['support']
    )):
        print(f"Class {i}:")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall: {r:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  Support: {s}")

    print("\nMacro Avg:")
    print("********************")
    print(f"Precision: {metrics['macro_avg']['precision']:.4f}")
    print(f"Recall: {metrics['macro_avg']['recall']:.4f}")
    print(f"F1-score: {metrics['macro_avg']['f1']:.4f}")

    print("\nMicro Avg:")
    print("********************")
    print(f"Precision: {metrics['micro_avg']['precision']:.4f}")
    print(f"Recall: {metrics['micro_avg']['recall']:.4f}")
    print(f"F1-score: {metrics['micro_avg']['f1']:.4f}")

    print("\nOverall Metrics:")
    print("********************")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Loss: {metrics['loss']:.4f}")


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # I heard this might make it faster so I added it in ~yayyyy~
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results_dir = "training_results"
    os.makedirs(results_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    special_tokens = {'additional_special_tokens': ['<trigger>', '</trigger>']}
    tokenizer.add_special_tokens(special_tokens)

    print("Data is tokenized, loading data!")
    train_examples = load_data("event_pairs.train", sample_fewer=0.25)
    # this was mostly for speed, but sampling more still garners good results :)
    dev_examples = load_data("event_pairs.dev", sample_fewer=0.5)
    test_examples = load_data("event_pairs.test", sample_fewer=0.5)

    # This pre-creates datasets
    train_dataset = EventCoreferenceDataset(train_examples, tokenizer)
    dev_dataset = EventCoreferenceDataset(dev_examples, tokenizer)
    test_dataset = EventCoreferenceDataset(test_examples, tokenizer)

    # I also learned this from a 'what to do with uneven data' medium article
    # I like how the weight algorithm scales based on the data
    class_counts = np.bincount(train_dataset.labels)
    weights = np.sqrt(1.0 / class_counts)
    weights = weights / weights.sum()
    weights = weights * len(weights)

    sample_weights = [weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # Using fixed batch sizes
    train_batch_size = 16
    eval_batch_size = 32

    print(f"Using batch sizes: training={train_batch_size}, eval={eval_batch_size}")

    # the postdoc in charge of me at the lab taught me this
    num_workers = min(4, multiprocessing.cpu_count() // 2)
    print(f"Using {num_workers} dataloader workers")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        'nlpaueb/legal-bert-base-uncased',
        num_labels=2,
    )
    # Disable mean-resizing overhead when expanding the embedding matrix
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # I could have done more here to eek out better scores imo
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "key", "value", "dense"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    max_epochs = 5

    # I could have changed my learning rate again for slightly better results
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # I like doing warmup steps
    # It's like when I was a track athlete, prepping for my race
    # But instead it's the little program preparing for data
    total_steps = len(train_loader) * max_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print("Training!")
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    best_metrics = None
    best_f1 = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    training_info = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': []
    }

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        model.train()
        total_loss = 0
        epoch_steps = 0

        for batch in tqdm(train_loader, desc="Training"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            # Mixed precision training is for memory saving
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    return_dict=True,
                    output_hidden_states=False,
                    output_attentions=False
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            epoch_steps += 1

            # more memory saving
            if epoch_steps % 100 == 0:
                torch.cuda.empty_cache()

        avg_loss = total_loss / epoch_steps
        training_info['train_loss'].append(avg_loss)
        print(f"Avg loss is {avg_loss:.4f}")

        # if you clear the cache it helps keep eval from crashing
        torch.cuda.empty_cache()

        val_metrics = evaluate(model, dev_loader, device)
        training_info['val_loss'].append(val_metrics['loss'])
        training_info['val_f1'].append(val_metrics['macro_avg']['f1'])

        print_metrics(val_metrics)

        # saving metrics~~~
        epoch_metrics_file = os.path.join(run_dir, f"epoch_{epoch + 1}_metrics.json")
        save_metrics(val_metrics, epoch_metrics_file)

        # Saving the best model to a json so I can compare to other runs
        if val_metrics['macro_avg']['f1'] > best_f1:
            best_f1 = val_metrics['macro_avg']['f1']
            best_metrics = val_metrics
            model_save_dir = os.path.join(run_dir, 'best_model_lora')
            model.save_pretrained(model_save_dir)
            save_metrics(val_metrics, os.path.join(run_dir, 'best_model_metrics.json'))
            with open(os.path.join(run_dir, 'train_info.json'), 'w') as f:
                json.dump(training_info, f, indent=4)

            print("\nNew best model!")


    with open(os.path.join(run_dir, 'train_info.json'), 'w') as f:
        json.dump(training_info, f, indent=4)

    print("\nDone with training!")
    print("\nHere's the best metrics we had:")
    print_metrics(best_metrics)

    torch.cuda.empty_cache()
    test_metrics = evaluate(model, test_loader, device)
    print("\nTest set metrics:")
    print_metrics(test_metrics)
    save_metrics(test_metrics, os.path.join(run_dir, 'test_metrics.json'))


if __name__ == "__main__":
    main()
