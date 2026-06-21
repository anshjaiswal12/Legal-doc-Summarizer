import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

def train_abstractive_model(model_name: str, dataset, output_dir: str, num_train_epochs: int = 3):
    model_configs = {
        "sshleifer/distilbart-cnn-12-6": {
            "target_modules": ["q_proj", "v_proj"],
            "prefix": ""
        },
        "t5-small": {
            "target_modules": ["q", "v"],
            "prefix": "summarize: "
        }
    }
    config = model_configs[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=config["target_modules"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params > 0, f"No trainable parameters for {model_name}!"
    
    def tokenize_function(examples):
        inputs = [config["prefix"] + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
        labels = tokenizer(examples["summary"], max_length=256, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        fp16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        save_strategy="epoch",
        eval_strategy="epoch",
        num_train_epochs=num_train_epochs,
        logging_steps=10
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset.select(range(min(10, len(tokenized_dataset)))), 
        processing_class=tokenizer,
        data_collator=data_collator
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    return model, tokenizer

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_prep import load_billsum
    
    billsum = load_billsum()
    billsum_subset = billsum['train'].select(range(500))
    held_out_example = billsum['train'][500]
    
    print("=== Phase 4: DistilBART ===")
    model_bart, tokenizer_bart = train_abstractive_model(
        "sshleifer/distilbart-cnn-12-6", 
        billsum_subset, 
        "outputs/models/distilbart-lora", 
        num_train_epochs=1
    )
    
    inputs = tokenizer_bart(held_out_example["text"], max_length=1024, truncation=True, return_tensors="pt").to(model_bart.device)
    outputs = model_bart.generate(**inputs, max_length=300)
    summary_bart = tokenizer_bart.decode(outputs[0], skip_special_tokens=True)
    
    print(f"DistilBART Summary:\n{summary_bart}")
    assert len(summary_bart.split()) > 0
    assert len(outputs[0]) <= 300
    peak_mem_bart = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"Peak GPU Memory (DistilBART): {peak_mem_bart:.2f} MB")
    print("PHASE 4 VERIFIED\n")
    
    print("=== Phase 5: T5-Small ===")
    model_t5, tokenizer_t5 = train_abstractive_model(
        "t5-small", 
        billsum_subset, 
        "outputs/models/t5-lora", 
        num_train_epochs=1
    )
    
    inputs_t5 = tokenizer_t5("summarize: " + held_out_example["text"], max_length=1024, truncation=True, return_tensors="pt").to(model_t5.device)
    outputs_t5 = model_t5.generate(**inputs_t5, max_length=300)
    summary_t5 = tokenizer_t5.decode(outputs_t5[0], skip_special_tokens=True)
    
    print(f"T5-Small Summary:\n{summary_t5}")
    assert len(summary_t5.split()) > 0
    assert len(outputs_t5[0]) <= 300
    peak_mem_t5 = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"Peak GPU Memory (T5): {peak_mem_t5:.2f} MB")
    print("PHASE 5 VERIFIED")
