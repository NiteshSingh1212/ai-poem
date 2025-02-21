import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoetryGPT2:
    def __init__(self, model_name: str = "gpt2", model_dir: Optional[str] = None):
        """Initialize the model with either a pre-trained model or a fine-tuned one"""
        self.model_name = model_name
        self.model_dir = model_dir or model_name
        
        # Load model and tokenizer
        if os.path.exists(self.model_dir):
            logger.info(f"Loading fine-tuned model from {self.model_dir}")
            self.model = GPT2LMHeadModel.from_pretrained(self.model_dir)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)
        else:
            logger.info(f"Loading pre-trained model {model_name}")
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Set up padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

    def prepare_dataset(self, text_file: str) -> TextDataset:
        """Prepare dataset for training"""
        logger.info(f"Preparing dataset from {text_file}")
        return TextDataset(
            tokenizer=self.tokenizer,
            file_path=text_file,
            block_size=128
        )

    def fine_tune(self, train_file: str, output_dir: str, epochs: int = 3, batch_size: int = 4):
        """Fine-tune the model on poetry data"""
        logger.info("Starting fine-tuning process")
        
        # Prepare dataset
        train_dataset = self.prepare_dataset(train_file)
        
        # Prepare data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=100,
            save_total_limit=2,
            logging_steps=50,
            logging_dir=os.path.join(output_dir, "logs"),
            overwrite_output_dir=True,
            warmup_steps=100,
            evaluation_strategy="steps",
            eval_steps=100
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset
        )
        
        # Train the model
        logger.info("Training started...")
        trainer.train()
        
        # Save the model
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Update model directory
        self.model_dir = output_dir

    def generate_poem(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """Generate a poem using the model"""
        # Encode the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and return the generated text
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
