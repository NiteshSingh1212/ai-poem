from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

def train_model():
    print("Starting training process...")
    
    # Initialize model and tokenizer
    print("Loading model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')  # Using the smallest GPT-2 model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Add pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Sample poems for training
    poems = [
        "Roses are red, violets are blue, sugar is sweet, and so are you.",
        "Two roads diverged in a yellow wood, and sorry I could not travel both.",
        "Because I could not stop for Death, He kindly stopped for me.",
        "Hope is the thing with feathers that perches in the soul.",
    ]
    
    # Prepare the poems
    print("Preparing training data...")
    inputs = tokenizer(poems, padding=True, truncation=True, return_tensors="pt")
    
    # Training settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    
    # Training loop
    print("Training the model...")
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    # Save the model
    print("Saving model...")
    os.makedirs("trained_model", exist_ok=True)
    model.save_pretrained("trained_model")
    tokenizer.save_pretrained("trained_model")
    
    # Generate a test poem
    print("\nGenerating test poem...")
    prompt = "Write a poem about spring:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=100,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=1
    )
    
    generated_poem = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    print(f"\nTest poem:\n{generated_poem}")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    train_model()
