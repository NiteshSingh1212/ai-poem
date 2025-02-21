import os
import logging
from models.gpt2_model import PoetryGPT2
from data.download_dataset import download_poetry_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(
    model_name="gpt2",
    data_dir="poetry_data",
    output_dir="trained_model",
    epochs=3,
    batch_size=4
):
    """Train the poetry generation model"""
    try:
        # Download and prepare dataset
        logger.info("Preparing dataset...")
        train_file = download_poetry_dataset(data_dir)
        
        # Initialize model
        logger.info(f"Initializing {model_name} model...")
        model = PoetryGPT2(model_name=model_name)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Fine-tune the model
        logger.info("Starting fine-tuning process...")
        model.fine_tune(
            train_file=train_file,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size
        )
        
        logger.info(f"Training completed! Model saved to: {output_dir}")
        
        # Test the model
        test_prompt = "Write a poem about spring"
        logger.info("\nGenerating test poem...")
        poem = model.generate_poem(test_prompt)
        logger.info(f"\nTest Poem:\n{poem}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
