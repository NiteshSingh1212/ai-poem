from datasets import load_dataset
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_poetry_dataset(output_dir="poetry_data"):
    """Download and prepare poetry dataset for fine-tuning"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load poetry dataset from Hugging Face
        logger.info("Downloading poetry dataset...")
        dataset = load_dataset("arthurflor23/poetry", split="train")
        
        # Prepare data for fine-tuning
        poetry_texts = []
        for item in dataset:
            # Format each poem with title and content
            formatted_poem = f"Title: {item['title']}\n\n{item['content']}\n\n"
            poetry_texts.append(formatted_poem)
        
        # Save the processed data
        train_file = os.path.join(output_dir, "poetry_train.txt")
        with open(train_file, "w", encoding="utf-8") as f:
            f.write("\n".join(poetry_texts))
        
        logger.info(f"Successfully downloaded and prepared {len(poetry_texts)} poems")
        logger.info(f"Data saved to: {train_file}")
        return train_file
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    download_poetry_dataset()
