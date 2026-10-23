import easyocr
import os
from pathlib import Path
import argparse
from datetime import datetime

class ImageTextExtractor:
    def __init__(self, languages=['en']):
        """
        Initialize the EasyOCR reader with specified languages
        
        Args:
            languages (list): List of language codes (e.g., ['en', 'es', 'fr'])
        """
        print("Initializing EasyOCR reader...")
        self.reader = easyocr.Reader(languages)
        print("EasyOCR reader initialized successfully!")
    
    def extract_text_from_image(self, image_path):
        """
        Extract text from a single image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Extracted text from the image
        """
        try:
            print(f"Processing image: {image_path}")
            
            # Read text from image
            results = self.reader.readtext(image_path)
            
            # Extract only the text content
            extracted_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter out low-confidence detections
                    extracted_text.append(text)
            
            return '\n'.join(extracted_text)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return ""
    
    def save_text_to_file(self, text, output_path):
        """
        Save extracted text to a file
        
        Args:
            text (str): Text to save
            output_path (str): Path where to save the text file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Text saved to: {output_path}")
        except Exception as e:
            print(f"Error saving text to file: {str(e)}")
    
    def process_single_image(self, image_path, output_path=None):
        """
        Process a single image and save extracted text
        
        Args:
            image_path (str): Path to the image file
            output_path (str): Optional path for output file
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found!")
            return
        
        # Generate output filename if not provided
        if output_path is None:
            image_name = Path(image_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{image_name}_extracted_text_{timestamp}.txt"
        
        # Extract text
        extracted_text = self.extract_text_from_image(image_path)
        
        if extracted_text:
            # Add metadata header
            header = f"Text extracted from: {image_path}\n"
            header += f"Extraction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            header += f"{'='*50}\n\n"
            
            full_text = header + extracted_text
            
            # Save to file
            self.save_text_to_file(full_text, output_path)
            print(f"Successfully extracted text from '{image_path}'")
        else:
            print(f"No text found in image '{image_path}'")
    
    def process_multiple_images(self, image_folder, output_folder=None):
        """
        Process multiple images from a folder
        
        Args:
            image_folder (str): Path to folder containing images
            output_folder (str): Optional path for output folder
        """
        if not os.path.exists(image_folder):
            print(f"Error: Folder '{image_folder}' not found!")
            return
        
        # Create output folder if not specified
        if output_folder is None:
            output_folder = os.path.join(image_folder, "extracted_texts")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Supported image extensions
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Process all images in folder
        image_files = [f for f in os.listdir(image_folder) 
                      if Path(f).suffix.lower() in supported_extensions]
        
        if not image_files:
            print(f"No supported image files found in '{image_folder}'")
            return
        
        print(f"Found {len(image_files)} image(s) to process...")
        
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            output_path = os.path.join(output_folder, f"{Path(image_file).stem}_extracted.txt")
            
            self.process_single_image(image_path, output_path)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Extract text from images using EasyOCR")
    parser.add_argument("input", help="Path to image file or folder containing images")
    parser.add_argument("-o", "--output", help="Output file/folder path")
    parser.add_argument("-l", "--languages", nargs='+', default=['en'], 
                       help="Languages for OCR (e.g., en es fr)")
    parser.add_argument("--folder", action="store_true", 
                       help="Process all images in the input folder")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = ImageTextExtractor(languages=args.languages)
    
    if args.folder:
        # Process multiple images
        extractor.process_multiple_images(args.input, args.output)
    else:
        # Process single image
        extractor.process_single_image(args.input, args.output)

# Example usage
if __name__ == "__main__":
    # Uncomment and modify these lines for direct usage without command line
    
    # Example 1: Process single image
    # extractor = ImageTextExtractor(['en'])  # You can add more languages like ['en', 'es', 'fr']
    # extractor.process_single_image("path/to/your/image.jpg")
    
    # Example 2: Process multiple images from a folder
    # extractor = ImageTextExtractor(['en'])
    # extractor.process_multiple_images("path/to/your/image/folder")
    
    # Run command line interface
    main()