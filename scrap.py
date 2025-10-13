from icrawler.builtin import BingImageCrawler
from PIL import Image, ImageEnhance
import os



def scrape_images(keyword, folder_name, max_images):
    save_dir = f'data/{folder_name}'
    os.makedirs(save_dir, exist_ok=True)

    crawler = BingImageCrawler(storage={'root_dir': save_dir})
    crawler.crawl(
        keyword=keyword,
        max_num=max_images,
        filters={ 'size': 'large',  'type': 'photo' }
    )

    for file in os.listdir(save_dir):
        path = os.path.join(save_dir, file)
        try:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = Image.open(path).convert("RGB")

                # Resize to 128x128
                img = img.resize((128, 128), Image.LANCZOS)

                # Enhance image quality
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(1.1)  # slightly brighter

                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)  # better contrast

                # Save in JPG format
                new_path = os.path.splitext(path)[0] + ".jpg"
                img.save(new_path, "JPEG", quality=90)

                # Remove original if format was different
                if new_path != path:
                    os.remove(path)

        except Exception as e:
            print(f"Error processing {path}: {e}")



# List of objects to scrape
objects_to_scrape = {
    "apples": ["apple fruit"],
    "bananas": ["banana fruit"],
    "oranges": ["orange fruit"],
    "mangoes": ["mango fruit"],
    "grapes": ["grape fruit"],
    "cats": ["cute cat"],
    "dogs": ["puppy dog"],
    "car": ["sports car"],
    "bike": ["motorbike"],
    "bus": ["city bus"],
    "train": ["train engine"],
    "flower": ["rose flower"],
    "tree": ["oak tree"],
    "bird": ["sparrow bird"],
    "fish": ["goldfish"],
    "chair": ["office chair"],
    "table": ["dining table"],
    "laptop": ["laptop computer"],
    "phone": ["smartphone"],
    "watch": ["wrist watch"]
}


# Scrape images for each category
for folder, keywords in objects_to_scrape.items():
    for keyword in keywords:
        print(f"Scraping: {keyword} into folder: {folder}")
        scrape_images(keyword, folder, max_images=30)
