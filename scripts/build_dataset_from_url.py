import argparse
import os
import sys
from PIL import Image
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dataset_collection.scrape_utils import *
from utils import *

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Download articles and images based on provided URLs.')
    parser.add_argument('--scrape_image', type=int, default=1,
                        help='If 1, downloads the FC images in addition to the article text. If 0, assumes the images have already been scraped.') 
    parser.add_argument('--image_processing_script', type=str, default='dataset/image_processing_instructions.txt',
                        help='Script with automated instructions to crop and clean the images.') 
    parser.add_argument('--sleep', type=int, default=5,
                        help='Waiting time in seconds between collection of articles.')

    args = parser.parse_args()

    # Collect the article and image for the 1676 instances of the 5Pillars dataset
    train = load_json('dataset/train.json')
    val = load_json('dataset/val.json')
    test = load_json('dataset/test.json')

    # ==== In thống kê dataset ====
    print("===== Dataset Statistics =====")
    print(f"Train samples: {len(train)}")
    print(f"Validation samples: {len(val)}")
    print(f"Test samples: {len(test)}")
    print(f"Total samples: {len(train) + len(val) + len(test)}")
    print("=================================")

    urls = [t['URL'] for t in train] + [t['URL'] for t in val] + [t['URL'] for t in test]
    image_urls = [t['image_URL'] for t in train] + [t['image_URL'] for t in val] + [t['image_URL'] for t in test]

    # Group the URL by FC organization, as each organization uses a different parser
    factly_urls, factly_image_urls =  [], []
    pesacheck_urls, pesacheck_image_urls =  [], []
    two11org_urls, two11org_image_urls =  [], []
    for u in range(len(urls)):
        if 'factly.in' in urls[u]:
            factly_urls.append(urls[u])
            factly_image_urls.append(image_urls[u])
        elif 'pesacheck.org' in urls[u]:
            pesacheck_urls.append(urls[u])
            pesacheck_image_urls.append(image_urls[u])
        elif '211check.org' in urls[u]:
            two11org_urls.append(urls[u])
            two11org_image_urls.append(image_urls[u])
        else:
            pass

    # Scrape the article content and the images
    collect_articles(factly_urls,factly_parser,args.scrape_image, factly_image_urls, args.sleep)
    collect_articles(pesacheck_urls,pesacheck_parser,args.scrape_image, pesacheck_image_urls, args.sleep)
    collect_articles(two11org_urls,two11org_parser,args.scrape_image, two11org_image_urls, args.sleep)

    # Image processing
    if not 'processed_img' in os.listdir('dataset/'):
        os.mkdir('dataset/processed_img/')

    print(">>> Start processing images...")
    process_images_from_instructions(args.image_processing_script, 'dataset/img/', 'dataset/processed_img/')

    # Đếm số ảnh đã xử lý
    processed_count = len([f for f in os.listdir('dataset/processed_img/') if os.path.isfile(os.path.join('dataset/processed_img/', f))])
    print(f"Total processed images: {processed_count}")
