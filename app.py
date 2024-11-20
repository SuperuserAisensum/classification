import requests
import re
import csv
import datetime
import gradio as gr
import os
from openai import OpenAI
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Define reference images directory
REFERENCE_IMAGES_DIR = 'reference_images'
os.makedirs(REFERENCE_IMAGES_DIR, exist_ok=True)

def load_reference_images():
    """Load all reference images from the reference directory"""
    reference_data = {}
    for category in os.listdir(REFERENCE_IMAGES_DIR):
        category_path = os.path.join(REFERENCE_IMAGES_DIR, category)
        if os.path.isdir(category_path):
            reference_data[category] = []
            for img_file in os.listdir(category_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(category_path, img_file)
                    reference_data[category].append(img_path)
    return reference_data

def compare_with_reference(image_url, product_category):
    """Compare product image with reference images using OpenAI Vision"""
    reference_images = load_reference_images().get(product_category, [])
    
    if not reference_images:
        return "Error: No reference images found for this category", 0

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Compare these images and determine if the product appears to be authentic. 
                        Consider:
                        1. Logo placement and quality
                        2. Product design details
                        3. Material quality appearance
                        4. Color accuracy
                        5. Overall build quality
                        
                        The first image is the reference (authentic product).
                        The second image is the product to verify.
                        
                        Respond with 'Pass' if it appears authentic or 'Not Pass' if it shows signs of being counterfeit.
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": reference_images[0]}  # Using first reference image
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip()
        confidence = 1.0 if result == "Pass" else 0.0
        
        return result, confidence
        
    except Exception as e:
        print(f"Error in comparison: {e}")
        return "Error", 0

def scrape_shopee_reviews(product_url, product_category):
    if not product_url or not product_url.strip():
        return "Error: Product URL is required.", None
    
    # Update the regex pattern to better match Shopee URLs
    match = re.search(r'i\.(\d+)\.(\d+)', product_url)
    if not match:
        return "Error: Invalid Shopee URL format. Please try again.", None

    shop_id, item_id = match.groups()
    
    # Update the API endpoint and add more headers
    product_url = f'https://shopee.co.id/api/v4/pdp/get_pc?shop_id={shop_id}&item_id={item_id}'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Referer': 'https://shopee.co.id/',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'x-api-source': 'pc',
        'x-requested-with': 'XMLHttpRequest',
        'x-shopee-language': 'id'
    }
    
    try:
        # Create a session to maintain cookies
        session = requests.Session()
        
        # First, visit the main product page to get cookies
        session.get(f'https://shopee.co.id/-i.{shop_id}.{item_id}', headers=headers)
        
        # Then make the API request
        response = session.get(product_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"Debug - Status Code: {response.status_code}")
            print(f"Debug - Response: {response.text[:200]}")  # Print first 200 chars of response
            return f"Error: Failed to fetch product data (HTTP {response.status_code}).", None

        product_data = response.json()
        
        # Update the path to images in the JSON response
        product_images = product_data.get('data', {}).get('product_info', {}).get('images', [])
        
        if not product_images:
            return "Error: No product images found.", None

        results = []
        for img_id in product_images:
            image_url = f"https://cf.shopee.co.id/file/{img_id}"
            classification_result, confidence = compare_with_reference(image_url, product_category)
            results.append({
                'image_url': image_url,
                'classification': classification_result,
                'confidence': confidence
            })

        output_file = 'authenticity_check.csv'
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['image_url', 'authenticity_result', 'confidence'])
            for result in results:
                writer.writerow([
                    result['image_url'],
                    result['classification'],
                    f"{result['confidence']:.2%}"
                ])

        pass_count = sum(1 for r in results if r['classification'] == 'Pass')
        total_images = len(results)
        summary = f"""
        Authenticity Check Results:
        Total Images Analyzed: {total_images}
        Appears Authentic: {pass_count}
        Potentially Counterfeit: {total_images - pass_count}
        
        Detailed results saved to {output_file}
        """
        
        return summary, results[0]['image_url']

    except requests.Timeout:
        return "Error: Request timed out. Please try again.", None
    except requests.RequestException as e:
        return f"Error: Failed to fetch data: {str(e)}", None

def gradio_scrape(product_url, product_category):
    result, image_url = scrape_shopee_reviews(product_url, product_category)
    if image_url:
        img = Image.open(BytesIO(requests.get(image_url).content))
        return result, img
    return result, None

# Get available categories from reference_images directory
categories = [d for d in os.listdir(REFERENCE_IMAGES_DIR) 
             if os.path.isdir(os.path.join(REFERENCE_IMAGES_DIR, d))]

# Gradio Interface
interface = gr.Interface(
    fn=gradio_scrape,
    inputs=[
        gr.Textbox(label="Shopee Product URL", placeholder="Enter Shopee product URL here"),
        gr.Dropdown(choices=categories, label="Product Category")
    ],
    outputs=[
        gr.Textbox(label="Authenticity Check Results"),
        gr.Image(label="Product Image Sample")
    ],
    title="Shopee Product Authenticity Checker",
    description="Enter a Shopee product URL to check product authenticity against reference images.",
)

if __name__ == "__main__":
    interface.launch()
