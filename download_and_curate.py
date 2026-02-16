import requests
import os
import json
import re
from PIL import Image
from io import BytesIO
import hashlib

def load_valid_images():
    """Load valid negative images from JSON file"""
    try:
        with open('hard_negatives/valid_negative_images.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: valid_negative_images.json not found. Run github_search.py first.")
        return []


def categorize_image(url):
    """Categorize image based on URL and filename patterns"""
    url_lower = url.lower()
    
    # Define category patterns
    categories = {
        'bright_artificial_lights': [
            'led', 'lamp', 'light', 'streetlight', 'headlight', 'stage-light', 'neon', 'fluorescent'
        ],
        'solar_lens_artifacts': [
            'sunset', 'sunrise', 'lens-flare', 'glare', 'golden-hour', 'sun', 'sky', 'cloud'
        ],
        'warm_colored_objects': [
            'orange', 'red', 'terracotta', 'rust', 'autumn', 'fall', 'curtain', 'flag', 'construction'
        ],
        'smoke_like_textures': [
            'steam', 'fog', 'mist', 'dust', 'breath', 'aerosol', 'vapor', 'smoke-like'
        ],
        'reflections': [
            'reflection', 'glass', 'water', 'mirror', 'car-paint', 'puddle'
        ]
    }
    
    # Determine category
    for category, patterns in categories.items():
        if any(pattern in url_lower for pattern in patterns):
            return category
    
    # Default to uncategorized if no patterns match
    return 'uncategorized'


def download_image(url, save_path):
    """Download image from URL to specified path"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Check if content is an image
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            print(f"Not an image file: {url}")
            return False
        
        # Verify image can be opened
        img = Image.open(BytesIO(response.content))
        img.verify()
        
        # Check resolution (minimum 640x480)
        img = Image.open(BytesIO(response.content))
        width, height = img.size
        if width < 640 or height < 480:
            print(f"Image too small: {url} ({width}x{height})")
            return False
        
        # Save image
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return True
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def create_category_directories():
    """Create category-specific directories"""
    categories = [
        'bright_artificial_lights',
        'solar_lens_artifacts',
        'warm_colored_objects',
        'smoke_like_textures',
        'reflections',
        'uncategorized'
    ]
    
    for category in categories:
        os.makedirs(f'hard_negatives/{category}', exist_ok=True)


def download_and_categorize_images(images):
    """Download and categorize all valid images"""
    create_category_directories()
    
    downloaded_images = []
    category_counts = {
        'bright_artificial_lights': 0,
        'solar_lens_artifacts': 0,
        'warm_colored_objects': 0,
        'smoke_like_textures': 0,
        'reflections': 0,
        'uncategorized': 0
    }
    
    for img in images:
        category = categorize_image(img['url'])
        
        # Check category count limit (50 images per category)
        if category_counts[category] >= 50:
            continue
        
        # Generate unique filename
        filename = hashlib.md5(img['url'].encode()).hexdigest()
        file_ext = os.path.splitext(img['path'])[1]
        save_path = f'hard_negatives/{category}/{filename}{file_ext}'
        
        # Download image
        if download_image(img['url'], save_path):
            downloaded_images.append({
                'url': img['url'],
                'repo': img['repo'],
                'path': img['path'],
                'category': category,
                'filename': f'{filename}{file_ext}'
            })
            
            category_counts[category] += 1
            print(f"Downloaded: {img['url']}")
    
    print("\nDownload summary:")
    for category, count in category_counts.items():
        print(f"- {category}: {count} images")
    
    return downloaded_images


def generate_markdown_table(images):
    """Generate markdown table of curated images"""
    table = "| Image URL | Repository | Category | Filename |\n"
    table += "|-----------|------------|----------|----------|\n"
    
    for img in images:
        table += f"| [View Image]({img['url']}) | [{img['repo']}](https://github.com/{img['repo']}) | {img['category']} | {img['filename']} |\n"
    
    return table


def generate_repositories_list(repos):
    """Generate text file listing repositories with negative samples"""
    repo_list = []
    for repo in repos:
        repo_list.append(f"{repo['name']} ({repo['stars']} stars) - {repo['url']}")
    
    return '\n'.join(repo_list)


def main():
    """Main function to download and curate images"""
    # Load valid images
    images = load_valid_images()
    
    if not images:
        print("No valid images to download.")
        return
    
    # Download and categorize images
    downloaded_images = download_and_categorize_images(images)
    
    # Generate markdown table
    markdown_table = generate_markdown_table(downloaded_images)
    
    with open('hard_negatives/hard_negatives_table.md', 'w') as f:
        f.write(markdown_table)
    
    print(f"\nMarkdown table saved to hard_negatives/hard_negatives_table.md")
    
    # Generate repositories list
    try:
        with open('hard_negatives/repositories.txt', 'r') as f:
            repos_text = f.read()
        
        # Parse repos from text file
        repos = []
        lines = repos_text.strip().split('\n')
        for line in lines:
            match = re.match(r'^(.+) \((\d+) stars\) - (.+)$', line)
            if match:
                repos.append({
                    'name': match.group(1),
                    'stars': int(match.group(2)),
                    'url': match.group(3)
                })
        
        # Save curated repos
        curated_repos = []
        repo_names = set()
        for img in downloaded_images:
            if img['repo'] not in repo_names:
                repo_names.add(img['repo'])
                # Find repo info
                repo_info = next((r for r in repos if r['name'] == img['repo']), None)
                if repo_info:
                    curated_repos.append(repo_info)
        
        curated_repos_text = generate_repositories_list(curated_repos)
        
        with open('hard_negatives/negative_sample_repositories.txt', 'w') as f:
            f.write(curated_repos_text)
        
        print(f"Negative sample repositories saved to hard_negatives/negative_sample_repositories.txt")
    
    except FileNotFoundError:
        print("Error: repositories.txt not found")
    
    print("\nDownload and curation complete!")


if __name__ == "__main__":
    main()
