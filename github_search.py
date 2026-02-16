import requests
import json
import os
import re

def search_github_repositories(token=None):
    """Search GitHub for fire/smoke detection repositories with negative samples"""
    headers = {
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28'
    }
    
    if token:
        headers['Authorization'] = f'token {token}'
    
    # Search for repositories with fire/smoke detection tags and negative samples
    search_queries = [
        'fire-detection dataset negative samples',
        'smoke-detection hard negatives',
        'fire-detection false positives',
        'smoke-detection background images',
        'fire-dataset negative examples'
    ]
    
    all_repos = []
    
    for query in search_queries:
        url = f'https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=100'
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'items' in data:
                for repo in data['items']:
                    if repo['stargazers_count'] >= 100 and repo['created_at'] >= '2020-01-01':
                        all_repos.append({
                            'name': repo['full_name'],
                            'url': repo['html_url'],
                            'stars': repo['stargazers_count'],
                            'created_at': repo['created_at']
                        })
        
        except requests.exceptions.RequestException as e:
            print(f"Error searching for '{query}': {e}")
            continue
    
    # Remove duplicates
    seen = set()
    unique_repos = []
    for repo in all_repos:
        if repo['name'] not in seen:
            seen.add(repo['name'])
            unique_repos.append(repo)
    
    return unique_repos


def search_negative_sample_files(repo_full_name, token=None):
    """Search for negative sample files in a repository"""
    headers = {
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28'
    }
    
    if token:
        headers['Authorization'] = f'token {token}'
    
    # Common paths for negative samples
    target_paths = [
        'negative_samples', 'hard_negatives', 'false_positives', 'background', 
        'test/negative', 'train/background', 'data/negative', 'images/negative'
    ]
    
    target_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    
    found_images = []
    
    # Search for files in the repository
    try:
        url = f'https://api.github.com/repos/{repo_full_name}/git/trees/main?recursive=1'
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'tree' in data:
            for item in data['tree']:
                if item['type'] == 'blob':
                    # Check if file is in target path and has target extension
                    has_target_ext = any(item['path'].lower().endswith(ext) for ext in target_extensions)
                    has_target_path = any(path in item['path'] for path in target_paths)
                    
                    if has_target_ext and has_target_path:
                        # Generate raw GitHub URL for the file
                        raw_url = f'https://github.com/{repo_full_name}/raw/main/{item["path"]}'
                        found_images.append({
                            'repo': repo_full_name,
                            'path': item['path'],
                            'url': raw_url,
                            'size': item['size']
                        })
    
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {repo_full_name}: {e}")
        return []
    
    return found_images


def search_specific_repos(token=None):
    """Search specific well-known repositories for negative samples"""
    specific_repos = [
        'openimages/dataset',
        'roboflow/fire-detection',
        'roboflow/smoke-detection',
        'ultralytics/yolov5',
        'WongKinYiu/yolov7'
    ]
    
    all_images = []
    
    for repo in specific_repos:
        print(f"Searching in {repo}...")
        images = search_negative_sample_files(repo, token)
        all_images.extend(images)
    
    return all_images


def main():
    """Main function to search GitHub for hard negative images"""
    # Get GitHub token from environment variable if available
    token = os.getenv('GITHUB_TOKEN')
    
    print("Searching for relevant GitHub repositories...")
    repos = search_github_repositories(token)
    
    print(f"\nFound {len(repos)} relevant repositories:")
    for repo in repos:
        print(f"- {repo['name']} ({repo['stars']} stars) - {repo['url']}")
    
    print("\nSearching for negative sample images...")
    all_images = search_specific_repos(token)
    
    # Search negative samples in discovered repos
    for repo in repos:
        print(f"\nSearching in {repo['name']}...")
        images = search_negative_sample_files(repo['name'], token)
        all_images.extend(images)
    
    print(f"\nTotal negative sample images found: {len(all_images)}")
    
    # Save results
    output_dir = 'hard_negatives'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'github_negative_images.json'), 'w') as f:
        json.dump(all_images, f, indent=2)
    
    with open(os.path.join(output_dir, 'repositories.txt'), 'w') as f:
        for repo in repos:
            f.write(f"{repo['name']} ({repo['stars']} stars) - {repo['url']}\n")
    
    # Filter images by size and format
    valid_images = []
    for img in all_images:
        # Check file size (minimum 10KB, maximum 10MB)
        if img['size'] > 10000 and img['size'] < 10000000:
            valid_images.append(img)
    
    print(f"\nValid negative images (size filtered): {len(valid_images)}")
    
    # Save valid images list
    with open(os.path.join(output_dir, 'valid_negative_images.json'), 'w') as f:
        json.dump(valid_images, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/ directory")


if __name__ == "__main__":
    main()
