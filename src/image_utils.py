"""Image utilities and Unsplash API integration for SensorMCP server."""

import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import requests

from state import (
    HTTP_OK,
    MAX_IMAGES_PER_PAGE,
    MAX_WAIT_TIME,
    SHORT_WAIT_TIME,
    STATE,
    SUPPORTED_IMAGE_EXTENSIONS,
)


def get_image_files(folder_path: str) -> List[str]:
    """Get all image files from a directory.
    
    Args:
        folder_path: Path to the folder
        
    Returns:
        List of image filenames
    """
    return [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and
        any(f.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS)
    ]


def validate_unsplash_params(access_key: str, save_dir: Optional[str]) -> Tuple[bool, str]:
    """Validate Unsplash parameters and setup directories.

    Args:
        access_key: Unsplash API key
        save_dir: Directory to save images

    Returns:
        Tuple of (is_valid, error_message or save_dir)
    """
    if not access_key:
        return (
            False,
            "Error: Unsplash API key not found. Please set the UNSPLASH_API_KEY environment variable in the .env file.",
        )

    # Use current directory if save_dir is not specified
    if save_dir is None:
        return True, os.getcwd()

    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
            return True, save_dir
        except Exception as e:
            return False, f"Error: Could not create directory '{save_dir}': {str(e)}"

    return True, save_dir


def search_unsplash_images(query: str, max_images: int, access_key: str) -> Tuple[bool, Union[List[Dict], str]]:
    """Search for images on Unsplash.

    Args:
        query: Search keyword
        max_images: Maximum number of images to fetch
        access_key: Unsplash API key

    Returns:
        Tuple of (success, photos_list or error_message)
    """
    per_page = MAX_IMAGES_PER_PAGE
    pages_needed = math.ceil(max_images / per_page)
    all_results = []

    for page in range(1, pages_needed + 1):
        url = (
            f"https://api.unsplash.com/search/photos"
            f"?query={query}&client_id={access_key}&per_page={per_page}&page={page}"
        )

        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code != HTTP_OK:
                return False, f"Image search failed: API returned status code {response.status_code}"
            
            # Parse response JSON
            response_data = response.json()
            if "results" not in response_data:
                return False, "Image search failed: Unexpected API response format"
                
            # Get results from current page
            current_results = response_data["results"]
            all_results.extend(current_results)

            # Stop if we have enough images or no more results
            if len(current_results) < per_page or len(all_results) >= max_images:
                break

            # Add delay to avoid API rate limits
            if page < pages_needed:
                time.sleep(MAX_WAIT_TIME)
                
        except requests.RequestException as e:
            return False, f"Network error during image search: {str(e)}"
        except json.JSONDecodeError:
            return False, "Failed to parse API response"
        except Exception as e:
            return False, f"Unexpected error during image search: {str(e)}"

    # Limit results to requested amount
    photos = all_results[:max_images]
    if not photos:
        return False, f"No images found for query '{query}'."

    return True, photos


def download_images(photos: List[Dict], full_path: str) -> int:
    """Download images from Unsplash.

    Args:
        photos: List of photo information
        full_path: Path to save images

    Returns:
        Number of successfully downloaded images
    """
    downloaded_count = 0
    for i, photo in enumerate(photos):
        try:
            image_url = photo["urls"]["regular"]
            image_response = requests.get(image_url, timeout=30)
            
            if image_response.status_code != HTTP_OK:
                print(f"Failed to download image {i+1}: HTTP status {image_response.status_code}")
                continue
                
            image_path = os.path.join(full_path, f"{photo['id']}.jpg")
            with open(image_path, "wb") as f:
                f.write(image_response.content)
            downloaded_count += 1

            # Add short delay to avoid rapid requests
            if i < len(photos) - 1:
                time.sleep(SHORT_WAIT_TIME)
                
        except requests.RequestException as e:
            print(f"Network error downloading image {i+1}: {e}")
        except IOError as e:
            print(f"I/O error saving image {i+1}: {e}")
        except Exception as e:
            print(f"Unexpected error downloading image {i+1}: {e}")

    return downloaded_count 