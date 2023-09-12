import os
import json
import datetime

METADATA_FILE = './static/data/metadata.json'


def update_metadata(video_filename, thumbnail_filename):
    # Load existing metadata or initialize an empty dictionary
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Update or add new metadata entry
    metadata[video_filename] = {
        "thumbnail": thumbnail_filename,
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # You can add any other details you wish here
    }

    # Save updated metadata back to file
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)

def delete_metadata_entry(video_filename):
    """Removes a metadata entry corresponding to a video."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        if video_filename in metadata:
            del metadata[video_filename]

        # Save updated metadata back to file
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f)

def validate_metadata():
    """Validate metadata against actual files and remove invalid entries or add new ones."""
    if not os.path.exists(METADATA_FILE):
        metadata = {}
    else:
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)

    directory = './static/data'
    actual_files = os.listdir(directory)

    # Add entries for video files that aren't in the metadata
    for file in actual_files:
        if file.endswith('.mp4') and file not in metadata:
            metadata[file] = {
                "thumbnail": file.replace('.mp4', '.jpg'),
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # This will give the current timestamp. If you need the file's actual timestamp,
                # you can use os.path.getmtime and convert it to a datetime object.
            }

    # Check each file in metadata against actual files and remove if not found
    keys_to_delete = [key for key in metadata if key not in actual_files]
    for key in keys_to_delete:
        del metadata[key]

    # Save updated metadata back to file
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)
if __name__ == "__main__":
    validate_metadata()
