#!/bin/bash

# Define a function to get the absolute path of the found directories
get_absolute_paths() {
    while IFS= read -r line; do
        realpath "$line"
    done
}

# Use the find command to locate directories with the specified names/patterns
dirs_to_delete=$(find . -type d \( -name "*B-format*" -o -name "*51*" -o -name "*MS*" -o -name "*B-Format*" -o -name "*5-1*" -o -name "*Surround*" -o -name "*Soundmorph*" \) | get_absolute_paths)

# Check if any directories were found
if [ -z "$dirs_to_delete" ]; then
    echo "No directories found."
    exit 0
fi

# Display the directories to the user
echo "The following directories will be deleted:"
echo "$dirs_to_delete"

# Ask for confirmation twice
read -p "Are you sure you want to delete these directories? [y/N] " confirmation1
if [[ $confirmation1 != 'y' && $confirmation1 != 'Y' ]]; then
    echo "Aborted."
    exit 0
fi

read -p "Are you absolutely sure? This cannot be undone. [y/N] " confirmation2
if [[ $confirmation2 != 'y' && $confirmation2 != 'Y' ]]; then
    echo "Aborted."
    exit 0
fi

# Delete the directories
echo "$dirs_to_delete" | xargs rm -rf

echo "Directories deleted."
