#!/bin/bash
# This script runs four python commands in sequence, waiting for each one to finish before starting the next one.

# Define an array of arguments to update the config file
ARGS=(2 5 10 20 30 50 100)

# Loop over the array of arguments
for arg in "${ARGS[@]}";
do
  # Update the config file with the current argument
  python3 source/update_config_file.py $arg

  # Check the exit status of the previous command
  if [ $? -eq 0 ]; then
      # If successful, run the main script
      python3 source/train.py
  else
      # If not successful, print an error message and exit
      echo "Failed to update config file with argument $arg"
      exit 1
  fi
done

# Print a success message at the end
echo "All commands executed successfully"
