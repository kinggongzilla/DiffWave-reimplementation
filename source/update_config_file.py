
import sys





# updates config file with new number of timesteps for experimentation
#Example: update_config_file("config.py", 100)
def update_config_file(file_path = 'source/config.py'):

    #parse commandline argument
    if len(sys.argv) > 1:
        time_steps = sys.argv[1]
    else:
        time_steps = 2

    # open the file in read mode
    with open(file_path, "r") as f:
        # read the lines of the file
        lines = f.readlines()

    # open the file in write mode
    with open(file_path, "w") as f:
        # loop through the lines
        for line in lines:
            # check if the line starts with VARIANCE_SCHEDULE
            if line.startswith("TIME_STEPS"):
                # replace the line with the new value
                line = f"TIME_STEPS = {time_steps}\n"
            # write the line to the file
            f.write(line)

#if name main run experiment
if __name__ == '__main__':
    update_config_file()