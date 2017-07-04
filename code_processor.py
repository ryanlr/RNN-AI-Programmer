import os
from pathlib import Path

path = "/home/PC/Downloads/JDK-master/src/java/"

dest = "/home/PC/Documents/jdk-chars.txt"
for subdir, dirs, files in os.walk(path):
    for file in files:
        name = os.path.join(subdir, file)
        if name.endswith(".java"):
            print(name)

            with open(name) as f:
                content = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            content = [x.strip() for x in content]
            for line in content:
                if "*" in line:
                    continue
                if len(line)==0:
                    continue
                with open(dest, "a") as myfile:
                    myfile.write(line+"\n")

"""        


"""