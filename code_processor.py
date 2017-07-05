import os
from pathlib import Path

path = "/home/pc/Downloads/JDK-master/src/java/"

dest = "./jdk-chars.txt"
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
                if line.startswith("//"):
                    continue
                if len(line)==0:
                    continue
                with open(dest, "a") as myfile:
                    myfile.write(line+"\n")

"""        


"""