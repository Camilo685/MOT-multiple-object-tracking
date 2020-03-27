import glob, os

current_dir = os.path.dirname(os.path.abspath(__file__))

current_dir = '/content/training/data_object_image_2/training/image_2'
count = 0
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.txt")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    with open(title + '.txt', "r") as f:
        lines = f.readlines()
    with open(title + '.txt', "w") as f:
        for line in lines:
            if line[0] == "0":
                f.write(line)
    count = count +1
    if(os.stat(title + '.txt').st_size == 0):
        os.remove(title + '.txt')
    if(os.path.exists(title + '.png')):
        if not os.path.exists(title + '.txt'):
           os.remove(title + '.png')
    print(count)


