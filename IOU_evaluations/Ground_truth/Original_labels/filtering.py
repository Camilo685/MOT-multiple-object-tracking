for x in range(1, 21):
	with open(("image" + str(x) + ".txt"), "r") as f:
		lines = f.readlines()
	with open(("/home/camilo685/Desktop/images_test/Ground_truth/label" + str(x) + ".txt"), "w") as f:
		for line in lines:
			f.write(line[2:])
