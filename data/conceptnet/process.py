
from collections import defaultdict
for file_name in ["dev1.txt", "dev2.txt", "test.txt", "train100k.txt"]:
	string_tuples = open("./raw_data/"+ file_name, "r").read().split("\n")
	tuples = [x.split("\t") for x in string_tuples if x]
	diction = defaultdict(int)
	# print(tuples)
	file_object = open(file_name, "w")
	for i in range(len(tuples)):
		tuples[i][1], tuples[i][2] = tuples[i][2], tuples[i][1]
		file_object.writelines("\t".join(tuples[i]) + "\n")
		diction[tuples[i][0]] += 1
	print(file_name,"\n")
	print(diction)
	print("\n\n")

	file_object.close()
