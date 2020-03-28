import glob;

vlabels = dict()
vlcount = 0

files = glob.glob("/mnt/c/home/anchupa/datasets_for_GPU/Families_Out/*/*")
outfile = open("pdb_graphs.dat","w")
t = 0
for f in files:
	infile = open(f,"r")
	outfile.write("t # "+str(t)+"\n") 
	for line in infile:
		w = line.split()				
		if w[0] == "v":
			if w[2] not in vlabels:
				vlabels[w[2]] = vlcount
				vlcount += 1
			outfile.write("v "+w[1]+" "+str(vlabels[w[2]])+"\n")
		elif w[0] == "e":
			outfile.write("e "+w[1]+" "+w[2]+" 0\n");
	infile.close()
	t += 1
outfile.close()

