#!/usr/bin/python
import sys
import networkx as nx
import glob 
import pdb

if len(sys.argv) < 2:
	print "Usage: python get_graph_stat.py <file-regex (in DOUBLE QUOTES)>"
	sys.exit()
	
def get_stat(gfile):
	num_graphs = 0
	vlabels = dict()
	elabels = dict()
	glist = []
	fin = open(gfile,"r")
	for lines in fin:
		w = lines.split()
		if w[0] == "t":
			num_graphs += 1
			glist.append(nx.Graph())
		if w[0] == "v":
			if w[2] not in vlabels:
				vlabels[w[2]] = 1
		if w[0] == "e" or w[0] == "u":
			if w[3] not in elabels:
				elabels[w[3]] = 1
			glist[num_graphs-1].add_edge(int(w[1]),int(w[2]))			
	fin.close()

			
	# remove the empty graphs
	for i in range(len(glist)-1,-1,-1):
		if glist[i].number_of_nodes() < 2:
			del(glist[i])
	num_graphs = len(glist)
	avg_v = sum([g.number_of_nodes() for g in glist])*1.0/num_graphs
	avg_e = sum([ g.number_of_edges() for g in glist])*1.0/num_graphs
	num_vlabels = len(vlabels.keys())
	num_elabels = len(elabels.keys())
	#pdb.set_trace()
	#print nx.info(glist[0])
	#sum([d[1] for d in self.degree_iter()])/float(len(self))
	avg_degrees = [sum(g.degree().values())*1.0/len(g.nodes()) for g in glist]
	avg_d =  sum(avg_degrees)/len(avg_degrees)
	ccoeff = [nx.average_clustering(g) for g in glist]
	avg_ccoeff = sum(ccoeff) / len(ccoeff)
	

	return [num_graphs, avg_v, avg_e, num_vlabels, num_elabels,  avg_d, avg_ccoeff]

def print_stats(files):
	print "--------------------------------------------------------------------------------------------------------"
	print "filename \t# graphs\tavg. |V| \tavg. |E|\tV labels\tE labels\tavg. deg\tavg. clus. coeff.\t"
	print "--------------------------------------------------------------------------------------------------------"
	for f in files:
		fname = f.split("/")[len(f.split("/"))-1]
		[n,v,e,vl,el,d,cc] = get_stat(f)
		print fname+"\t"+str(n)+"\t"+str(v)+"\t"+str(e)+"\t"+str(vl)+"\t"+str(el)+"\t"+str(d)+"\t"+str(cc)+"\n"		



if __name__ == "__main__":
	files = glob.glob(sys.argv[1])
	print sys.argv[1]
	print files
	print_stats(files)
