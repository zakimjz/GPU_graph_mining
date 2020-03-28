import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random

#nodeLabels = [ 'a' , 'b', 'c', 'd']
#edgeLabels = [ 'w','x','y','z']
nodeLabels = range(0,10)
edgeLabels = range(0,12)

numgraphs = 1000
numsequences = 100   # number of graphs when seq_len_min = seq_len_max = 1

pattern_edge_add_prob = 0.3	
pattern_edge_rem_prob = 0.05
graph_edge_prob = 0.05

pattern_num_nodes_min = 3
pattern_num_nodes_max = 6
seq_len_min = 1
seq_len_max = 1
graph_num_nodes_min = 15 
graph_num_nodes_max = 20


min_sup = 10
max_sup = min_sup + 10 


non_overlapping_embedding = 0

##################################################
# Function Definitions							

## print the graph ( graph, file )
def print_graph(*arg):
	if len(arg) == 1: #if no file specified
		g = arg[0]
		for n in g.nodes():
			print "v "+str(n)+" "+str(g.node[n]['l']) 
		for e in g.edges():
			print "e "+str(e[0])+" "+str(e[1])+" "+str(g.edge[e[0]][e[1]]['l'])	
    		#print "\n"
     		return

	if len(arg) == 2: # if file is provided
		g = arg[0]
		ostream = arg[1]
		for n in g.nodes():
			ostream.write("v "+str(n)+" "+str(g.node[n]['l'])+"\n") 
		for e in g.edges():
			ostream.write("e "+str(e[0])+" "+str(e[1])+" "+str(g.edge[e[0]][e[1]]['l'])+"\n")	
    		#print "\n"
     		return


## augment the pattern with new node addition, deletion and new edge addition and deletion

def augment_pattern(p):
	# randomly choose number of nodes for the new pattern
	aug_num_nodes = random.randint(pattern_num_nodes_min,pattern_num_nodes_max)
	# copy the pattern first
	g  = nx.Graph(p)	

	#print "received Pattern"
	#print_graph(p)

	maxn = max(g.nodes())
	#if original pattern has fewer nodes, add nodes and connect with the existing graph
	if (maxn + 1) < aug_num_nodes:
		for t in range(maxn+1,aug_num_nodes):
			li = random.randint(0,len(nodeLabels)-1)
			g.add_node(t, l = nodeLabels[li])
			li = random.randint(0,len(edgeLabels)-1)
			n = random.randint(0,t-1)
			g.add_edge(t, n, l = edgeLabels[li])
						

	# if original pattern has more nodes, remove nodes
	if (maxn + 1) > aug_num_nodes:
		rlist = range(maxn)
		random.shuffle(rlist)
		rlist = rlist[0:maxn - aug_num_nodes -1]
		#print g.nodes()
		#print rlist
		count = 0 
		for j in rlist:
			#recovery point
			nbrs = g.neighbors(j)
			node_lbl = g.node[j]['l'] 	
			edge_lbls = []
			for v in nbrs:
				edge_lbls.append(g[j][v]['l']) 
						
			g.remove_node(j)

			# if the graph became  disconnected put the node back with all its edges
			if len(nx.connected_components(g) ) > 1:
				g.add_node(j,l=node_lbl)
				for k in range(0, len(nbrs)):
					g.add_edge(j,nbrs[k],l=edge_lbls[k])
				
			
	# add and remove  edge
	for j in range(0,aug_num_nodes-1):
		if j in g.nodes():
			for k in range(j+1,aug_num_nodes):
				if k in g.nodes(): 
					# add edge if does not exist already
					if (j,k) not in g.edges():
						if random.random() <= pattern_edge_add_prob:
							li = random.randint(0,len(edgeLabels)-1)
							g.add_edge(j,k , l = edgeLabels[li])
					# remove edge only when the pattern does not become disconnected
					else:
						if len(g.edges(j))> 1 and len(g.edges(k))> 1 and random.random() <= pattern_edge_rem_prob:
							#copy the label first
							lab = g[j][k]['l']
							# remove the edge
							g.remove_edge(j,k)
							#if the graph became disconnected put the edge back
							if len(nx.connected_components(g))>1:
								g.add_edge(j,k,l=lab)
						

	return g


## reorder the node ids starting from 0
def rearrange_pattern(p):
	node_id_map = {}
	count = 0
	for v in p.nodes():
		node_id_map[v] = count
		count += 1
	g = nx.Graph()
	for v in p.nodes():
		g.add_node(node_id_map[v],l = p.node[v]['l'])
	for e in p.edges():
		g.add_edge(node_id_map[e[0]],node_id_map[e[1]], l = p[e[0]][e[1]]['l'])
	
	return g

#########################################################################################
##########################################################################################
# Random graph generation code starts here

graphList = []
sequenceList = []

# create empty graphs
for i in range(0,numgraphs):
	graphList.append( nx.Graph())

# generate random sequences of different lengths

p_out = open("inserted_patterns.out","w")

pattern_tid = 0

for i in range(0,numsequences):
	p_out.write("sequence "+ str(i)+"\n")

	# clear the sequence list if it is already populated
	sequenceList = []
	
	#insert the first empty pattern in the sequence
	sequenceList.append( nx.Graph())
	
	#select number of nodes randomly
	numnodes = random.randint(pattern_num_nodes_min,pattern_num_nodes_max)	

	#create random nodes and add to the  pattern
	for j in range(0, numnodes):
		# choose label randomly
		li = random.randint(0,len(nodeLabels)-1)
		#add node to the pattern 
		sequenceList[0].add_node(j,l = nodeLabels[li])
	
		#add an edge to one of the existing nodes(to make sure we get a connected subgraph as a pattern)
		if j > 0:
			li = random.randint(0,len(edgeLabels)-1)
			n = random.randint(0,j-1)
			sequenceList[0].add_edge(n,j,l = edgeLabels[li])
		
	
	# add more edges between two nodes with pattern edge probability
	for j in range(0,numnodes-1):
		for k in range(j+1,numnodes):	
			#add an edge with pattern edge probability
			if random.random() <= pattern_edge_add_prob:
				# choose label randomly
				li = random.randint(0,len(edgeLabels)-1)
				sequenceList[0].add_edge(j,k, l = edgeLabels[li])
	
	#choose the sequence length randomly
	seq_len = random.randint(seq_len_min,seq_len_max)
	
	p_out.write("t "+str(pattern_tid)+"\n")	
	pattern_tid += 1
	print_graph(sequenceList[0],p_out)
	
	
	
	# generate other patterns in the sequence
	# skip the first pattern
	for l in range(1,seq_len):
		#tweak the pattern the generate the new sequence	
		#sequenceList.append(augment_pattern(sequenceList[0]))
		#print_graph(sequenceList[l])
		#g = rearrange_pattern(sequenceList[l])
		#print_graph(g)
		
		#modify the previous pattern in the sequence to generate the new pattern
		g = augment_pattern(sequenceList[l-1])
		sequenceList.append(rearrange_pattern(g))
		
		p_out.write("t "+str(pattern_tid)+"\n")
		pattern_tid += 1
		
		print_graph(sequenceList[l],p_out)	 

	#print "Patterns\n"	
	#print_graph(patternList[i])

	#Embed the pattern in random graphs
	sup = random.randint(min_sup,max_sup)
	
	
	# don't allow overlapped embeddinbg of the patterns in the sequences
	if non_overlapping_embedding == 1:
		while True:
			rglist = range(numgraphs-sup+1)
			random.shuffle(rglist)
			rglist = rglist[0:sup]
			rglist.sort()
			flg  = 1
			for b in range(0,sup-1):
				if(rglist[b+1]-rglist[b] < seq_len):
					flg = 0
					break
			if flg == 1:
				break	

	else: # allow overlapped embedding in the sequences
		rglist = range(numgraphs-sup+1)
		random.shuffle(rglist)
		rglist = rglist[0:sup]
	
	#print rglist

	for j in rglist:
		for l in range(0,seq_len):
			 #if (j+l)< numgraphs:
			graphList[j+l] = nx.disjoint_union(sequenceList[l],graphList[j+l])

p_out.close()

# now add random edges to the graphs
for i in range(0,len(graphList)):
	if len(graphList[i].nodes()) == 0:
		maxnodeid = -1
	else:
		maxnodeid = max(graphList[i].nodes())
	
	# select number of nodes randomly
	numnodes = random.randint(graph_num_nodes_min,graph_num_nodes_max)	
	
	for j in range(maxnodeid+1,numnodes ):
		# choose label randomly
		li = random.randint(0,len(nodeLabels)-1)
		# add node to the graph
		graphList[i].add_node(j, l = nodeLabels[li])
	
	# add edges randomly
	for j in range(0,numnodes-1):
		for k in range(j+1, numnodes):
			if (j,k) not in graphList[i].edges():
				prob = random.random()
				# add an edge with graph edge probability
				if prob <= graph_edge_prob:
					#print "adding edge ("+str(j)+","+str(k)+") with prob "+str(prob)
					#choose label randomly
					li = random.randint(0,len(edgeLabels)-1)
					graphList[i].add_edge(j,k, l = edgeLabels[li])


g_out = open("graphs.out","w")
#print "Graphs\n"
i = 0
for G in graphList:
	g_out.write( "t # "+str(i)+"\n");
	print_graph(G,g_out)
	i = i + 1
	#nx.draw(G,labels=l)
	#plt.savefig("graph"+str(i)+".png")
g_out.close()

