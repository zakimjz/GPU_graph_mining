# GPU_graph_mining
GPU graph mining

CUDA implementation of graph mining on GPUs as described in the paper:

Robert Kessl, Nilothpal Talukder, Pranay Anchuri, and Mohammed J. Zaki. 
Parallel graph mining with GPUs. 
Proceedings of the 3rd International Workshop on Big Data, Streams and Heterogeneous Source Mining: 
Algorithms, Systems, Programming Models and Applications (with SIGKDD'14), 
Journal of Machine Learning Research: Conference and Workshop Proceedings, 
36:1–36, 2014. URL: http://jmlr.org/proceedings/papers/v36/.

In this paper, we propose a novel approach for parallel graph mining on GPUs, which have emerged as a relatively cheap but powerful architecture for general purpose computing. However, the thread-model for GPUs is different from that of CPUs, which makes the parallelization of graph mining algorithms on GPUs a challenging task. We investigate the major challenges for GPU-based graph mining. We perform extensive experiments on several real-world and synthetic datasets, achieving speedups up to 9 over the sequential algorithm
