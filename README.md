------

# Embedding survey:

Network Representation Learning: A Survey 

TKDE 18 A Survey on Network Embedding 

Arxiv 18 A Tutorial on Network Embeddings 

Arxiv 19 A Comprehensive Survey on Graph Neural Networks 

Arxiv 19 Graph Neural Networks A Review of Methods and Applications 

TKDE 17 A Comprehensive Survey of Graph Embedding Problems, Techniques and Applications 

 IEEE 17 Representation Learning on Graphs: Methods and Applications 

   https://zhuanlan.zhihu.com/p/42022918

# Knowledge Base Embedding:

NIPS 13 Translating Embeddings for Modeling Multi-relational Data   (TransE) 

​     https://github.com/thunlp/TensorFlow-TransX 

​     https://github.com/thunlp/KB2E 

​     https://github.com/ZichaoHuang/TransE 

AAAI 14 Knowledge Graph Embedding by Translating on Hyperplanes (TransH) 

AAAI 15 Learning entity and relation embeddings for knowledge graph completion  (TransR) 

------

# Embedding in NLP:

NIPS 2013 Distributed Representations of Words and Phrases and their Compositionality     (word2vec) 

​     https://www.tensorflow.org/tutorials/representation/word2vec 

​     https://code.google.com/archive/p/word2vec/ 

EMNLP 14 GloVe: Global Vectors for Word Representation 

​     https://github.com/maciejkula/glove-python 

------

# Network Embedding:

## (1) Homogeneous Network Embedding:

KDD 14 Deepwalk: Online learning of social representations   (Deepwalk) 

​     random walk + skip-gram 

​     https://sites.google.com/site/bryanperozzi/projects/deepwalk 

​     https://github.com/phanein/deepwalk 

WWW 15 Line: Large-scale information network embedding  （Line） 

​     preserving both first-order and second-order proximities 

​     https://github.com/tangjianpku/LINE 

KDD 16 node2vec: Scalable Feature Learning for Networks     (Node2vec)   

​     biased random walk procedure to efficiently explore diverse neighborhoods 

​     http://snap.stanford.edu/node2vec 

​     https://github.com/aditya-grover/node2vec 

KDD 16 Structural deep network embedding       (SDNE) 

​     structure-preserving embedding method to capture first and second order structural information of the network. 

KDD 17 struc2vec: Learning Node Representations from Structural Identity  (struc2vec) 

​     https://github.com/aliysefian/struct2vecttrue 

------

Embedding: Factorization

ICDM 10 Factorization machines 

IJCAI 17 DeepFM: a factorization-machine based neural network for CTR prediction 

KDD 18 xDeepFM: Combining explicit and implicit feature interactions for recommender systems 

WSDM 18 Network embedding as matrix factorization: Unifying deepwalk, line, pte, and node2vec (NetMF) 

------

## (2) Heterogeneous Network Embedding:

   Characteristics: multiple types of nodes and edges 

KDD 15 Pte: Predictive text embedding through large-scale heterogeneous text networks   

​     constructs large-scale heterogeneous text network from labeled information and different levels of word co-occurrence information, 

KDD 17 metapath2vec: Scalable representation learning for heterogeneous networks       (metapath2vec) 

​     formalizes meta-path based random walk to construct the heterogeneous neighborhood of a node and then     

leverages a heterogeneous skip-gram model to per- form node embeddings. 

​     https://ericdongyx.github.io/metapath2vec/m2v.html 

​     https://github.com/apple2373/metapath2vec 

​     https://github.com/prakhar-agarwal/metapath2Vec 

KDD 17 Meta-Graph Based Recommendation Fusion over Heterogeneous Information Networks 

Algorithm 18 Learning Heterogeneous Knowledge Base Embeddings for Explainable Recommendation.  (CFKG) 

KDD 18 Leveraging Meta-path based Context for Top-N Recommendation with A Neural Co-Attention Model  (MCRec) 

​     https://github.com/librahu/MCRec 

TKDE 18 Heterogeneous information network embedding for recommendation  (HERec) 

​     meta-path based random walk strategy to generate meaningful node sequences to learn network embeddings that are first transformed by a set of fusion functions and subsequently integrated into an extended matrix factorization (MF) model. 

​     https://github.com/librahu/HERec 

------

## (3) Multiplex Heterogeneous Network Embedding:

​     Characteristics: multiple types of proximities between nodes 

ICDMW 17  Principled multilayer network embedding     (PMNE) 

​     three methods to project a multiplex network into a continuous vector space. 

CIKM 17 An Attention-based Collaboration Framework for Multi-View Network Representation Learning.   (MVE) 

​     embeds networks with multiple views in a single collab- orated embedding using attention mechanism. 

IJCAI 18 Scalable Multiplex Network Embedding   (MNE) 

​     uses one common embedding and several additional embeddings of each edge type for each node 

​     https://github.com/HKUST-KnowComp/MNE 

Arxiv 18 mvn2vec: Preservation and Collaboration in Multi-View Network Embedding 

​     explores the feasibility to achieve better embedding results by simultaneously modeling preservation and collaboration to represent semantic meanings of edges in different views respectively. 

KDD 19 Representation Learning for Attributed Multiplex Heterogeneous Network 

​     https://github.com/THUDM/GATNE 

WWW 19: MARINE: Multi-relational Network Embeddings with Relational Proximity and Node Attributes 

------

## (4) Attributed Network Embedding:

​     Characteristics: network topological structure and node attribute proximity can be preserved in such repre- sentations 

KDD 15 Heterogeneous network embedding via deep architectures    (HNE) 

​     jointly consider contents and topological structures in networks 

IJCAI 15 Network representation learning with rich text information   (TADW) 

​     incorporates text features of vertices into network representation learning under the framework of matrix factorization 

WSDM 17 Label informed attributed network embedding     (LANE) 

​     smoothly incorporates label information into the attributed network embedding while preserving their cor- relations 

SDM 17 Accelerated attributed network embedding       (AANE) 

​     joint learning process to be done in a distributed manner for accelerated attributed network embed- ding 

TKDE 18 Attributed social network embedding    (SNE) 

​     embedding social networks by capturing both the structural proximity and attribute proximity 

IJCAI 18 Deep Attributed Network Embedding     (DANE) 

​     capture the high nonlinearity and preserve various proximities in both topological structure and node attributes 

IJCAI 18 ANRL: Attributed Network Representation Learning via Deep Neural Networks     (ANRL) 

​     uses a neighbor enhancement autoencoder to model the node attribute information and an attribute-aware skip-gram model based on the attribute encoder to capture the network structure. 

------

## (5) Attributed Multiplex Heterogeneous Network:

KDD 19 Representation Learning for Attributed Multiplex Heterogeneous Network 

   https://github.com/cenyk1230/GATNE 

------

# GNN:

Survey: 

​     https://github.com/nnzhan/Awesome-Graph-Neural-Networks 

## (1) GNN 

IEEE 2009 The Graph Neural Network Model 

## (2) GGS-NN: Gated Graph Sequence Neural Networks 

ICLR 16  Gated Graph Sequence Neural Networks 

​     https://github.com/yujiali/ggnn   

​      https://github.com/JamesChuanggg/ggnn.pytorch 

ACL 18 Graph-to-Sequence Learning using Gated Graph Neural Networks      

​      https://github.com/beckdaniel/acl2018_graph2seq.git 

ACL 18 Semi-supervised User Geolocation via Graph Convolutional Networks 

​     https://github.com/afshinrahimi/geographconv 

AAAI 19 Session-based Recommendation with Graph Neural Networks 

​     https://github.com/CRIPAC-DIG/SR-GNN 

## (3) GCN 

ICLR 2017 Semi-Supervised Classification with Graph Convolutional Networks 

​     https://github.com/tkipf/gcn 

KDD 18 Graph Convolutional Neural Networks for Web-Scale Recommender Systems     

AAAI 19 SocialGCN: An Efficient Graph Convolutional Network based Model for Social Recommendation 

CIKM 18 Multiresolution Graph Attention Networks for Relevance Matching 

## (4) Large-scale GCN 

NIPS 17 Inductive Representation Learning on Large Graphs     GraphSAGE 

   https://github.com/williamleif/GraphSAGE

​      https://github.com/williamleif/graphsage-simple 

KDD 18 Large-Scale Learnable Graph Convolutional Networks 

   https://github.com/divelab/lgcn/

## (5) GAT 

ICLR 18 Graph Attention Networks 

​     https://github.com/PetarV-/GAT 

KDD 18 DeepInf: Social Influence Prediction with Deep Learning 

   https://github.com/xptree/DeepInf

WWW 19 Graph Neural Networks for Social Recommendation 

WSDM 19 Session-based Social Recommendation via Dynamic Graph Attention Networks 

WWW 19 Heterogeneous Graph Attention Network 

​     https://github.com/Jhy1993/HAN 

Model MultiRelation: 

   ICLR 19 Relational Graph Attention Networks 

   ESWC 18 Modeling Relational Data with Graph Convolutional Networks 

​     

Graph: 

   NIPS 18: Hierarchical Graph Representation Learning with Differentiable Pooling 

Degree: 

​     KDD 19 DEMO-Net: Degree-specific Graph Neural Networks for Node and Graph Classification 

------

# Wor2Vec for Rec:

KDD 15 E-commerce in Your Inbox: Product Recommendations at Scale 

RecSys 16 Meta-Prod2Vec : Product Embeddings Using Side-Information for Recommendation 

​     https://github.com/labdac/Meta-Prod2Vec 

WSDM 18 A Path-constrained Framework for Discriminating Substitutable and Complementary Products in E-commerce 

KDD 18 Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba 

------

# Knowledge Base Embedding for Rec:

KDD 16 Collaborative Knowledge Base Embedding for Recommender Systems     (CKE) 

RecSys 17 Translation-based Recommendation 

CIKM 18 RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems.     (RippleNet) 

​     https://github.com/hwwang55/RippleNet 

WWW 19 Unifying Knowledge Graph Learning and Recommendation: Towards a Better Understanding of User Preferences 

​     https://github.com/TaoMiner/joint-kg-recommender 

WWW 19 Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation 

------

GNN for Rec:

WWW 19 Graph Neural Networks for Social Recommendation 

SIGIR 19 Graph Intention Network for Click-through Rate Prediction in Sponsored Search 

IJCAI 19 Graph Contextualized Self-Attention Network for Session-based Recommendation 

------

# Knowledge Base + GNN for Rec:

KDD 19 KGAT: Knowledge Graph Attention Network for Recommendation         

​     https://github.com/xiangwang1223/knowledge_graph_attention_network 

WWW 19 Knowledge Graph Convolutional Networks for Recommender Systems 

KDD 19 Knowledge-aware Graph Neural Networks with Label Smoothness Regularization for Recommender Systems 

​     https://github.com/hwwang55/KGNN-LS 

------

Outfit:

WWW 19 Dressing as a Whole: Outfit Compatibility Learning Based on Node-wise Graph Neural Networks 

------

# Embedding for RecSys:

CIKM 15 Semantic Path based Personalized Recommendation on Weighted Heterogeneous Information Networks 

RecSys 15 ensemble learning with categorical features 

KDD 16 Collaborative Knowledge Base Embedding for Recommender Systems 

IJCAI 16 Sherlock: Sparse Hierarchical Embeddings for Visually-aware One-class Collaborative Filtering 

IJCAI 16 Tri-Party Deep Network Representation 

RecSys 16 Music Playlist Recommendation via Preference Embedding 

RecSys 16 Query-based Music Recommendations via Preference Embedding 

RecSys 16 Factorization Meets the Item Embedding: Regularizing Matrix Factorization with Item Co-occurrence 

ICDM16 Learning Compatibility Across Categories for Heterogeneous Item Recommendation 

KDD 17 Embedding-based News Recommendation for Millions of Users 

IJCAI 17 MRLR Multi-level Representation Learning for Personalized Ranking in Recommendation 

RecSys 17 Translation-based Recommendation 

SIGIR 17 Attentive Collaborative Filtering: Multimedia Recommendation with Item- and Component-Level Attention 

SIGIR 17 Embedding Factorization Models for Jointly Recommending Items and User Generated Lists 

AAAI 17 Scalable Graph Embedding for Asymmetric Proximity 

WSDM 18 Multi-Dimensional Network Embedding with Hierarchical Structure 

------

# Library

  Tensorflow and Sonnet https://github.com/deepmind/graph_nets 

  Pytorch: https://github.com/rusty1s/pytorch_geometric 

​          https://github.com/dmlc/dgl 

  Ali Euler: https://github.com/alibaba/euler 

  AliGraph: https://github.com/alibaba/AliGraph  https://arxiv.org/pdf/1902.08730.pdf 

  Facebook Pytorch-BigGraph: https://github.com/facebookresearch/PyTorch-BigGraph 