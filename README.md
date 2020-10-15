# Must-read papers on Knowledge Graph Embedding (KGE)

## [Content](#content)
- <a href="#distance-based-models">Distance-based Models</a>
- <a href="#tensor-factorization-based-models">Tensor Factorization-based Models</a>
- <a href="#neural-network-based-models">Neural Network-based Models</a>
- <a href="#others">Others</a>



## [Distance-based Models](#content)
1. **Learning Structured Embeddings of Knowledge Bases**. AAAI 2011. [[Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI11/paper/viewFile/3659/3898)]  
    *Antoine Bordes, Jason Weston, Ronan Collobert, Yoshua Bengio*.
    > This paper proposes Structured Embeddings (**SE**), which assumes that the head and tail entities are similar in a relation-specific subspace: 
![formula](https://render.githubusercontent.com/render/math?math=R^{(h)}h=R^{(t)}t).

1. **Translating Embeddings for Modeling Multi-relational Data**. NIPS 2013. [[Paper](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)]  
    *Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana Yakhnenko*.
    > This paper proposes **TransE**, which models the relations are translation operations between head and tail entities: 
![formula](https://render.githubusercontent.com/render/math?math=h%2Br=t).

1. **Knowledge Graph Embedding by Translating on Hyperplanes**. AAAI 2014. [[Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531)]  
    *Zhen Wang, Jianwen Zhang, Jianlin Feng, Zheng Chen*.
    > This paper proposes **TransH** to model the many-to-many property. It interpretes the relations as: 1) relation-specfic hyperplanes; 2) translation operations between head and tail entities projected on the hyperplane: 
![formula](https://render.githubusercontent.com/render/math?math=h_{r}%2Br=t_{r}).

1. **Learning Entity and Relation Embeddings for Knowledge Graph Completion**. AAAI 2015. [[Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523)]  
    *Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, Xuan Zhu*.
    > This paper proposes **TransR/CTransR**, which interpretes the relations as: 1) relation-specfic spaces; 2) translation operations between head and tail entities projected on the hyperplane: 
![formula](https://render.githubusercontent.com/render/math?math=Rh%2Br=Rt).

1. **Knowledge Graph Embedding via Dynamic Mapping Matrix**. ACL 2015. [[Paper](https://www.aclweb.org/anthology/P15-1067.pdf)]  
    *Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, Jun Zhao*.
    > This paper proposes **TransD** to improve TransR/CTransR. It uses two vectors to represent each entity and relation. 
The first one represents the meaning of a(n) entity (relation), the other one is used to construct mapping matrix dynamically. 
Compared with TransR/CTransR, TransD not only considers the diversity of relations, but also entities. It has less parameters and has no matrix-vector multiplication operations.

1. **Learning to Represent Knowledge Graphs with Gaussian Embedding**. CIKM 2015. [[Paper](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Learning%20to%20Represent%20Knowledge%20Graphs%20with%20Gaussian%20Embedding.pdf)]  
    *Shizhu He, Kang Liu, Guoliang Ji and Jun Zhao*.
    > This paper proposes **KG2E** to explicitly model the certainty of entities and relations, which learns the representations of KGs in the space of multi-dimensional Gaussian distributions. Each entity/relation is represented by a Gaussian distribution, where the mean denotes its position and the covariance (currently with diagonal covariance) can properly represent its certainty.

1. **Modeling Relation Paths for Representation Learning of Knowledge Bases**. EMNLP 2015. [[Paper](https://www.aclweb.org/anthology/D15-1082.pdf)]  
    *Yankai Lin, Zhiyuan Liu, Huanbo Luan, Maosong Sun, Siwei Rao, Song Liu*.
    > This paper proposes **pTransE** to consider relation paths as translations between entities for representation learning, and addresses two key challenges: (1) we design a path-constraint resource allocation algorithm to measure the reliability of relation paths; (2) we represent relation paths via semantic composition of relation embeddings.

1. **Composing Relationships with Translations**. EMNLP 2015. [[Paper](https://www.aclweb.org/anthology/D15-1034.pdf)]  
    *Alberto García-Durán, Antoine Bordes, Nicolas Usunier*.
    > This paper proposes **RTransE**, which is an extension of TransE that learns to explicitly model composition of relationship via the addition of their corresponding translation vectors.

1. **From One Point to A Manifold: Knowledge Graph Embedding For Precise Link Prediction**. IJCAI 2016. [[Paper](https://arxiv.org/pdf/1512.04792.pdf)]  
    *Han Xiao, Minlie Huang, Xiaoyan Zhu*.
    > This paper proposes a manifold-based embedding principle (**ManifoldE**) which could be treated as a well-posed algebraic system that expands point-wise modeling in current models to manifold-wise modeling. The score function is designed by measuing the distance of the triple away from a manifold.

1. **A Generative Mixture Model for Knowledge Graph Embedding**. ACL 2016. [[Paper](https://www.aclweb.org/anthology/P16-1219.pdf)]  
    *Han Xiao, Minlie Huang, Xiaoyan Zhu*.
    > This paper proposes **TransG** to address the issue of multiple relation semantics that a relation may have multiple meanings revealed by the entity pairs associated with the corresponding triples. TransG can discover latent semantics for a relation and leverage a mixture of relation-specific component vectors to embed a fact triple.

1. **Knowledge Graph Completion with Adaptive Sparse Transfer Matrix**. AAAI 2016. [[Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11982/11693)]  
    *Guoliang Ji, Kang Liu, Shizhu He, Jun Zhao*.
    > This paper proposes **TranSparse** to model the heterogeneity (some relations link many entity pairs and others do not) and the imbalance (the number of head entities 
      and that of tail entities in a relation could be different) of knowledge graphs. In TranSparse, transfer matrices are replaced by adaptive sparse matrices, 
      whose sparse degrees are determined by the number of entities (or entity pairs) linked by relations.

1. **Knowledge Graph Embedding on a Lie Group**. AAAI 2018. [[Paper](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16227/15885)]  
    *Takuma Ebisu, Ryutaro Ichise*.
    >  This paper proposes **TorusE**, to solve the regularization problem of TransE. The principle of TransE can be defined on any Lie group. A torus, which is one of
        the compact Lie groups, can be chosen for the embedding space to avoid regularization.

1. **Knowledge Graph Embedding by Relational Rotation in Complex Space**. ICLR 2019. [[Paper](https://openreview.net/forum?id=HkgEQnRqYQ)]  [[Code](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)]  
    *Zhiqing Sun, Zhi Hong Deng, Jian Yun Nie, Jian Tang*.
    > This paper proposes **RotatE** to model the three relation patterns: symmetry/antisymmetry, inversion, and composition. RotatE defines each relation as a 
      rotation from the source entity to the target entity in the complex vector space. 

1. **Relation Embedding with Dihedral Group in Knowledge Graph**. ACL 2019. [[Paper](https://arxiv.org/pdf/1906.00687.pdf)]  
    *Canran Xu, Ruijiang Li*.
    > This paper proposes **DihEdral**, named after dihedral symmetry group. This model learns knowledge graph embeddings that can capture
      relation compositions by nature. Furthermore, DihEdral models the relation embeddings parametrized by discrete values, thereby decrease 
      the solution space drastically.   

1. **Multi-relational Poincaré Graph Embeddings**. NeurIPS 2019. [[Paper](https://papers.nips.cc/paper/8696-multi-relational-poincare-graph-embeddings.pdf)]  
    *Ivana Balaževic, Carl Allen, Timothy Hospedales*.
    > This paper proposes **MuRP** to capture multiple simultaneous hierarchies. MuRP embeds multi-relational graphdata in the Poincaré ball model of hyperbolic 
    space.

1. **Quaternion Knowledge Graph Embeddings**. NeurIPS 2019. [[Paper](https://papers.nips.cc/paper/8541-quaternion-knowledge-graph-embeddings.pdf)]  
    *Shuai Zhang, Yi Tay, Lina Yao, Qi Liu*.
    > This paper proposes **QuatE** to model relations as rotations in the quaternion space.

1. **Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction**. AAAI 2020. [[Paper](https://arxiv.org/abs/1911.09419)] [[Code](https://github.com/MIRALab-USTC/KGE-HAKE)]  
    *Zhanqiu Zhang, Jianyu Cai, Yongdong Zhang, Jie Wang*.
    > This paper proposes a novel knowledge graph embedding model---namely, Hierarchy-Aware Knowledge Graph Embedding (**HAKE**)---which maps entities into the polar coordinate system. HAKE
is inspired by the fact that concentric circles in the polar coordinate system can naturally reflect the hierarchy.

## [Tensor Factorization-based Models](#content)
1. **RESCAL:** **A Three-Way Model for Collective Learning on Multi-Relational Data**. ICML 2011. [[Paper](http://www.icml-2011.org/papers/438_icmlpaper.pdf)]  
    *Maximilian Nickel, Volker Tresp, Hans-Peter Kriegel*.
    > This paper proposes **RESCAL** to perform relational learning based on the factorization of a three-way tensor. RESCAL is able to perform collective learning via the latent components of the model and provide an efficient algorithm to compute the factorization.

1. **A Latent Factor Model for Highly Multi-relational Data**. NIPS 2012. [[Paper](http://papers.nips.cc/paper/4744-a-latent-factor-model-for-highly-multi-relational-data.pdf)]  
    *Rodolphe Jenatton, Nicolas L. Roux, Antoine Bordes, Guillaume R. Obozinski*.
    > This paper proposes **LFM** for modeling large multi-relational datasets, with possibly thousands of relations. LFM  is based on a bilinear structure, which captures various orders of interaction of the data, and also shares sparse latent factors across different relations. 

1. **Embedding Entities and Relations for Learning and Inference in Knowledge Bases**. ICLR 2015. [[Paper](https://arxiv.org/abs/1412.6575)]  
    *Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, Li Deng*.
    > This paper proposes **DistMult** to simplify RESCAL by restricting the semantic matching matrices to diagonal matrices.

1. **Holographic Embeddings of Knowledge Graphs**. AAAI 2016. [[Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12484/11828)]  
    *Maximilian Nickel, Lorenzo Rosasco, Tomaso A. Poggio*.
    > This paper proposes holographic embeddings(**HolE**), which employs circular correlation to create compositional representations.

1. **Complex Embeddings for Simple Link Prediction**. ICML 2016. [[Paper](http://proceedings.mlr.press/v48/trouillon16.pdf)]  
    *Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier and Guillaume Bouchard*.
    > This paper proposes **ComplEx**, which maps entity and relation embeddings in complex spaces to effectively capture antisymmetric relations.

1. **Knowledge Graph Completion via Complex Tensor Factorization**. JMLR 2017. [[Paper](https://arxiv.org/pdf/1702.06879.pdf)]  
    *Théo Trouillon, Christopher R. Dance, Johannes Welbl, Sebastian Riedel, Éric Gaussier, Guillaume Bouchard*.
    > This paper is the JMLR version of  **ComplEx**.

1. **Analogical Inference for Multi-relational Embeddings**. ICML 2017. [[Paper](https://arxiv.org/pdf/1705.02426.pdf)]  
    *Hanxiao Liu, Yuexin Wu, Yiming Yang*.
    > This paper proposes **ANALOGY** to model analogical properties of the embedded entities and relations.

1. **On Multi-Relational Link Prediction with Bilinear Models**. AAAI 2018. [[Paper](https://arxiv.org/pdf/1709.04808.pdf)]  
    *Yanjie Wang, Rainer Gemulla, Hui Li*.
    > This paper explores the expressiveness of and the connections between various bilinear models proposed in the literature.

1. **Canonical Tensor Decomposition for Knowledge Base Completion**. ICML 2018. [[Paper](https://arxiv.org/pdf/1806.07297.pdf)] [[Code](https://github.com/facebookresearch/kbc)]  
    *Timothée Lacroix, Nicolas Usunier, Guillaume Obozinski*.
    > This paper motivates and tests a novel regularizer: **N3**, based on tensor nuclear p-norms. Then, this paper presents a reformulation of the problem that makes it invariant to arbitrary choices in the inclusion of predicates or their reciprocals in the dataset.

1. **Embedding for Link Prediction in Knowledge Graphs**. NeurIPS 2018. [[Paper](https://www.cs.ubc.ca/~poole/papers/Kazemi_Poole_SimplE_NIPS_2018.pdf)]  
    *Seyed Mehran Kazemi, David Poole*.
    > This paper presents a simple enhancement of CP (called **SimplE**) to allow the two embeddings of each entity to be learned dependently.

1. **Tensor Factorization for Knowledge Graph Completion**. EMNLP-IJCNLP 2019. [[Paper](https://www.aclweb.org/anthology/D19-1522/)]  
    *Ivana Balazevic, Carl Allen, Timothy Hospedales*.
    > This paper proposes **TuckER**, a relatively straightforward but powerful linear model based on Tucker decomposition of the binary tensor representation of knowledge graph triples.

## [Neural Network-based Models](#content)
1. **Reasoning With Neural Tensor Networks for Knowledge Base Completion**. NIPS 2013. [[Paper](http://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-networks-for-knowledge-base-completion.pdf)]  
    *Richard Socher, Danqi Chen, Christopher D. Manning, Andrew Ng*.
    > This paper introduces an expressive neural tensor network **NTN:**, which is suitable for reasoning over relationships between two entities.

1. **Embedding Projection for Knowledge Graph Completion**. AAAI 2017. [[Paper](https://arxiv.org/pdf/1611.05425.pdf)]  
    *Baoxu Shi, Tim Weninger*.
    > This paper presents a shared variable neural network model called **ProjE** that fills-in missing information in a knowledge graph by learning joint embeddings of the knowledge graph’s entities and edges, and through subtle, but important, changes to the standard loss function.

1. **ConvE:** **Convolutional 2D Knowledge Graph Embeddings**. AAAI 2018. [[Paper](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17366/15884)] [[Code](https://github.com/TimDettmers/ConvE)]  
    *Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, Sebastian Riedel*.

1. **ConvKB:** **A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network**. NAACL-HLT 2018. [[Paper](https://www.aclweb.org/anthology/N18-2053/)] [[Code](https://github.com/daiquocnguyen/ConvKB)]  
    *Dai Quoc Nguyen, Tu Dinh Nguyen, Dat Quoc Nguyen, Dinh Phung*.

1. **R-GCN:** **Modeling Relational Data with Graph Convolutional Networks**. ESWC 2018. [[Paper](https://arxiv.org/pdf/1703.06103.pdf)]  
    *Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling*.

1. **KBGAT:** **Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs**. ACL 2019. [[Paper](https://arxiv.org/pdf/1906.01195.pdf)] [[Code](https://github.com/deepakn97/relationPrediction)]  
    *Deepak Nathani, Jatin Chauhan, Charu Sharma, Manohar Kaul*.

1. **RSN:** **Learning to Exploit Long-term Relational Dependencies in Knowledge Graphs**. ICML 2019. [[Paper](http://proceedings.mlr.press/v97/guo19c/guo19c.pdf)] [[Code](https://github.com/nju-websoft/RSN)]  
    *Lingbing Guo, Zequn Sun, Wei Hu*.

1. **CapsE:** **A Capsule Network-based Embedding Model for Knowledge Graph Completion and Search Personalization**. NAACL-HIT 2019. [[Paper](https://www.aclweb.org/anthology/N19-1226/)] [[Code](https://github.com/daiquocnguyen/CapsE)]  
    *Dai Quoc Nguyen, Thanh Vu, Tu Dinh Nguyen, Dat Quoc Nguyen, Dinh Q. Phung*.

1. **InteractE:** **InteractE: Improving Convolution-based Knowledge Graph Embeddings by Increasing Feature Interactions**. AAAI 2020. [[Paper](https://arxiv.org/pdf/1911.00219.pdf)]    
    *Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, Nilesh Agrawal, Partha Talukdar*.

## [Others](#content)
1. **You CAN Teach an Old Dog New Tricks! On Training Knowledge Graph Embeddings**. ICLR 2020. [[Paper](https://openreview.net/pdf?id=BkxSmlBFvr)] [[Code](https://github.com/uma-pi1/kge)]  
    *Daniel Ruffinelli, Samuel Broscheit, Rainer Gemulla*.

1. **A Re-evaluation of Knowledge Graph Completion Methods**. ACL 2020. [[Paper](https://arxiv.org/abs/1911.03903)] [[Code](https://github.com/svjan5/kg-reeval)]  
    *Zhiqing Sun, Shikhar Vashishth, Soumya Sanyal, Partha Talukdar, Yiming Yang*.
