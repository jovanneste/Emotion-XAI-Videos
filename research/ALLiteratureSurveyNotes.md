### Active Learning Literature Survey 
Burr Settles 

## Query strategy frameworks 

Ways to evaluate the informativeness of unlabeled instances 

We will ignore random sampling

# Uncertainty Sampling 

D. Lewis and W. Gale. A sequential algorithm for training text classifiers. In
Proceedings of the ACM SIGIR Conference on Research and Development in Information Retrieval, pages 3–12. ACM/Springer, 1994.

Least confident: An active learner queries the instances about which it is least certain how to label. For example, for binary classification, an active learner would query the instances whose posterior probability of being positive if nearest to 0.5. Can be efficienty computed.


Entropy: The most popular uncertainty sampling strategy uses entropy as an uncertainty measure.

C.E. Shannon. A mathematical theory of communication. Bell System Technical Journal, 27:379–423,623–656, 1948.

HEATMAPS

# Query-By-Committee

Theorectrically motivated query selection framework

H.S. Seung, M. Opper, and H. Sompolinsky. Query by committee. In Proceed- ings of the ACM Workshop on Computational Learning Theory, pages 287–294, 1992.

# Expected Model Change 

Select the instance that would impact the greatest change to the current model if we knew its label.

Expected gradient length (EGL)

B. Settles, M. Craven, and S. Ray. Multiple-instance active learning. In Advances in Neural Information Processing Systems (NIPS), volume 20, pages 1289– 1296. MIT Press, 2008b.

# Density-Weighted Methods

Addresses problems with previous frameworks - if least certain instance lies on fication boundary but is not representative of other instances in the distribution. 

B. Settles and M. Craven. An analysis of active learning strategies for sequence labeling tasks. In Proceedings of the Conference on Empirical Methods in Nat- ural Language Processing (EMNLP), pages 1069–1078. ACL Press, 2008.