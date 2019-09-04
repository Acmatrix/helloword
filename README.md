# 写的很好，留个纪念。
异常监测方法
本文来源于知乎话题“异常检测（anomaly/ outlier detection）领域还有那些值得研究的问题？” 

原文链接：
https://www.zhihu.com/question/324999831

基于图像的异常检测，比如工业上用的表面瑕疵检测（surface defect detection）发展到了哪一步？还有无进一步研究的必要？对讯号异常（series anomaly）的呢？
极市为大家精选了2位知乎大佬的高质量回答，供大家参考。

微调@知乎

我觉得异常检测可以被理解为一种在「无监督或者弱监督下的非平衡数据下的多分类，且要求一定的解释性」的任务，且往往异常点（不平衡数据中较少的部分）对我们更为重要。和非平衡学习（imbalanced learning）不同异常检测一般是无监督的，和普通的二分类问题也不大相同，因为异常检测往往看似是二分类，但其实是多分类（造成异常的原因各不相同）。问题的核心就在于我们既不知道事实上有多少类别（class），也没有真实的标签（ground truth），在这种情况下异常检测的效果往往不尽人意。

说到异常检测，一般会先从无监督说起。传统的方法就是衡量相似度（proximity）比如距离[1]、密度[2]、角度[3]、隔离所需的难度[4]、基于簇的方法[5]等，这些算法在低维上其实表现都接近，因为核心假设都是“异常点的表示与正常点不同，是少数派”。但大部分类似的算法都会面临维数灾难（the curse of dimensionality），即常见的相似性度量（比如欧氏距离）在高维数据上往往会失效[6][7]。为了解决这个问题，人们提出了很多方法包括：

降维或者特征选择[8]
subspace方法，比如在多个低维空间上做检测再合并，比如random projection（随机产生多个子空间并在每个子空间上单独建模，feature bagging）和random forest很像
用graph来表示关系，提取特征[9]，但往往维度会继续升高
找intrinsic dimensionality以及其他度量方法，如reversenearest neighbors[10]

其实换句话说，对于高维数据，核心目的都还是想找到一个好的空间/表示[11]，之后找异常就变成了衡量相似度的简单问题。高维数据所带来的另一个问题是可扩展性（scalability）。众所周知，衡量相似度的运算开销是很大的，大部分距离度量的复杂度以上，在这种情况下利用数据结构（比如kd tree）进行优化或者dynamic programming来降低复杂度也是常见的探索方向。最理想的情况还是控制维度，找到更好的数据表示，因为这才是问题的根本。

为了找到好的表示（representation），或者单纯只是更简单的、tractable的表示，线性降维有用PCA的，非线性的有用autoencoder的[12]，人们的核心假设都是降维模型所找到的低维空间主要受到正常点的影响，因此异常点距离所找到的低维空间的距离更远。在这个基础上人们也引入了variational autoencoder（VAE），后来也有用GAN的方法[13]。对于高维数据而言，往往一个模型是不够的，比如前面的feature bagging（类比于监督学习中的随机森林）会建立多个模型[14]，因此就会涉及到模型合并的问题，也就是集成学习，这个话题主要是13年以后开始比较火。考虑到无监督学习的特性，集成异常检测（outlier ensembles）[15] 一般是平行式（parallellearning）的比如求平均，bagging类型为主流，而非序列式（sequential）如boosting。现在的主流集成异常检测因此性能还是有限的，毕竟取多个模型的均值或者最大值是现阶段的可行方法。如果要做序列式或者要在集成过程中做评估，那就需要生成伪标签[16]，这类方法现在依然是heuristic的，缺乏性能保证。如果要做stacking可能更为复杂，现在只有一些非常初步的探索[11]。

前文说了，异常检测往往是无监督学习，因此这些方法都是heuristic，一般缺乏性能保障。假设运气特别好，我们发现了一个有效的异常检测算法暂时不为性能担忧，那么就会自然的想「异常检测的规则是什么，如何解释」。据我所知现在关于可解释性的主流方法还是在局部空间或者contextual based 方法[17][18]，也有提供直观图像的方法[19]，也有通过找subspace的方法[20]，通过找低维空间（或者特征）来解释的（其实也属于前面的方法）。大部分解释性主要是考虑如何调整特征使得一个异常点成为正常点，那么这就是决定因素。另外一种思路就是不如我们让人类帮着解释吧，这就进入了众包（crowd sourcing）和主动学习（active learning）的范围，让人类在修正错误预测的同时同时提供一定的解释[21]，未来和HCI的交叉是大趋势。

在互动型异常检测（interactive）的范畴下，我们也可以把异常检测看做是一个排序问题（根据异常值/确定性的）排序，因此传统的排序算法也被引入了这个范围[22]。不难看出，这可以被看做是一个exploration和exploitation的问题，那么还可以考虑bandit的算法[23]比如UCB等等。

说到bandit问题，我们就会想到贝叶斯优化，这就引入了另一个问题，就是我们用贝叶斯做自动调参，能不能把这个方法应用在异常检测调参中呢？难点在于如何在无监督或者说半监督的范畴下达成这个目标，可能需要和前面提到的interactive或者active learning相结合。

另一个常见的问题就是异常特征随时间的变化，也就是在evolving data上的异常检测[24][25]。在更传统的机器学习领域，我们一般把这个叫做concept drift。

这些都是最纯粹的异常检测算法层面上的问题，拓展到文字[26]、结构数据、图像、时间序列[27]等不同的范畴上每个问题都又有大量的可做的内容，所以可以做的方向很多。

以我粗浅的了解，异常检测能做的方向很多，尽管大多都不容易啃，毕竟无监督、不平衡的假设太强了，正如前文我提到了可以把异常检测看做是一种「无监督或者弱监督下的非平衡数据多分类分体并要求一定的解释性」。现阶段无监督的各个方向已经有了很多探索，更大的机会可能在半监督/强化学习/弱监督上，毕竟监督学习的成本对于异常检测而言可能还是太高了一点。当然，这只是我的个人理解，有很强的主观性，请谨慎参考（且以防撞车）。

以上提到的所有文章（PDF）与参考均可以在GitHub的资料汇总Repo上找到，欢迎关注、订阅、Star~
异常检测资源汇总：
https://github.com/yzhao062/anomaly-detection-resources


参考
1. Ramaswamy,S., Rastogi, R. and Shim, K., 2000, May. Efficient algorithms for miningoutliers from large data sets. ACM SIGMOD Record, 29(2), pp. 427-438.
2. Breunig,M.M., Kriegel, H.P., Ng, R.T. and Sander, J., 2000, May. LOF: identifyingdensity-based local outliers. ACM SIGMOD Record, 29(2), pp. 93-104.
3. Kriegel,H.P. and Zimek, A., 2008, August. Angle-based outlier detection inhigh-dimensional data. In KDD '08, pp. 444-452. ACM.
4. Liu,F.T., Ting, K.M. and Zhou, Z.H., 2008, December. Isolation forest. InInternational Conference on Data Mining, pp. 413-422. IEEE.
5. He, Z.,Xu, X. and Deng, S., 2003. Discovering cluster-based local outliers. PatternRecognition Letters, 24(9-10), pp.1641-1650.
6. Zimek,A., Schubert, E. and Kriegel, H.P., 2012. A survey on unsupervised outlierdetection in high‐dimensional numerical data. Statistical Analysis andData Mining: The ASA Data Science Journal, 5(5), pp.363-387.
7. Ro, K.,Zou, C., Wang, Z. and Yin, G., 2015. Outlier detection for high-dimensionaldata. Biometrika, 102(3), pp.589-599.
8.  Pang,G., Cao, L., Chen, L. and Liu, H., 2017, August. Learning homophily couplingsfrom non-iid data for joint feature selection and noise-resilient outlierdetection. In Proceedings of the 26th International Joint Conference onArtificial Intelligence (pp. 2585-2591). AAAI Press.
9. Akoglu,L., Tong, H. and Koutra, D., 2015. Graph based anomaly detection anddescription: a survey. Data Mining and Knowledge Discovery, 29(3), pp.626-688.
10. Radovanović, M.,Nanopoulos, A. and Ivanović, M., 2015. Reverse nearest neighbors in unsuperviseddistance-based outlier detection. IEEE transactions on knowledge and dataengineering, 27(5), pp.1369-1382.
11. Zhao, Y. and Hryniewicki, M.K., 2018, July. XGBOD:improving supervised outlier detection with unsupervised representationlearning. In 2018 International Joint Conference on Neural Networks (IJCNN).IEEE.
12. Zong,B., Song, Q., Min, M.R., Cheng, W., Lumezanu, C., Cho, D. and Chen, H., 2018.Deep autoencoding gaussian mixture model for unsupervised anomaly detection.International Conference on Learning Representations (ICLR).
13. Liu,Y., Li, Z., Zhou, C., Jiang, Y., Sun, J., Wang, M. and He, X., 2019. GenerativeAdversarial Active Learning for Unsupervised Outlier Detection. IEEEtransactions on knowledge and data engineering.
14. Pang,G., Cao, L., Chen, L. and Liu, H., 2018. Learning Representations ofUltrahigh-dimensional Data for Random Distance-based Outlier Detection. In 24thACM SIGKDD International Conference on Knowledge Discovery and Data mining(KDD). 2018.
15.Aggarwal,C.C., 2013. Outlier ensembles: position paper. ACM SIGKDD ExplorationsNewsletter, 14(2), pp.49-58.
16. Zhao,Y., Nasrullah, Z., Hryniewicki, M.K. and Li, Z., 2019, May. LSCP: Locallyselective combination in parallel outlier ensembles. In Proceedings of the 2019SIAM International Conference on Data Mining (SDM), pp. 585-593. Society forIndustrial and Applied Mathematics.
17. Liu,N., Shin, D. and Hu, X., 2017. Contextual outlier interpretation. InInternational Joint Conference on Artificial Intelligence (IJCAI-18),pp.2461-2467.
18. Tang,G., Pei, J., Bailey, J. and Dong, G., 2015. Mining multidimensional contextualoutliers from categorical relational data. Intelligent Data Analysis, 19(5),pp.1171-1192.
19. Gupta,N., Eswaran, D., Shah, N., Akoglu, L. and Faloutsos, C., Beyond OutlierDetection: LookOut for Pictorial Explanation. ECML PKDD 2018.
20. Macha,M. and Akoglu, L., 2018. Explaining anomalies in groups with characterizingsubspace rules. Data Mining and Knowledge Discovery, 32(5), pp.1444-1480.
21. Siddiqui,M.A., Fern, A., Dietterich, T.G. and Wong, W.K., 2019. Sequential FeatureExplanations for Anomaly Detection. ACM Transactions on Knowledge Discoveryfrom Data (TKDD), 13(1), p.1.
22. Lamba,H. and Akoglu, L., 2019, May. Learning On-the-Job to Re-rank Anomalies fromTop-1 Feedback. In Proceedings of the 2019 SIAM International Conference onData Mining (SDM), pp. 612-620. Society for Industrial and Applied Mathematics.
23. Ding,K., Li, J. and Liu, H., 2019, January. Interactive anomaly detection onattributed networks. In Proceedings of the Twelfth ACM International Conferenceon Web Search and Data Mining, pp. 357-365. ACM.
24. Salehi,Mahsa & Rashidi, Lida. (2018). A Survey on Anomaly detection in EvolvingData: [with Application to Forest Fire Risk Prediction]. ACM SIGKDDExplorations Newsletter. 20. 13-23.
25. Manzoor,E., Lamba, H. and Akoglu, L. Outlier Detection in Feature-Evolving DataStreams. In 24th ACM SIGKDD International Conference on Knowledge Discovery andData mining (KDD). 2018.
26. Kannan,R., Woo, H., Aggarwal, C.C. and Park, H., 2017, June. Outlier detection fortext data. In Proceedings of the 2017 SIAM International Conference on DataMining, pp. 489-497. Society for Industrial and Applied Mathematics.
27. Gupta,M., Gao, J., Aggarwal, C.C. and Han, J., 2014. Outlier detection for temporaldata: A survey. IEEE Transactions on Knowledge and Data Engineering, 26(9),pp.2250-2267.
