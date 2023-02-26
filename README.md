# bayesian-materials-science-benchmarking

I conducted a study to evaluate the performance of Bayesian Optimization (BO) algorithms for general optimization across a wide range
of experimental materials science domains. I used six different materials systems, including laser-induced graphene, carbon nanotube polymer blends,
silver nanoparticles, lead-halide perovskites, as well as additively manufactured polymer structures and shapes. I defined acceleration and
enhancement metrics for general materials optimization objectives and found that Gaussian Process (GP) with anisotropic kernels and
Random Forests (RF) had comparable performance in BO as surrogate models, both outperforming GP with isotropic kernels. GP with
anisotropic kernel was more robust as a surrogate model across most design spaces, while RF is a close alternative with benefits of
being free of distribution assumptions, having lower time complexities, and requiring less effort in initial hyperparameter selection.
The study raises awareness about the benefits of using GP with anisotropic kernels over GP with isotropic kernels in future materials
optimization campaigns.

For reuse for code and materials datasets in this repo, please cite both this study and the following authors for sharing their datasets.

Materials datasets used to benchmark BO performance in this repository are provided by:

(1) Laser-induced Graphene (PI) dataset
```
@incollection{kotthoff2022optimizing,
  title={Optimizing Laser-Induced Graphene Production},
  author={Kotthoff, Lars and Dey, Sourin and Heil, Jake and Jain, Vivek and Muller, Todd and Tyrrell, Alexander and Wahab, Hud and Johnson, Patrick},
  booktitle={PAIS 2022},
  pages={31--44},
  year={2022},
  publisher={IOS Press}
}

link: https://ebooks.iospress.nl/volumearticle/60357

@article{wahab2020lig,
   author = {Wahab, Hud and Jain, Vivek and Tyrrell, Alexander Scott and Seas, Michael Alan and Kotthoff, Lars and Johnson, Patrick Alfred},
   title = {Machine-learning-assisted fabrication: Bayesian optimization of laser-induced graphene patterning using in-situ Raman analysis},
   journal = {Carbon},
   volume = {167},
   pages = {609-619},
   ISSN = {0008-6223},
   DOI = {https://doi.org/10.1016/j.carbon.2020.05.087},
   url = {http://www.sciencedirect.com/science/article/pii/S0008622320305285},
   year = {2020},
   type = {Journal Article}
}
link: http://www.sciencedirect.com/science/article/pii/S0008622320305285
```
(2) Crossed barrel dataset
```
@article{gongora2020bayesian,
  title={A Bayesian experimental autonomous researcher for mechanical design},
  author={Gongora, Aldair E and Xu, Bowen and Perry, Wyatt and Okoye, Chika and Riley, Patrick and Reyes, Kristofer G and Morgan, Elise F and Brown, Keith A},
  journal={Science advances},
  volume={6},
  number={15},
  pages={eaaz1708},
  year={2020},
  publisher={American Association for the Advancement of Science}
}

link: https://advances.sciencemag.org/content/6/15/eaaz1708
```

(3) Perovskite dataset
```
 @article{sun2021data,
   title={A data fusion approach to optimize compositional stability of halide perovskites},
   author={Sun, Shijing and Tiihonen, Armi and Oviedo, Felipe and Liu, Zhe and Thapa, Janak and Zhao, Yicheng and Hartono, Noor Titan P and Goyal, Anuj and Heumueller, Thomas and Batali, Clio and others},
   journal={Matter},
   volume={4},
   number={4},
   pages={1305--1322},
   year={2021},
   publisher={Elsevier}
 }
 link: https://www.sciencedirect.com/science/article/pii/S2590238521000084
 ```
 (4) AutoAM dataset
 ```
 @article{deneault2021toward,
   title={Toward autonomous additive manufacturing: Bayesian optimization on a 3D printer},
   author={Deneault, James R and Chang, Jorge and Myung, Jay and Hooper, Daylond and Armstrong, Andrew and Pitt, Mark and Maruyama, Benji},
   journal={MRS Bulletin},
   pages={1--10},
   year={2021},    
   publisher={Springer}
 }
 
 link: https://link.springer.com/article/10.1557/s43577-021-00051-1
 ```
 (5) P3HT/CNT dataset

```
@article{bash2021multi,
title={Multi-Fidelity High-Throughput Optimization of Electrical Conductivity in P3HT-CNT Composites},
author={Bash, Daniil and Cai, Yongqiang and Chellappan, Vijila and Wong, Swee Liang and Yang, Xu and Kumar, Pawan and Tan, Jin Da and Abutaha, Anas and Cheng, Jayce JW and Lim, Yee-Fun and others},
journal={Advanced Functional Materials},
pages={2102606},
year={2021},
publisher={Wiley Online Library}
}

link: https://onlinelibrary.wiley.com/doi/abs/10.1002/adfm.202102606
```
(6) AgNP dataset
```
@article{mekki2021two,
  title={Two-step machine learning enables optimized nanoparticle synthesis},
  author={Mekki-Berrada, Flore and Ren, Zekun and Huang, Tan and Wong, Wai Kuan and Zheng, Fang and Xie, Jiaxun and Tian, Isaac Parker Siyu and Jayavelu, Senthilnath and Mahfoud, Zackaria and Bash, Daniil and others},
  journal={npj Computational Materials},
  volume={7},
  number={1},
  pages={1--10},
  year={2021},
  publisher={Nature Publishing Group}
}

link: https://www.nature.com/articles/s41524-021-00520-w
```
