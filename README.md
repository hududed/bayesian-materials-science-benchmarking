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
