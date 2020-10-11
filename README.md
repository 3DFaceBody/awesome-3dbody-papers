# Awesome 3D Body Papers

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> An awesome & curated list of papers about 3D human body.

-----

## Table of Contents

- [Body Model](#body-model)
- [Body Pose](#body-pose)
- [Naked Body Mesh](#naked-body-mesh)
- [Clothed Body Mesh](#clothed-body-mesh)
- [Human Motion](#human-motion)
- [Human-Object Interaction](#human-object-interaction)
- [Animation](#animation)
- [Cloth/Try-On](#cloth/try-on)
- [Dataset](#dataset)

-----


## Body Model


[SCAPE: Shape Completion and Animation of People](http://robots.stanford.edu/papers/anguelov.shapecomp.pdf). SIGGRAPH, 2005. [[Page]](http://robotics.stanford.edu/~drago/Projects/scape/scape.html) 

[SMPL: A Skinned Multi-Person Linear Model](http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf). SIGGRAPH Asia, 2015. [[Page]](https://smpl.is.tue.mpg.de) [[Code]](https://github.com/vchoutas/smplx)

[Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf). CVPR, 2019. [[Page]](https://smpl-x.is.tue.mpg.de) [[Code]](https://github.com/vchoutas/smplify-x)

[SoftSMPL: Data-driven Modeling of Nonlinear Soft-tissue Dynamics for Parametric Humans](https://arxiv.org/pdf/2004.00326). Eurographics, 2020. [[Page]](http://dancasas.github.io/projects/SoftSMPL) 

[STAR: Sparse Trained Articulated Human Body Regressor](https://arxiv.org/pdf/2008.08535). ECCV, 2020. [[Page]](http://star.is.tue.mpg.de) [[Code]](https://github.com/ahmedosman/STAR)

[GHUM & GHUML: Generative 3D Human Shape and Articulated Pose Models](https://arxiv.org/pdf/2008.08535). CVPR (Oral), 2020.  [[Code]](https://github.com/google-research/google-research/tree/master/ghum)


## Body Pose


[MotioNet: 3D Human Motion Reconstruction from Monocular Video with Skeleton Consistency](https://arxiv.org/abs/2006.12075). ToG, 2020. [[Page]](http://rubbly.cn/publications/motioNet) [[Code]](https://github.com/Shimingyi/MotioNet)

[VNect: real-time 3D human pose estimation with a single RGB camera](http://gvv.mpi-inf.mpg.de/projects/VNect/content/VNect_SIGGRAPH2017.pdf). SIGGRAPH Asia, 2017. [[Page]](http://gvv.mpi-inf.mpg.de/projects/VNect) [[Code]](http://gvv.mpi-inf.mpg.de/projects/VNect)

[XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera](https://arxiv.org/abs/1907.00837). SIGGRAPH, 2020. [[Page]](https://sites.google.com/view/http%3A%2F%2Fgvv.mpi-inf.mpg.de%2Fprojects%2FXNect%2F) [[Code]](https://sites.google.com/view/https%3A%2F%2Fgithub.com%2Fmehtadushy%2FSelecSLS-Pytorch%2F)

[PhysCap: Physically Plausible Monocular 3D Motion Capture in Real Time](https://arxiv.org/abs/2008.08880). SIGGRAPH Asia, 2020. [[Page]](http://gvv.mpi-inf.mpg.de/projects/PhysCap) 

[Unsupervised 3D Human Pose Representation with Viewpoint and Pose Disentanglement](https://arxiv.org/abs/2007.07053). ECCV, 2020.  [[Code]](https://github.com/NIEQiang001/unsupervised-human-pose)

[Cascaded Deep Monocular 3D Human Pose Estimation with Evolutionary Training Data](https://arxiv.org/abs/2006.07778). CVPR, 2020.  [[Code]](https://github.com/Nicholasli1995/EvoSkeleton)

[End-to-End Estimation of Multi-Person 3D Poses from Multiple Cameras](None). ECCV (Oral), 2020.  

[Learnable Triangulation of Human Pose](https://arxiv.org/abs/1905.05754). ICCV (Oral), 2019.  [[Code]](https://github.com/karfly/learnable-triangulation-pytorch)

[Compressed Volumetric Heatmaps for Multi-Person 3D Pose Estimation](https://arxiv.org/abs/2004.00329). CVPR, 2020.  [[Code]](https://github.com/fabbrimatteo/LoCO)

[Multi-person 3D Pose Estimation in Crowded Scenes Based on Multi-View Geometry](https://arxiv.org/abs/2007.10986). arXiv, 2020.  [[Code]](https://github.com/HeCraneChen/3D-Crowd-Pose-Estimation-Based-on-MVG)


## Naked Body Mesh


[Neural Body Fitting: Unifying Deep Learning and Model Based Human Pose and Shape Estimation](http://virtualhumans.mpi-inf.mpg.de/papers/omran2018NBF/omran2018NBF.pdf). 3DV (Oral), 2018.  [[Code]](https://github.com/mohomran/neural_body_fitting)

[Appearance Consensus Driven Self-Supervised Human Mesh Recovery](https://arxiv.org/pdf/2008.01341.pdf). ECCV (Oral), 2020. [[Page]](https://sites.google.com/view/ss-human-mesh) [[Code]](https://github.com/rakeshramesha/SS_Human_Mesh)

[Delving Deep Into Hybrid Annotations for 3D Human Recovery in the Wild](https://arxiv.org/abs/1908.06442). ICCV, 2019. [[Page]](https://penincillin.github.io/dct_iccv2019) [[Code]](https://github.com/penincillin/DCT_ICCV-2019)

[Learning 3D Human Shape and Pose from Dense Body Parts](https://hongwenzhang.github.io/dense2mesh/pdf/learning3Dhuman.pdf). ArXiv, 2019. [[Page]](https://hongwenzhang.github.io/dense2mesh/) [[Code]](https://hongwenzhang.github.io/dense2mesh)

[Full-Body Awareness from Partial Observations](https://arxiv.org/abs/2008.06046). ECCV, 2020. [[Page]](https://crockwell.github.io/partial_humans) [[Code]](https://github.com/crockwell/partial_humans)

[Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop](https://arxiv.org/pdf/1909.12828.pdf). ICCV, 2019. [[Page]](https://www.seas.upenn.edu/~nkolot/projects/spin) [[Code]](https://github.com/nkolot/SPIN)

[3D Human Mesh Regression with Dense Correspondence](https://arxiv.org/pdf/2006.05734.pdf). CVPR, 2020.  [[Code]](https://github.com/zengwang430521/DecoMR)

[I2L-MeshNet: Image-to-Lixel Prediction Network for Accurate 3D Human Pose and Mesh Estimation from a Single RGB Image](https://arxiv.org/abs/2008.03713). ECCV, 2020.  [[Code]](https://github.com/mks0601/I2L-MeshNet_RELEASE)

[CenterHMR: a Bottom-up Single-shot Method for Multi-person 3D Mesh Recovery from a Single Image](https://arxiv.org/pdf/2008.12272.pdf). ArXiv, 2020.  [[Code]](https://github.com/Arthur151/CenterHMR)

[VIBE: Video Inference for Human Body Pose and Shape Estimation](https://arxiv.org/abs/1912.05656). CVPR, 2020.  [[Code]](https://github.com/mkocabas/VIBE)

[Exemplar Fine-Tuning for 3D Human Pose Fitting Towards In-the-Wild 3D Human Pose Estimation](https://arxiv.org/pdf/2004.03686). ArXiv, 2020.  [[Code]](https://github.com/facebookresearch/eft)

[Total Capture: A 3D Deformation Model for Tracking Faces, Hands, and Bodies](http://openaccess.thecvf.com/content_cvpr_2018/papers/Joo_Total_Capture_A_CVPR_2018_paper.pdf). CVPR(Oral), 2018. [[Page]](https://jhugestar.github.io/totalcapture) 

[Monocular Total Capture: Posing Face, Body and Hands in the Wild](https://arxiv.org/abs/1812.01598). CVPR(Oral), 2019. [[Page]](http://domedb.perception.cs.cmu.edu/mtc.html) [[Code]](https://github.com/CMU-Perceptual-Computing-Lab/MonocularTotalCapture)

[FrankMocap: A Fast Monocular 3D Hand and Body Motion Capture by Regression and Integration](https://arxiv.org/pdf/2008.08324.pdf). ArXiv, 2020. [[Page]](https://penincillin.github.io/frank_mocap) 

[Monocular Expressive Body Regression through Body-Driven Attention](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/620/0983.pdf). ECCV, 2020. [[Page]](https://expose.is.tue.mpg.de) [[Code]](https://github.com/vchoutas/expose)


## Clothed Body Mesh


[LiveCap: Real-time Human Performance Capture from Monocular Video](https://gvv.mpi-inf.mpg.de/projects/LiveCapV2/data/livecap.pdf). SIGGRAPH, 2019. [[Page]](https://gvv.mpi-inf.mpg.de/projects/LiveCapV2/) 

[DeepCap: Monocular Human Performance Capture Using Weak Supervision](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/data/paper.pdf). CVPR (Oral), 2020. [[Page]](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap) 

[DoubleFusion: Real-time Capture of Human Performance with Inner Body Shape from a Depth Sensor](https://arxiv.org/abs/1804.06023). CVPR (Oral), 2018. [[Page]](http://www.liuyebin.com/doublefusion/doublefusion.htm) [[Code]](http://www.liuyebin.com/doublefusion/doublefusion_software.htm)

[SimulCap : Single-View Human Performance Capture with Cloth Simulation](https://arxiv.org/abs/1903.06323). CVPR, 2019. [[Page]](http://www.liuyebin.com/simulcap/simulcap.html) 

[Robust 3D Self-portraits in Seconds](https://arxiv.org/abs/2004.02460). CVPR (Oral), 2020. [[Page]](http://www.liuyebin.com/portrait/portrait.html) 

[MulayCap: Multi-layer Human Performance Capture Using A Monocular Video Camera](https://arxiv.org/abs/2004.05815). TVCG, 2020. [[Page]](http://www.liuyebin.com/MulayCap/MulayCap.html) 

[RobustFusion: Human Volumetric Capture with Data-driven Visual Cues using a RGBD Camera](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490239.pdf). ECCV, 2020.  

[SIZER: A Dataset and Model for Parsing 3D Clothing and Learning Size Sensitive 3D Clothing](https://arxiv.org/abs/2007.11610). ECCV (Oral), 2020. [[Page]](http://virtualhumans.mpi-inf.mpg.de/sizer) [[Code]](https://github.com/garvita-tiwari/sizer)

[PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization](https://arxiv.org/pdf/1905.05172.pdf). ICCV, 2019. [[Page]](https://shunsukesaito.github.io/PIFu) [[Code]](https://github.com/shunsukesaito/PIFu)

[PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization](https://arxiv.org/pdf/2004.00452.pdf). CVPR (Oral), 2020. [[Page]](https://shunsukesaito.github.io/PIFuHD) [[Code]](https://github.com/facebookresearch/pifuhd)

[SiCloPe: Silhouette-Based Clothed People](https://arxiv.org/pdf/1901.00049). CVPR, 2019.  

[ARCH: Animatable Reconstruction of Clothed Humans](https://arxiv.org/pdf/2004.04572.pdf). CVPR, 2020.  

[Monocular Real-Time Volumetric Performance Capture](https://arxiv.org/abs/2007.13988). ECCV, 2020. [[Page]](http://xiuyuliang.cn/monoport) [[Code]](https://github.com/Project-Splinter/MonoPort)

[Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion](https://arxiv.org/abs/2003.01456). CVPR, 2020. [[Page]](http://virtualhumans.mpi-inf.mpg.de/ifnets) [[Code]](https://github.com/jchibane/if-net)

[Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction](https://arxiv.org/abs/2007.11432). ECCV (Oral), 2020. [[Page]](https://virtualhumans.mpi-inf.mpg.de/ipnet) [[Code]](https://github.com/bharat-b7/IPNet)

[PaMIR: Parametric Model-Conditioned Implicit Representation for Image-based Human Reconstruction](https://arxiv.org/abs/2007.03858). TPAMI, 2020. [[Page]](http://www.liuyebin.com/pamir/pamir.html) 

[NormalGAN: Learning Detailed 3D Human from a Single RGB-D Image](https://arxiv.org/abs/2007.15340). ECCV, 2020. [[Page]](http://www.liuyebin.com/NormalGan/normalgan.html) 

[MonoClothCap: Towards Temporally Coherent Clothing Capture from Monocular RGB Video](http://arxiv.org/abs/2009.10711). ArXiv, 2020.  


## Human Motion


[Long-term Human Motion Prediction with Scene Context](https://arxiv.org/pdf/2007.03672.pdf). ECCV (Oral), 2020. [[Page]](https://people.eecs.berkeley.edu/~zhecao/hmp/index.html) [[Code]](https://github.com/ZheC/GTA-IM-Dataset)


## Human-Object Interaction


[Perceiving 3D Human-Object Spatial Arrangements from a Single Image in the Wild](https://arxiv.org/abs/2007.15649). ECCV, 2020. [[Page]](https://jasonyzhang.com/phosa) [[Code]](https://github.com/facebookresearch/phosa)

[Resolving 3D Human Pose Ambiguities with 3D Scene Constraints](https://arxiv.org/abs/1908.06963). ICCV, 2019. [[Page]](https://prox.is.tue.mpg.de) [[Code]](https://github.com/MohameHassan/PROX)


## Animation


[RigNet: Neural Rigging for Articulated Characters](https://people.cs.umass.edu/~zhanxu/papers/RigNet.pdf). SIGGRAPH, 2020. [[Page]](https://zhan-xu.github.io/rig-net) [[Code]](https://github.com/zhan-xu/RigNet)

[Skeleton-Aware Networks for Deep Motion Retargeting](https://deepmotionediting.github.io/papers/skeleton-aware-camera-ready.pdf). SIGGRAPH, 2020. [[Page]](https://deepmotionediting.github.io/retargeting) [[Code]](https://github.com/DeepMotionEditing/deep-motion-editing)

[Motion Retargetting based on Dilated Convolutions and Skeleton-specific Loss Functions](https://diglib.eg.org/bitstream/handle/10.1111/cgf13947/v39i2pp497-507.pdf). Eurographics, 2020. [[Page]](https://sites.google.com/view/retargetting-tdcn) [[Code]](https://sites.google.com/view/https%3A%2F%2Fgithub.com%2Fmedialab-ku%2Fretargetting-tdcn)

[DeePSD: Automatic Deep Skinning And Pose Space Deformation For 3D Garment Animation](https://arxiv.org/pdf/2009.02715). ArXiv, 2020.  


## Cloth/Try-On


[Deep Fashion3D: A Dataset and Benchmark for 3D Garment Reconstruction from Single-view Images](https://arxiv.org/abs/2003.12753). ECCV (Oral), 2020. [[Page]](https://kv2000.github.io/2020/03/25/deepFashion3DRevisited) 

[TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style](https://arxiv.org/abs/2003.04583). CVPR (Oral), 2020. [[Page]](http://virtualhumans.mpi-inf.mpg.de/tailornet) [[Code]](https://github.com/chaitanya100100/TailorNet)

[Learning-Based Animation of Clothing for Virtual Try-On](http://dancasas.github.io/docs/santesteban_Eurographics2019.pdf). Eurographics, 2019. [[Page]](http://dancasas.github.io/projects/LearningBasedVirtualTryOn/index.html) 


## Dataset


[AMASS: Archive of Motion Capture as Surface Shapes](https://arxiv.org/abs/1904.03278). ICCV, 2019. [[Page]](https://amass.is.tue.mpg.de) [[Code]](https://github.com/nghorbani/amass)

[3DBodyTex: Textured 3D Body Dataset](https://orbilu.uni.lu/bitstream/10993/36414/1/saint_et_al-3dbodytex-3dv_2018.pdf). 3DV, 2018. [[Page]](https://cvi2.uni.lu/datasets) 

[Motion Capture from Internet Videos](https://arxiv.org/pdf/2008.07931.pdf). ECCV, 2020. [[Page]](https://zju3dv.github.io/iMoCap) [[Code]](https://github.com/zju3dv/iMoCap)

[Full-Body Awareness from Partial Observations](https://arxiv.org/abs/2008.06046). ECCV, 2020. [[Page]](https://crockwell.github.io/partial_humans) [[Code]](https://github.com/crockwell/partial_humans)

-----

## [Back to Top](#table-of-contents)
