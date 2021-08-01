# Awesome 3D Body Papers

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> An awesome & curated list of papers about 3D human body.



:point_right: **Note**: see paper list sorted by [**year**](README_year.md) or [**publication**](README_pub.md).


-----

## Table of Contents

- [Body Model](#body-model)
- [Body Pose](#body-pose)
- [Naked Body Mesh](#naked-body-mesh)
- [Clothed Body Mesh](#clothed-body-mesh)
- [Human Depth Estimation](#human-depth-estimation)
- [Human Motion](#human-motion)
- [Human-Object Interaction](#human-object-interaction)
- [Animation](#animation)
- [Cloth/Try-On](#cloth/try-on)
- [Neural Rendering](#neural-rendering)
- [Dataset](#dataset)

-----


## Body Model


[SCAPE: Shape Completion and Animation of People](http://robots.stanford.edu/papers/anguelov.shapecomp.pdf). SIGGRAPH, 2005. [[Page]](http://robotics.stanford.edu/~drago/Projects/scape/scape.html) 

[SMPL: A Skinned Multi-Person Linear Model](http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf). SIGGRAPH Asia, 2015. [[Page]](https://smpl.is.tue.mpg.de) [[Code]](https://github.com/vchoutas/smplx)

[Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf). CVPR, 2019. [[Page]](https://smpl-x.is.tue.mpg.de) [[Code]](https://github.com/vchoutas/smplify-x)

[SoftSMPL: Data-driven Modeling of Nonlinear Soft-tissue Dynamics for Parametric Humans](https://arxiv.org/pdf/2004.00326). Eurographics, 2020. [[Page]](http://dancasas.github.io/projects/SoftSMPL) 

[Modeling and Estimation of Nonlinear Skin Mechanics for Animated Avatars](http://dancasas.github.io/docs/romero_Eurographics2020.pdf). Eurographics, 2020. [[Page]](https://dancasas.github.io/projects/SkinMechanics) 

[STAR: Sparse Trained Articulated Human Body Regressor](https://arxiv.org/pdf/2008.08535). ECCV, 2020. [[Page]](http://star.is.tue.mpg.de) [[Code]](https://github.com/ahmedosman/STAR)

[BLSM: A Bone-Level Skinned Model of the Human Mesh](https://www.arielai.com/blsm/data/paper.pdf). ECCV, 2020. [[Page]](https://www.arielai.com/blsm) 

[Joint Optimization for Multi-Person Shape Models from Markerless 3D-Scans](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630035.pdf). ECCV, 2020.  [[Code]](https://github.com/Intelligent-Systems-Research-Group/JOMS)

[GHUM & GHUML: Generative 3D Human Shape and Articulated Pose Models](https://arxiv.org/pdf/2008.08535). CVPR (Oral), 2020.  [[Code]](https://github.com/google-research/google-research/tree/master/ghum)

[BASH: Biomechanical Animated Skinned Human for Visualization of Kinematics and Muscle Activity](https://www.scitepress.org/Papers/2021/102106/102106.pdf). GRAPP, 2021.  [[Code]](https://github.com/mad-lab-fau/BASH-Model)

[SMPLicit: Topology-aware Generative Model for Clothed People](https://arxiv.org/abs/2103.06871). CVPR, 2021. [[Page]](http://www.iri.upc.edu/people/ecorona/smplicit) [[Code]](https://github.com/enriccorona/SMPLicit)

[NPMs: Neural Parametric Models for 3D Deformable Shapes](https://arxiv.org/abs/2104.00702). ArXiv, 2021. [[Page]](https://www.youtube.com/watch?v=muZXXgkkMPY) 

[LEAP: Learning Articulated Occupancy of People](https://arxiv.org/abs/2104.06849). CVPR, 2021. [[Page]](https://neuralbodies.github.io/LEAP) [[Code]](https://github.com/neuralbodies/leap)

[SCALE: Modeling Clothed Humans with a Surface Codec of Articulated Local Elements](https://arxiv.org/abs/2104.07660). CVPR, 2021. [[Page]](https://qianlim.github.io/SCALE) 


## Body Pose


[MotioNet: 3D Human Motion Reconstruction from Monocular Video with Skeleton Consistency](https://arxiv.org/abs/2006.12075). ToG, 2020. [[Page]](http://rubbly.cn/publications/motioNet) [[Code]](https://github.com/Shimingyi/MotioNet)

[VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera](http://gvv.mpi-inf.mpg.de/projects/VNect/content/VNect_SIGGRAPH2017.pdf). SIGGRAPH Asia, 2017. [[Page]](http://gvv.mpi-inf.mpg.de/projects/VNect) [[Code]](http://gvv.mpi-inf.mpg.de/projects/VNect)

[XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera](https://arxiv.org/abs/1907.00837). SIGGRAPH, 2020. [[Page]](https://sites.google.com/view/http%3A%2F%2Fgvv.mpi-inf.mpg.de%2Fprojects%2FXNect%2F) [[Code]](https://sites.google.com/view/https%3A%2F%2Fgithub.com%2Fmehtadushy%2FSelecSLS-Pytorch%2F)

[PhysCap: Physically Plausible Monocular 3D Motion Capture in Real Time](https://arxiv.org/abs/2008.08880). SIGGRAPH Asia, 2020. [[Page]](http://gvv.mpi-inf.mpg.de/projects/PhysCap) 

[Neural Monocular 3D Human Motion Capture with Physical Awareness](https://arxiv.org/abs/2105.01057). SIGGRAPH, 2021. [[Page]](http://gvv.mpi-inf.mpg.de/projects/PhysAware) 

[PoseAug: A Differentiable Pose Augmentation Framework for 3D Human Pose Estimation](https://arxiv.org/abs/2105.02465). CVPR (Oral), 2021. [[Page]](https://jeff95.me) [[Code]](https://github.com/jfzhang95/PoseAug)

[Cascaded Deep Monocular 3D Human Pose Estimation with Evolutionary Training Data](https://arxiv.org/abs/2006.07778). CVPR, 2020.  [[Code]](https://github.com/Nicholasli1995/EvoSkeleton)

[PoseLifter: Absolute 3D Human Pose Lifting Network from a Single Noisy 2D Human Pose](https://arxiv.org/abs/1910.12029). ArXiv, 2020.  [[Code]](hhttps://github.com/juyongchang/PoseLifter)

[Probabilistic Monocular 3D Human Pose Estimation with Normalizing Flows](https://arxiv.org/abs/2107.13788). ICCV, 2021.  [[Code]](https://github.com/twehrbein/Probabilistic-Monocular-3D-Human-Pose-Estimation-with-Normalizing-Flows)

[Learnable Triangulation of Human Pose](https://arxiv.org/abs/1905.05754). ICCV (Oral), 2019.  [[Code]](https://github.com/karfly/learnable-triangulation-pytorch)

[FLEX: Parameter-free Multi-view 3D Human Motion Reconstruction](https://arxiv.org/abs/2105.01937). ArXiv, 2021. [[Page]](https://briang13.github.io/FLEX) 

[Weakly-supervised Cross-view 3D Human Pose Estimation](https://arxiv.org/abs/2105.10882). ArXiv, 2021.  

[Compressed Volumetric Heatmaps for Multi-Person 3D Pose Estimation](https://arxiv.org/abs/2004.00329). CVPR, 2020.  [[Code]](https://github.com/fabbrimatteo/LoCO)

[PandaNet: Anchor-Based Single-Shot Multi-Person 3D Pose Estimation](https://arxiv.org/abs/2101.02471). ArXiv, 2021.  

[SMAP: Single-Shot Multi-Person Absolute 3D Pose Estimation](https://arxiv.org/abs/2008.11469). ECCV, 2020. [[Page]](https://zju3dv.github.io/SMAP) [[Code]](https://github.com/zju3dv/SMAP)

[PI-Net: Pose Interacting Network for Multi-Person Monocular 3D Pose Estimation](https://arxiv.org/pdf/2010.05302). WACV, 2021.  

[Monocular 3D Multi-Person Pose Estimation by Integrating Top-Down and Bottom-Up Networks](https://arxiv.org/abs/2104.01797). CVPR, 2021.  [[Code]](https://github.com/3dpose/3D-Multi-Person-Pose)

[FCPose: Fully Convolutional Multi-Person Pose Estimation with Dynamic Instance-Aware Convolutions](https://arxiv.org/abs/2105.14185). CVPR, 2021.  [[Code]](https://git.io/AdelaiDet)

[End-to-End Estimation of Multi-Person 3D Poses from Multiple Cameras](None). ECCV (Oral), 2020.  

[Multi-person 3D Pose Estimation in Crowded Scenes Based on Multi-View Geometry](https://arxiv.org/abs/2007.10986). ArXiv, 2020.  [[Code]](https://github.com/HeCraneChen/3D-Crowd-Pose-Estimation-Based-on-MVG)

[Multi-View Multi-Person 3D Pose Estimation with Plane Sweep Stereo](https://arxiv.org/abs/2104.02273). CVPR, 2021.  [[Code]](https://github.com/jiahaoLjh/PlaneSweepPose)

[Temporal Smoothing for 3D Human Pose Estimation and Localization for Occluded People](https://arxiv.org/abs/2011.00250). ArXiv, 2020.  [[Code]](https://github.com/vegesm/pose_refinement)

[Attention Mechanism Exploits Temporal Contexts: Real-time 3D Human Pose Reconstruction](http://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Attention_Mechanism_Exploits_Temporal_Contexts_Real-Time_3D_Human_Pose_Reconstruction_CVPR_2020_paper.html). CVPR (Oral), 2020.  [[Code]](https://github.com/vegesm/pose_refinement)

[3D Human Pose Estimation with Spatial and Temporal Transformers](https://arxiv.org/abs/2103.10455). ArXiv, 2021.  [[Code]](https://github.com/zczcwh/PoseFormer)

[Skeletor: Skeletal Transformers for Robust Body-Pose Estimation](https://arxiv.org/abs/2104.11712). ArXiv, 2021.  

[A Graph Attention Spatio-temporal Convolutional Networks for 3D Human Pose Estimation in Video](https://arxiv.org/abs/2003.14179). ArXiv, 2020. [[Page]](http://www.juanrojas.net/gast) [[Code]](https://github.com/fabro66/GAST-Net-3DPoseEstimation)

[TriPose: A Weakly-Supervised 3D Human Pose Estimation via Triangulation from Video](https://arxiv.org/abs/2105.06599). ArXiv, 2021.  

[MeTRAbs: Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose Estimation](https://arxiv.org/abs/2007.07227). T-BIOM, 2020. [[Page]](https://sites.google.com/a/udayton.edu/jshen1/cvpr2020) [[Code]](https://github.com/lrxjason/Attention3DHumanPose)

[PCLs: Geometry-aware Neural Reconstruction of 3D Pose with Perspective Crop Layers](https://arxiv.org/abs/2011.13607). CVPR, 2021.  

[Real-time Lower-body Pose Prediction from Sparse Upper-body Tracking Signals](https://arxiv.org/abs/2103.01500). ArXiv, 2021.  

[Context Modeling in 3D Human Pose Estimation: A Unified Perspective](https://arxiv.org/abs/2103.15507). CVPR, 2021.  

[CanonPose: Self-Supervised Monocular 3D Human Pose Estimation in the Wild](https://arxiv.org/abs/2011.14679). CVPR, 2021.  

[Invariant Teacher and Equivariant Student for Unsupervised 3D Human Pose Estimation](https://arxiv.org/abs/2012.09398). AAAI, 2021.  [[Code]](https://github.com/sjtuxcx/ITES)

[Unsupervised 3D Human Pose Representation with Viewpoint and Pose Disentanglement](https://arxiv.org/abs/2007.07053). ECCV, 2020.  [[Code]](https://github.com/NIEQiang001/unsupervised-human-pose)

[DOPE: Distillation Of Part Experts for whole-body 3D pose estimation in the wild](https://arxiv.org/abs/2008.09457). ECCV, 2020.  [[Code]](https://github.com/naver/dope)

[MocapNET: Ensemble of SNN Encoders for 3D Human Pose Estimation in RGB Images](http://users.ics.forth.gr/~argyros/mypapers/2019_09_BMVC_mocapnet.pdf). BMVC, 2019.  [[Code]](https://github.com/FORTH-ModelBasedTracker/MocapNET)

[Residual Pose: A Decoupled Approach for Depth-based 3D Human Pose Estimation](https://arxiv.org/pdf/2011.05010.pdf). IROS, 2020.  [[Code]](https://github.com/idiap/residual_pose)

[PoP-Net: Pose over Parts Network for Multi-Person 3D Pose Estimation from a Depth Image](https://arxiv.org/abs/2012.06734). ArXiv, 2020.  [[Code]](https://github.com/idiap/residual_pose)


## Naked Body Mesh


[Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image](http://files.is.tue.mpg.de/black/papers/BogoECCV2016.pdf). ECCV, 2016. [[Page]](http://smplify.is.tue.mpg.de) [[Code]](https://github.com/vchoutas/smplify-x)

[Learning to Estimate 3D Human Pose and Shape from a Single Color Image](https://arxiv.org/pdf/1805.04092.pdf). CVPR, 2018. [[Page]](https://www.seas.upenn.edu/~pavlakos/projects/humanshape) 

[Neural Body Fitting: Unifying Deep Learning and Model Based Human Pose and Shape Estimation](http://virtualhumans.mpi-inf.mpg.de/papers/omran2018NBF/omran2018NBF.pdf). 3DV (Oral), 2018.  [[Code]](https://github.com/mohomran/neural_body_fitting)

[Appearance Consensus Driven Self-Supervised Human Mesh Recovery](https://arxiv.org/pdf/2008.01341.pdf). ECCV (Oral), 2020. [[Page]](https://sites.google.com/view/ss-human-mesh) [[Code]](https://github.com/rakeshramesha/SS_Human_Mesh)

[Delving Deep Into Hybrid Annotations for 3D Human Recovery in the Wild](https://arxiv.org/abs/1908.06442). ICCV, 2019. [[Page]](https://penincillin.github.io/dct_iccv2019) [[Code]](https://github.com/penincillin/DCT_ICCV-2019)

[Learning 3D Human Shape and Pose from Dense Body Parts](https://hongwenzhang.github.io/dense2mesh/pdf/learning3Dhuman.pdf). ArXiv, 2019. [[Page]](https://hongwenzhang.github.io/dense2mesh/) [[Code]](https://hongwenzhang.github.io/dense2mesh)

[Heuristic Weakly Supervised 3D Human Pose Estimation in Novel Contexts without Any 3D Pose Ground Truth](https://arxiv.org/abs/2105.10996). ArXiv, 2021.  

[Revitalizing Optimization for 3D Human Pose and Shape Estimation: A Sparse Constrained Formulation](https://arxiv.org/abs/2105.13965). ArXiv, 2021.  

[Full-Body Awareness from Partial Observations](https://arxiv.org/abs/2008.06046). ECCV, 2020. [[Page]](https://crockwell.github.io/partial_humans) [[Code]](https://github.com/crockwell/partial_humans)

[Object-Occluded Human Shape and Pose Estimation from a Single Color Image](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Object-Occluded_Human_Shape_and_Pose_Estimation_From_a_Single_Color_CVPR_2020_paper.pdf). CVPR, 2020. [[Page]](https://www.yangangwang.com/papers/ZHANG-OOH-2020-03.html) [[Code]](https://gitee.com/seuvcl/CVPR2020-OOH)

[PARE: Part Attention Regressor for 3D Human Body Estimation](https://arxiv.org/abs/2104.08527). ArXiv, 2021. [[Page]](https://pare.is.tue.mpg.de) 

[3D Multi-bodies: Fitting Sets of Plausible 3D Human Models to Ambiguous Image Data](https://arxiv.org/abs/2011.00980). NeurIPS, 2020.  

[Parametric Shape Estimation of Human Body under Wide Clothing](https://ieeexplore.ieee.org/document/9219144). ACM MM, 2020.  [[Code]](https://github.com/YCL92/SHADER)

[Everybody Is Unique: Towards Unbiased Human Mesh Recovery](https://arxiv.org/abs/2107.06239). ArXiv, 2021.  

[3D Human Pose, Shape and Texture from Low-Resolution Images and Videos](https://arxiv.org/abs/2103.06498). ArXiv, 2021.  

[On Self-Contact and Human Pose](https://arxiv.org/abs/2104.03176). CVPR, 2021. [[Page]](https://tuch.is.tue.mpg.de) 

[Probabilistic 3D Human Shape and Pose Estimation from Multiple Unconstrained Images in the Wild](https://arxiv.org/abs/2103.10978). CVPR, 2021.  

[Human Body Model Fitting by Learned Gradient Descent](https://arxiv.org/abs/2008.08474). ECCV, 2020. [[Page]](https://ait.ethz.ch/projects/2020/learned-body-fitting) 

[End-to-end Recovery of Human Shape and Pose](https://arxiv.org/pdf/1712.06584.pdf). CVPR, 2018. [[Page]](https://akanazawa.github.io/hmr) [[Code]](https://github.com/akanazawa/hmr)

[Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop](https://arxiv.org/pdf/1909.12828.pdf). ICCV, 2019. [[Page]](https://www.seas.upenn.edu/~nkolot/projects/spin) [[Code]](https://github.com/nkolot/SPIN)

[3D Human Mesh Regression with Dense Correspondence](https://arxiv.org/pdf/2006.05734.pdf). CVPR, 2020.  [[Code]](https://github.com/zengwang430521/DecoMR)

[Hierarchical Kinematic Human Mesh Recovery](https://arxiv.org/abs/2003.04232). ECCV, 2020. [[Page]](https://cs.gmu.edu/~ggeorgak) 

[I2L-MeshNet: Image-to-Lixel Prediction Network for Accurate 3D Human Pose and Mesh Estimation from a Single RGB Image](https://arxiv.org/abs/2008.03713). ECCV, 2020.  [[Code]](https://github.com/mks0601/I2L-MeshNet_RELEASE)

[MeshLifter: Weakly Supervised Approach for 3D Human Mesh Reconstruction from a Single 2D Pose Based on Loop Structure](https://www.researchgate.net/publication/343339747_MeshLifter_Weakly_Supervised_Approach_for_3D_Human_Mesh_Reconstruction_from_a_Single_2D_Pose_Based_on_Loop_Structure). Sensors, 2020.  [[Code]](https://github.com/sunwonlikeyou/MeshLifter)

[Pose2Mesh: Graph Convolutional Network for 3D Human Pose and Mesh Recovery from a 2D Human Pose](https://arxiv.org/abs/2008.09047). ECCV, 2020.  [[Code]](https://github.com/hongsukchoi/Pose2Mesh_RELEASE)

[PoseNet3D: Learning Temporally Consistent 3D Human Pose via Knowledge Distillation](https://arxiv.org/abs/2003.03473). 3DV, 2020.  

[Human Mesh Recovery from Monocular Images via a Skeleton-disentangled Representation](https://arxiv.org/abs/1908.07172). ICCV, 2019.  [[Code]](https://github.com/JDAI-CV/DSD-SATN)

[Learning 3D Human Shape and Pose from Dense Body Parts](https://arxiv.org/pdf/1912.13344.pdf). TPAMI, 2020. [[Page]](https://hongwenzhang.github.io/dense2mesh) [[Code]](https://github.com/HongwenZhang/DaNet-3DHumanReconstruction)

[Exemplar Fine-Tuning for 3D Human Pose Fitting Towards In-the-Wild 3D Human Pose Estimation](https://arxiv.org/pdf/2004.03686). ArXiv, 2020.  [[Code]](https://github.com/facebookresearch/eft)

[HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation](https://arxiv.org/abs/2011.14672). CVPR, 2021. [[Page]](https://jeffli.site/HybrIK) [[Code]](https://github.com/Jeff-sjtu/HybrIK)

[Chasing the Tail in Monocular 3D Human Reconstruction with Prototype Memory](https://arxiv.org/abs/2012.14739). ArXiv, 2020.  

[Beyond Weak Perspective for Monocular 3D Human Pose Estimation](https://arxiv.org/abs/2009.06549). ArXiv, 2020.  

[PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop](https://arxiv.org/abs/2103.16507). ICCV (Oral), 2021. [[Page]](https://hongwenzhang.github.io/pymaf) [[Code]](https://github.com/HongwenZhang/PyMAF)

[KAMA: 3D Keypoint Aware Body Mesh Articulation](https://arxiv.org/abs/2104.13502). ArXiv, 2021.  

[SimPoE: Simulated Character Control for 3D Human Pose Estimation](https://arxiv.org/abs/2104.00683). CVPR (Oral), 2021. [[Page]](https://www.ye-yuan.com/simpoe) 

[Reconstructing 3D Human Pose by Watching Humans in the Mirror](https://arxiv.org/abs/2104.00340). CVPR (Oral), 2021. [[Page]](https://zju3dv.github.io/Mirrored-Human) [[Code]](https://github.com/zju3dv/Mirrored-Human)

[CenterHMR: a Bottom-up Single-shot Method for Multi-person 3D Mesh Recovery from a Single Image](https://arxiv.org/pdf/2008.12272.pdf). ArXiv, 2020.  [[Code]](https://github.com/Arthur151/CenterHMR)

[Full-body motion capture for multiple closely interacting persons](http://cic.tju.edu.cn/faculty/likun/GM.pdf). CVM, 2020.  

[Coherent Reconstruction of Multiple Humans from a Single Image](https://arxiv.org/pdf/2006.08586.pdf). CVPR, 2020. [[Page]](https://jiangwenpl.github.io/multiperson) [[Code]](https://github.com/JiangWenPL/multiperson)

[Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image](https://arxiv.org/abs/1907.11346). ICCV, 2019.  [[Code]](https://github.com/mks0601/3DMPPE_POSENET_RELEASE)

[Monocular, One-stage, Regression of Multiple 3D People](https://arxiv.org/abs/2008.12272). ArXiv, 2020.  [[Code]](https://github.com/Arthur151/ROMP)

[Body Meshes as Points](https://arxiv.org/abs/2105.02467). CVPR, 2021. [[Page]](https://jeff95.me) [[Code]](https://github.com/jfzhang95/BMP)

[Learning 3D Human Dynamics from Video](https://arxiv.org/abs/1812.01601). CVPR, 2019. [[Page]](https://akanazawa.github.io/human_dynamics) [[Code]](https://github.com/akanazawa/human_dynamics)

[VIBE: Video Inference for Human Body Pose and Shape Estimation](https://arxiv.org/abs/1912.05656). CVPR, 2020.  [[Code]](https://github.com/mkocabas/VIBE)

[3D Human Motion Estimation via Motion Compression and Refinement](https://arxiv.org/abs/2008.03789). ACCV (Oral), 2020. [[Page]](https://zhengyiluo.github.io/projects/meva) [[Code]](https://github.com/ZhengyiLuo/MEVA)

[Beyond Static Features for Temporally Consistent 3D Human Pose and Shape from a Video](https://arxiv.org/abs/2011.08627). CVPR, 2021. [[Page]](https://youtu.be/WB3nTnSQDII) [[Code]](https://github.com/hongsukchoi/TCMR_RELEASE)

[End-to-End Human Pose and Mesh Reconstruction with Transformers](https://arxiv.org/abs/2012.09760). CVPR, 2021.  [[Code]](https://github.com/microsoft/MeshTransformer)

[THUNDR: Transformer-based 3D HUmaN Reconstruction with Markers](https://arxiv.org/abs/2106.09336). ArXiv, 2021.  

[Human Mesh Recovery from Multiple Shots](https://arxiv.org/abs/2012.09843). ArXiv, 2020. [[Page]](https://geopavlakos.github.io/multishot/) 

[PC-HMR: Pose Calibration for 3D Human Mesh Recovery from 2D Images/Videos](https://arxiv.org/abs/2103.09009). AAAI, 2021.  

[Self-Attentive 3D Human Pose and Shape Estimation from Videos](https://arxiv.org/abs/2103.14182). ArXiv, 2021.  

[HuMoR: 3D Human Motion Model for Robust Pose Estimation](https://arxiv.org/abs/2105.04668). ICCV, 2021. [[Page]](https://geometry.stanford.edu/projects/humor) 

[Bilevel Online Adaptation for Out-of-Domain Human Mesh Reconstruction](https://arxiv.org/abs/2103.16449). CVPR, 2021. [[Page]](https://sites.google.com/view/humanmeshboa) [[Code]](https://github.com/syguan96/BOA)

[Learning Local Recurrent Models for Human Mesh Recovery](https://arxiv.org/abs/2107.12847). ArXiv, 2021.  

[Total Capture: A 3D Deformation Model for Tracking Faces, Hands, and Bodies](http://openaccess.thecvf.com/content_cvpr_2018/papers/Joo_Total_Capture_A_CVPR_2018_paper.pdf). CVPR (Oral), 2018. [[Page]](https://jhugestar.github.io/totalcapture) 

[Monocular Total Capture: Posing Face, Body and Hands in the Wild](https://arxiv.org/abs/1812.01598). CVPR (Oral), 2019. [[Page]](http://domedb.perception.cs.cmu.edu/mtc.html) [[Code]](https://github.com/CMU-Perceptual-Computing-Lab/MonocularTotalCapture)

[Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf). CVPR, 2019. [[Page]](https://smpl-x.is.tue.mpg.de) [[Code]](https://github.com/vchoutas/smplify-x)

[FrankMocap: A Fast Monocular 3D Hand and Body Motion Capture by Regression and Integration](https://arxiv.org/pdf/2008.08324.pdf). ArXiv, 2020. [[Page]](https://penincillin.github.io/frank_mocap) [[Code]](https://github.com/facebookresearch/frankmocap)

[Monocular Expressive Body Regression through Body-Driven Attention](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/620/0983.pdf). ECCV, 2020. [[Page]](https://expose.is.tue.mpg.de) [[Code]](https://github.com/vchoutas/expose)

[NeuralAnnot: Neural Annotator for in-the-wild Expressive 3D Human Pose and Mesh Training Sets](https://arxiv.org/abs/2011.11232). ArXiv, 2020. [[Page]](https://mks0601.github.io) 

[Pose2Pose: 3D Positional Pose-Guided 3D Rotational Pose Prediction for Expressive 3D Human Pose and Mesh Estimation](https://arxiv.org/abs/2011.11534). ArXiv, 2020. [[Page]](https://mks0601.github.io) 

[Monocular Real-time Full Body Capture with Inter-part Correlations](https://arxiv.org/abs/2012.06087). CVPR, 2021. [[Page]](https://calciferzh.github.io/publications/zhou2021monocular) 

[Collaborative Regression of Expressive Bodies using Moderation](https://arxiv.org/abs/2105.05301). ArXiv, 2021. [[Page]](https://pixie.is.tue.mpg.de) 

[Real-time RGBD-based Extended Body Pose Estimation](https://arxiv.org/abs/2103.03663). WACV, 2021.  [[Code]](https://saic-violet.github.io/rgbd-kinect-pose)

[TransPose: Real-time 3D Human Translation and Pose Estimation with Six Inertial Sensors](https://arxiv.org/abs/2105.04605). SIGGRAPH, 2021. [[Page]](https://xinyu-yi.github.io/TransPose) 


## Clothed Body Mesh


[LiveCap: Real-time Human Performance Capture from Monocular Video](https://gvv.mpi-inf.mpg.de/projects/LiveCapV2/data/livecap.pdf). SIGGRAPH, 2019. [[Page]](https://gvv.mpi-inf.mpg.de/projects/LiveCapV2/) 

[DeepCap: Monocular Human Performance Capture Using Weak Supervision](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/data/paper.pdf). CVPR (Oral), 2020. [[Page]](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap) 

[MonoClothCap: Towards Temporally Coherent Clothing Capture from Monocular RGB Video](http://arxiv.org/abs/2009.10711). 3DV, 2020.  

[MulayCap: Multi-layer Human Performance Capture Using A Monocular Video Camera](https://arxiv.org/abs/2004.05815). TVCG, 2020. [[Page]](http://www.liuyebin.com/MulayCap/MulayCap.html) 

[ChallenCap: Monocular 3D Capture of Challenging Human Performances using Multi-Modal References](https://arxiv.org/abs/2103.06747). CVPR, 2021.  

[Deep Physics-aware Inference of Cloth Deformation for Monocular Human Performance Capture](https://arxiv.org/abs/2011.12866). ArXiv, 2020.  

[DoubleFusion: Real-time Capture of Human Performance with Inner Body Shape from a Depth Sensor](https://arxiv.org/abs/1804.06023). CVPR (Oral), 2018. [[Page]](http://www.liuyebin.com/doublefusion/doublefusion.htm) [[Code]](http://www.liuyebin.com/doublefusion/doublefusion_software.htm)

[SimulCap : Single-View Human Performance Capture with Cloth Simulation](https://arxiv.org/abs/1903.06323). CVPR, 2019. [[Page]](http://www.liuyebin.com/simulcap/simulcap.html) 

[Robust 3D Self-portraits in Seconds](https://arxiv.org/abs/2004.02460). CVPR (Oral), 2020. [[Page]](http://www.liuyebin.com/portrait/portrait.html) 

[RobustFusion: Human Volumetric Capture with Data-driven Visual Cues using a RGBD Camera](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490239.pdf). ECCV, 2020.  

[NormalGAN: Learning Detailed 3D Human from a Single RGB-D Image](https://arxiv.org/abs/2007.15340). ECCV, 2020. [[Page]](http://www.liuyebin.com/NormalGan/normalgan.html) 

[TexMesh: Reconstructing Detailed Human Texture and Geometry from RGB-D Video](https://arxiv.org/abs/2008.00158). ECCV, 2020. [[Page]](https://research.fb.com/publications/texmesh-reconstructing-detailed-human-texture-and-geometry-from-rgb-d-video) 

[Neural Deformation Graphs for Globally-consistent Non-rigid Reconstruction](https://arxiv.org/abs/2012.01451). CVPR (Oral), 2021. [[Page]](https://aljazbozic.github.io/neural_deformation_graphs) 

[Function4D: Real-time Human Volumetric Capture from Very Sparse Consumer RGBD Sensors](http://www.liuyebin.com/Function4D/assets/Function4D.pdf). CVPR (Oral), 2021. [[Page]](http://www.liuyebin.com/Function4D/Function4D.html) 

[POSEFusion:Pose-guided Selective Fusion for Single-view Human Volumetric Capture](https://arxiv.org/abs/2103.15331). CVPR (Oral), 2021. [[Page]](http://www.liuyebin.com/posefusion/posefusion.html) 

[DSFN: Dynamic Surface Function Networks for Clothed Human Bodies](https://arxiv.org/abs/2104.03978). ArXiv, 2021. [[Page]](https://andreiburov.github.io/DSFN) [[Code]](https://github.com/andreiburov/DSFN)

[Fast Generation of Realistic Virtual Humans](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490239.pdf). VRST, 2017. [[Page]](https://ls7-gv.cs.tu-dortmund.de/downloads/publications/2017/vrst17a.mp4) 

[Realistic Virtual Humans from Smartphone Videos](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490239.pdf). VRST, 2020. [[Page]](https://ls7-gv.cs.tu-dortmund.de/downloads/publications/2020/vrst20.mp4) 

[DeepMultiCap: Performance Capture of Multiple Characters Using Sparse Multiview Cameras](https://arxiv.org/abs/2105.00261). ArXiv, 2021. [[Page]](http://liuyebin.com/dmc/dmc.html) 

[Video Based Reconstruction of 3D People Models](https://arxiv.org/abs/1803.04758). CVPR, 2018. [[Page]](https://graphics.tu-bs.de/people-snapshot) 

[Learning to Reconstruct People in Clothing from a Single RGB Camera](http://virtualhumans.mpi-inf.mpg.de/papers/alldieck19cvpr/alldieck19cvpr.pdf). CVPR, 2019. [[Page]](http://virtualhumans.mpi-inf.mpg.de/octopus) [[Code]](https://github.com/thmoa/octopus)

[SiCloPe: Silhouette-Based Clothed People](https://arxiv.org/pdf/1901.00049). CVPR, 2019.  

[Tex2Shape: Detailed Full Human Body Geometry from a Single Image](https://arxiv.org/abs/1904.08645). ICCV, 2019. [[Page]](http://virtualhumans.mpi-inf.mpg.de/tex2shape) [[Code]](https://github.com/thmoa/tex2shape)

[Multi-Garment Net: Learning to Dress 3D People from Images](http://virtualhumans.mpi-inf.mpg.de/papers/bhatnagar2019mgn/bhatnagar2019mgn.pdf). ICCV, 2019. [[Page]](https://virtualhumans.mpi-inf.mpg.de/mgn) 

[3DPeople: Modeling the Geometry of Dressed Humans](https://arxiv.org/abs/1904.04571). ICCV, 2019. [[Page]](https://www.albertpumarola.com/research/3DPeople/index.html) [[Code]](https://github.com/albertpumarola/3DPeople-Dataset)

[SIZER: A Dataset and Model for Parsing 3D Clothing and Learning Size Sensitive 3D Clothing](https://arxiv.org/abs/2007.11610). ECCV (Oral), 2020. [[Page]](http://virtualhumans.mpi-inf.mpg.de/sizer) [[Code]](https://github.com/garvita-tiwari/sizer)

[PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization](https://arxiv.org/pdf/1905.05172.pdf). ICCV, 2019. [[Page]](https://shunsukesaito.github.io/PIFu) [[Code]](https://github.com/shunsukesaito/PIFu)

[PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization](https://arxiv.org/pdf/2004.00452.pdf). CVPR (Oral), 2020. [[Page]](https://shunsukesaito.github.io/PIFuHD) [[Code]](https://github.com/facebookresearch/pifuhd)

[Geo-PIFu: Geometry and Pixel Aligned Implicit Functions for Single-view Human Reconstruction](https://arxiv.org/abs/2006.08072). NeurIPS, 2020.  [[Code]](https://github.com/simpleig/Geo-PIFu)

[StereoPIFu: Depth Aware Clothed Human Digitization via Stereo Vision](https://arxiv.org/abs/2006.08072). CVPR, 2021. [[Page]](https://hy1995.top/StereoPIFuProject) [[Code]](https://github.com/CrisHY1995/StereoPIFu_Code)

[ARCH: Animatable Reconstruction of Clothed Humans](https://arxiv.org/pdf/2004.04572.pdf). CVPR, 2020.  

[S3: Neural Shape, Skeleton, and Skinning Fields for 3D Human Modeling](https://arxiv.org/abs/2101.06571). CVPR, 2021.  

[Detailed Human Avatars from Monocular Video](https://arxiv.org/abs/1808.01338). 3DV, 2018.  [[Code]](https://github.com/thmoa/semantic_human_texture_stitching)

[Monocular Real-Time Volumetric Performance Capture](https://arxiv.org/abs/2007.13988). ECCV, 2020. [[Page]](http://xiuyuliang.cn/monoport) [[Code]](https://github.com/Project-Splinter/MonoPort)

[Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion](https://arxiv.org/abs/2003.01456). CVPR, 2020. [[Page]](http://virtualhumans.mpi-inf.mpg.de/ifnets) [[Code]](https://github.com/jchibane/if-net)

[Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction](https://arxiv.org/abs/2007.11432). ECCV (Oral), 2020. [[Page]](https://virtualhumans.mpi-inf.mpg.de/ipnet) [[Code]](https://github.com/bharat-b7/IPNet)

[PaMIR: Parametric Model-Conditioned Implicit Representation for Image-based Human Reconstruction](https://arxiv.org/abs/2007.03858). TPAMI, 2020. [[Page]](http://www.liuyebin.com/pamir/pamir.html) 

[RIN: Textured Human Model Recovery and Imitation with a Single Image](https://arxiv.org/abs/2011.12024). ArXiv, 2020.  

[3D Human Avatar Digitization from a Single Image](https://www.cs.rochester.edu/u/lchen63/vrcai2019.pdf). VRCAI, 2019.  

[SMPLicit: Topology-aware Generative Model for Clothed People](https://arxiv.org/abs/2103.06871). CVPR, 2021. [[Page]](http://www.iri.upc.edu/people/ecorona/smplicit) [[Code]](https://github.com/enriccorona/SMPLicit)

[SCANimate: Weakly Supervised Learning of Skinned Clothed Avatar Networks](https://arxiv.org/abs/2104.03313). CVPR (Oral), 2021. [[Page]](https://scanimate.is.tue.mpg.de) 

[Reconstructing NBA Players](https://arxiv.org/abs/2007.13303). ECCV, 2020. [[Page]](http://grail.cs.washington.edu/projects/nba_players) [[Code]](https://github.com/luyangzhu/NBA-Players)

[Capturing Detailed Deformations of Moving Human Bodies](https://arxiv.org/abs/2102.07343). ArXiv, 2021.  

[Towards Real-World Category-level Articulation Pose Estimation](https://arxiv.org/abs/2105.03260). CVPR, 2021. [[Page]](https://lasr-google.github.io) 


## Human Depth Estimation


[Learning the Depths of Moving People by Watching Frozen People](https://arxiv.org/abs/1904.11111). CVPR, 2019. [[Page]](https://mannequin-depth.github.io) [[Code]](https://github.com/google/mannequinchallenge)

[A Neural Network for Detailed Human Depth Estimation from a Single Image](https://arxiv.org/abs/1910.01275). ICCV, 2019.  [[Code]](https://github.com/sfu-gruvi-3dv/deep_human)

[Self-Supervised Human Depth Estimation from Monocular Videos](https://arxiv.org/abs/2005.03358). CVPR, 2020.  [[Code]](https://github.com/sfu-gruvi-3dv/Self-Supervised-Human-Depth)

[DressNet: High Fidelity Depth Estimation of Dressed Humans from a Single View Image](None). ArXiv, 2021.  

[Learning High Fidelity Depths of Dressed Humans by Watching Social Media Dance Videos](https://arxiv.org/abs/2103.03319). CVPR (Oral), 2021. [[Page]](https://www.yasamin.page/hdnet_tiktok) [[Code]](https://github.com/yasaminjafarian/HDNet_TikTok)

[Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging](https://arxiv.org/abs/2105.14021). CVPR, 2021. [[Page]](http://yaksoy.github.io/highresdepth) [[Code]](http://yaksoy.github.io/highresdepth)


## Human Motion


[3D Semantic Trajectory Reconstruction from 3D Pixel Continuum](https://www-users.cs.umn.edu/~jsyoon/JaeShin_homepage/SemanticTrajectory.pdf). CVPR, 2018. [[Page]](https://www-users.cs.umn.edu/~jsyoon/Semantic_trajectory) 

[Task-Generic Hierarchical Human Motion Prior using VAEs](https://arxiv.org/abs/2106.04004). ArXiv, 2021.  

[Convolutional Autoencoders for Human Motion Infilling](https://arxiv.org/pdf/2010.11531.pdf). 3DV, 2020.  

[Robust Motion In-betweening](https://arxiv.org/abs/2102.04942). SIGGRAPH, 2020. [[Page]](https://montreal.ubisoft.com/en/robust-motion-in-betweening-2) 

[Single-Shot Motion Completion with Transformer](https://arxiv.org/abs/2103.00776). ArXiv, 2021.  [[Code]](https://github.com/FuxiCV/SSMCT)

[Learning Compositional Representation for 4D Captures with Neural ODE](https://arxiv.org/abs/2103.08271). CVPR (Oral), 2021. [[Page]](https://boyanjiang.github.io/4D-CR) [[Code]](https://github.com/BoyanJIANG/4D-Compositional-Representation)

[Graph Constrained Data Representation Learning for Human Motion Segmentation](https://arxiv.org/abs/2107.13362). ICCV, 2021.  

[Predicting 3D Human Dynamics from Video](https://arxiv.org/abs/1908.04781). ICCV, 2019. [[Page]](https://jasonyzhang.com/phd) [[Code]](https://github.com/jasonyzhang/phd)

[Long-term Human Motion Prediction with Scene Context](https://arxiv.org/pdf/2007.03672.pdf). ECCV (Oral), 2020. [[Page]](https://people.eecs.berkeley.edu/~zhecao/hmp/index.html) [[Code]](https://github.com/ZheC/GTA-IM-Dataset)

[Adversarial Refinement Network for Human Motion Prediction](https://arxiv.org/abs/2011.11221v2). ACCV, 2020.  

[Towards Accurate 3D Human Motion Prediction from Incomplete Observations](https://openaccess.thecvf.com/content/CVPR2021/papers/Cui_Towards_Accurate_3D_Human_Motion_Prediction_From_Incomplete_Observations_CVPR_2021_paper.pdf). CVPR, 2021.  

[Aggregated Multi-GANs for Controlled 3D Human Motion Prediction](https://arxiv.org/abs/2103.09755). AAAI, 2021.  [[Code]](https://github.com/herolvkd/AM-GAN)

[Flow-based Autoregressive Structured Prediction of Human Motion](https://arxiv.org/abs/2104.04391). ArXiv, 2021.  

[TRiPOD: Human Trajectory and Pose Dynamics Forecasting in the Wild](https://arxiv.org/abs/2104.04029). ArXiv, 2021. [[Page]](http://somof.stanford.edu) 

[Multi-level Motion Attention for Human Motion Prediction](https://arxiv.org/abs/2106.09300). ArXiv, 2021.  [[Code]](https://github.com/wei-mao-2019/HisRepItself)

[We are More than Our Joints: Predicting how 3D Bodies Move](https://arxiv.org/abs/2012.00619). CVPR, 2021. [[Page]](https://yz-cnsdqz.github.io/MOJO/MOJO.html) 

[Improving Human Motion Prediction Through Continual Learning](https://arxiv.org/abs/2107.00544). ArXiv, 2021.  

[Synthesizing Long-Term 3D Human Motion and Interaction in 3D](https://arxiv.org/pdf/2012.05522.pdf). CVPR, 2021. [[Page]](https://jiashunwang.github.io/Long-term-Motion-in-3D-Scenes) [[Code]](https://github.com/jiashunwang/Long-term-Motion-in-3D-Scenes)

[GlocalNet: Class-aware Long-term Human Motion Synthesis](https://arxiv.org/abs/2012.10744). MACV, 2021.  

[A Causal Convolutional Neural Network for Motion Modeling and Synthesis](https://arxiv.org/abs/2101.12276). ArXiv, 2021.  

[TrajeVAE - Controllable Human Motion Generation from Trajectories](https://arxiv.org/abs/2104.00351). ArXiv, 2021. [[Page]](https://kacperkan.github.io/trajevae-supplementary) 

[Action-Conditioned 3D Human Motion Synthesis with Transformer VAE](https://arxiv.org/abs/2104.05670). ArXiv, 2021. [[Page]](https://imagine.enpc.fr/~petrovim/actor) 

[Scene-aware Generative Network for Human Motion Synthesis](https://arxiv.org/abs/2105.14804). CVPR, 2021.  

[Learning a Family of Motor Skills from a Single Motion Clip](http://mrl.snu.ac.kr/research/ProjectParameterizedMotion/ParameterizedMotion.pdf). SIGGRAPH, 2021. [[Page]](http://mrl.snu.ac.kr/research/ProjectParameterizedMotion/ParameterizedMotion.html) [[Code]](https://github.com/syleemrl/ParameterizedMotion)

[Character Controllers using Motion VAEs](https://arxiv.org/abs/2103.14274). ToG, 2020. [[Page]](https://www.cs.ubc.ca/~hyuling/projects/mvae) [[Code]](https://github.com/electronicarts/character-motion-vaes)

[Learn to Dance with AIST++: Music Conditioned 3D Dance Generation](https://arxiv.org/abs/2101.08779). ArXiv, 2021. [[Page]](https://google.github.io/aichoreographer) 

[Learning Speech-driven 3D Conversational Gestures from Video](https://arxiv.org/abs/2102.06837). ArXiv, 2021.  

[DanceNet3D: Music Based Dance Generation with Parametric Motion Transformer](https://arxiv.org/abs/2103.10206). ArXiv, 2021. [[Page]](https://huiye-tech.github.io/project/dancenet3d) [[Code]](https://github.com/huiye-tech/DanceNet3D)


## Human-Object Interaction


[Perceiving 3D Human-Object Spatial Arrangements from a Single Image in the Wild](https://arxiv.org/abs/2007.15649). ECCV, 2020. [[Page]](https://jasonyzhang.com/phosa) [[Code]](https://github.com/facebookresearch/phosa)

[Resolving 3D Human Pose Ambiguities with 3D Scene Constraints](https://arxiv.org/abs/1908.06963). ICCV, 2019. [[Page]](https://prox.is.tue.mpg.de) [[Code]](https://github.com/MohameHassan/PROX)

[GRAB: A Dataset of Whole-Body Human Grasping of Objects](https://arxiv.org/abs/2008.11200). ECCV, 2020. [[Page]](https://grab.is.tue.mpg.de) [[Code]](https://github.com/otaheri/GRAB)

[Populating 3D Scenes by Learning Human-Scene Interaction](https://arxiv.org/abs/2012.11581). CVPR, 2021. [[Page]](https://posa.is.tue.mpg.de) [[Code]](https://github.com/mohamedhassanmus/POSA)

[Human POSEitioning System (HPS): 3D Human Pose Estimation and Self-localization in Large Scenes from Body-Mounted Sensors](https://arxiv.org/abs/2103.17265). CVPR, 2021. [[Page]](http://virtualhumans.mpi-inf.mpg.de/hps) 

[Holistic 3D Human and Scene Mesh Estimation from Single View Images](https://arxiv.org/abs/2012.01591). CVPR, 2021.  

[Soft Walks: Real-Time, Two-Ways Interaction between a Character and Loose Grounds](https://arxiv.org/abs/2104.10898). Eurographics, 2021.  

[RobustFusion: Robust Volumetric Performance Reconstruction under Human-object Interactions from Monocular RGBD Stream](https://arxiv.org/abs/2104.14837). TPAMI, 2021.  


## Animation


[Predicting Animation Skeletons for 3D Articulated Models via Volumetric Nets](http://people.cs.umass.edu/~zhanxu/papers/AnimSkelVolNet.pdf). 3DV (Oral), 2019. [[Page]](https://people.cs.umass.edu/~zhanxu/projects/AnimSkelVolNet/) [[Code]](https://github.com/zhan-xu/AnimSkelVolNet)

[RigNet: Neural Rigging for Articulated Characters](https://people.cs.umass.edu/~zhanxu/papers/RigNet.pdf). SIGGRAPH, 2020. [[Page]](https://zhan-xu.github.io/rig-net) [[Code]](https://github.com/zhan-xu/RigNet)

[HeterSkinNet: A Heterogeneous Network for Skin Weights Prediction](https://arxiv.org/abs/2103.10602). I3D, 2021.  

[Skeleton-Aware Networks for Deep Motion Retargeting](https://deepmotionediting.github.io/papers/skeleton-aware-camera-ready.pdf). SIGGRAPH, 2020. [[Page]](https://deepmotionediting.github.io/retargeting) [[Code]](https://github.com/DeepMotionEditing/deep-motion-editing)

[Motion Retargetting based on Dilated Convolutions and Skeleton-specific Loss Functions](https://diglib.eg.org/bitstream/handle/10.1111/cgf13947/v39i2pp497-507.pdf). Eurographics, 2020. [[Page]](https://sites.google.com/view/retargetting-tdcn) [[Code]](https://sites.google.com/view/https%3A%2F%2Fgithub.com%2Fmedialab-ku%2Fretargetting-tdcn)

[Flow Guided Transformable Bottleneck Networks for Motion Retargeting](https://arxiv.org/abs/2106.07771). CVPR, 2021.  

[Functionality-Driven Musculature Retargeting](https://arxiv.org/abs/2007.15311). CGF, 2020. [[Page]](http://mrl.snu.ac.kr/research/ProjectFunctionalityDriven/fdmr.htm) [[Code]](https://github.com/snumrl/SkelGen)

[A Deep Emulator for Secondary Motion of 3D Characters](https://arxiv.org/abs/2103.01261). CVPR (Oral), 2021. [[Page]](http://barbic.usc.edu/deepEmulator/index.html) 

[DeePSD: Automatic Deep Skinning And Pose Space Deformation For 3D Garment Animation](https://arxiv.org/pdf/2009.02715). ArXiv, 2020.  

[UniCon: Universal Neural Controller For Physics-based Character Motion](https://arxiv.org/abs/2011.15119). ArXiv, 2020. [[Page]](https://nv-tlabs.github.io/unicon) 

[Learning Skeletal Articulations With Neural Blend Shapes](https://arxiv.org/abs/2105.02451). SIGGRAPH, 2021. [[Page]](https://peizhuoli.github.io/neural-blend-shapes) [[Code]](https://github.com/PeizhuoLi/neural-blend-shapes)


## Cloth/Try-On


[DeepWrinkles: Accurate and Realistic Clothing Modeling](https://arxiv.org/abs/1808.03417). ECCV (Oral), 2018.  

[Wallpaper Pattern Alignment along Garment Seams](https://igl.ethz.ch/projects/aligned-seams/Aligned-Seams-2019.pdf). SIGGRAPH, 2019. [[Page]](https://igl.ethz.ch/projects/aligned-seams) 

[Reﬂection Symmetry in Textured Sewing Patterns](https://igl.ethz.ch/projects/reflection-symmetry-sewing/sym_wallpaper_patter.pdf). VMV, 2019. [[Page]](https://igl.ethz.ch/projects/reflection-symmetry-sewing) 

[Deep Fashion3D: A Dataset and Benchmark for 3D Garment Reconstruction from Single-view Images](https://arxiv.org/abs/2003.12753). ECCV (Oral), 2020. [[Page]](https://kv2000.github.io/2020/03/25/deepFashion3DRevisited) 

[TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style](https://arxiv.org/abs/2003.04583). CVPR (Oral), 2020. [[Page]](http://virtualhumans.mpi-inf.mpg.de/tailornet) [[Code]](https://github.com/chaitanya100100/TailorNet)

[Learning-Based Animation of Clothing for Virtual Try-On](http://dancasas.github.io/docs/santesteban_Eurographics2019.pdf). Eurographics, 2019. [[Page]](http://dancasas.github.io/projects/LearningBasedVirtualTryOn/index.html) 

[Self-Supervised Collision Handling via Generative 3D Garment Models for Virtual Try-On](https://arxiv.org/abs/2105.06462). CVPR, 2021. [[Page]](http://mslab.es/projects/SelfSupervisedGarmentCollisions) 

[Physically Based Neural Simulator for Garment Animation](https://arxiv.org/abs/2012.11310). ArXiv, 2020.  

[Deep Deformation Detail Synthesis for Thin Shell Models](https://arxiv.org/abs/2102.11541). ArXiv, 2021.  

[DeepCloth: Neural Garment Representation for Shape and Style Editing](https://arxiv.org/abs/2011.14619). ArXiv, 2020. [[Page]](http://www.liuyebin.com/DeepCloth/DeepCloth.html) 

[3D Custom Fit Garment Design with Body Movement](https://arxiv.org/abs/2102.05462). ArXiv, 2021.  

[Dynamic Neural Garments](https://arxiv.org/abs/2102.11811). ArXiv, 2021.  

[DiffCloth: Differentiable Cloth Simulation with Dry Frictional Contact](https://arxiv.org/abs/2106.05306). ArXiv, 2021.  

[Example-based Real-time Clothing Synthesis for Virtual Agents](https://arxiv.org/abs/2101.03088). ArXiv, 2021.  

[BCNet: Learning Body and Cloth Shape from a Single Image](https://arxiv.org/abs/2004.00214). ECCV, 2020.  [[Code]](https://github.com/jby1993/BCNet)

[Fully Convolutional Graph Neural Networks for Parametric Virtual Try-On](https://arxiv.org/abs2009.04592). SCA, 2020. [[Page]](http://mslab.es/projects/FullyConvolutionalGraphVirtualTryOn) 

[Neural 3D Clothes Retargeting from a Single Image](https://arxiv.org/abs/2102.00062). ArXiv, 2021.  


## Neural Rendering


[Neural3D: Light-weight Neural Portrait Scanning via Context-aware Correspondence Learning](https://dl.acm.org/doi/abs/10.1145/3394171.3413734). ACM MM, 2020.  

[Multi-view Neural Human Rendering](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Multi-View_Neural_Human_Rendering_CVPR_2020_paper.pdf). CVPR, 2020. [[Page]](https://wuminye.com/NHR) [[Code]](https://github.com/wuminye/NHR)

[NeuralHumanFVV: Real-Time Neural Volumetric Human Performance Rendering using RGB Cameras](https://arxiv.org/abs/2103.07700). CVPR, 2021.  

[Few-shot Neural Human Performance Rendering from Sparse RGBD Videos](https://arxiv.org/abs/2107.06505). ArXiv, 2021.  

[ANR: Articulated Neural Rendering for Virtual Avatars](https://arxiv.org/pdf/2012.12890.pdf). ArXiv, 2020. [[Page]](https://anr-avatars.github.io) 

[SMPLpix: Neural Avatars from 3D Human Models](https://arxiv.org/abs/2008.06872). WACV, 2020. [[Page]](https://sergeyprokudin.github.io/smplpix) [[Code]](https://github.com/sergeyprokudin/smplpix)

[Vid2Actor: Free-viewpoint Animatable Person Synthesis from Video in the Wild](https://arxiv.org/abs/2012.12884). ArXiv, 2020. [[Page]](https://grail.cs.washington.edu/projects/vid2actor) 

[Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans](https://arxiv.org/abs/2012.15838). CVPR, 2021. [[Page]](https://zju3dv.github.io/neuralbody) [[Code]](https://github.com/zju3dv/neuralbody)

[Neural Actor: Neural Free-view Synthesis of Human Actors with Pose Control](https://arxiv.org/abs/2106.02019). ArXiv, 2021.  

[StylePeople: A Generative Model of Fullbody Human Avatars](https://arxiv.org/abs/2104.08363). CVPR, 2021. [[Page]](http://saic-violet.github.io/style-people) [[Code]](https://github.com/saic-vul/style-people)

[A-NeRF: Surface-free Human 3D Pose Refinement via Neural Rendering](https://arxiv.org/abs/2102.06199). ArXiv, 2021. [[Page]](https://lemonatsu.github.io/ANeRF-Surface-free-Pose-Refinement) 

[D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://arxiv.org/abs/2011.13961). CVPR, 2021. [[Page]](https://www.albertpumarola.com/research/D-NeRF/index.html) 

[Neural Articulated Radiance Field](https://arxiv.org/abs/2104.03110). ArXiv, 2021.  [[Code]](https://github.com/nogu-atsu/NARF)

[Animatable Neural Radiance Fields for Human Body Modeling](https://arxiv.org/abs/2105.02872). ArXiv, 2021. [[Page]](https://zju3dv.github.io/animatable_nerf) [[Code]](https://github.com/zju3dv/animatable_nerf)

[Editable Free-viewpoint Video Using a Layered Neural Representation](https://arxiv.org/abs/2104.14786). SIGGRAPH, 2021. [[Page]](https://www.youtube.com/watch?v=Wp4HfOwFGP4) 

[MoCo-Flow: Neural Motion Consensus Flow for Dynamic Humans in Stationary Monocular Cameras](https://arxiv.org/abs/2106.04477). ArXiv, 2021.  


## Dataset


[3DPW: Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera](https://openaccess.thecvf.com/content_ECCV_2018/papers/Timo_von_Marcard_Recovering_Accurate_3D_ECCV_2018_paper.pdf). ECCV, 2018. [[Page]](http://virtualhumans.mpi-inf.mpg.de/3DPW) 

[AMASS: Archive of Motion Capture as Surface Shapes](https://arxiv.org/abs/1904.03278). ICCV, 2019. [[Page]](https://amass.is.tue.mpg.de) [[Code]](https://github.com/nghorbani/amass)

[3DBodyTex: Textured 3D Body Dataset](https://orbilu.uni.lu/bitstream/10993/36414/1/saint_et_al-3dbodytex-3dv_2018.pdf). 3DV, 2018. [[Page]](https://cvi2.uni.lu/datasets) 

[Motion Capture from Internet Videos](https://arxiv.org/pdf/2008.07931.pdf). ECCV (Oral), 2020. [[Page]](https://zju3dv.github.io/iMoCap) [[Code]](https://github.com/zju3dv/iMoCap)

[3DPeople: Modeling the Geometry of Dressed Humans](https://arxiv.org/abs/1904.04571). ICCV, 2019. [[Page]](https://cv.iri.upc-csic.es) [[Code]](https://github.com/albertpumarola/3DPeople-Dataset)

[Full-Body Awareness from Partial Observations](https://arxiv.org/abs/2008.06046). ECCV, 2020. [[Page]](https://crockwell.github.io/partial_humans) [[Code]](https://github.com/crockwell/partial_humans)

[Object-Occluded Human Shape and Pose Estimation from a Single Color Image](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Object-Occluded_Human_Shape_and_Pose_Estimation_From_a_Single_Color_CVPR_2020_paper.pdf). CVPR, 2020. [[Page]](https://www.yangangwang.com/papers/ZHANG-OOH-2020-03.html) [[Code]](https://gitee.com/seuvcl/CVPR2020-OOH)

[HUMBI: A Large Multiview Dataset of Human Body Expressions](https://arxiv.org/abs/1812.00281). CVPR, 2020. [[Page]](https://humbi-data.net) [[Code]](https://github.com/zhixuany/HUMBI)

[SMPLy Benchmarking 3D Human Pose Estimation in the Wild](https://arxiv.org/abs/2012.02743). 3DV (Oral), 2020. [[Page]](https://europe.naverlabs.com/research/computer-vision/mannequin-benchmark) 

[Reconstructing 3D Human Pose by Watching Humans in the Mirror](https://arxiv.org/abs/2104.00340). CVPR (Oral), 2021. [[Page]](https://zju3dv.github.io/Mirrored-Human) [[Code]](https://github.com/zju3dv/Mirrored-Human)

[AGORA: Avatars in Geography Optimized for Regression Analysis](https://arxiv.org/abs/2104.14643). CVPR, 2021. [[Page]](https://agora.is.tue.mpg.de) 

[BABEL: Bodies, Action and Behavior with English Labels](https://arxiv.org/abs/2106.09696). CVPR, 2021. [[Page]](https://babel.is.tue.mpg.de) 

-----

## [Back to Top](#table-of-contents)
