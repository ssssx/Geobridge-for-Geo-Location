
<div align="center">

<h1>GeoBridge: A Semantic-Anchored Multi-View Foundation Model Bridging Images and Text for Geo-Localization</h1>

Zixuan Song<sup>1,3</sup>, 
Jing Zhang<sup>2,3 †</sup>, 
Di Wang<sup>2,3 †</sup>, 
Zidie Zhou<sup>1</sup>, 
Wenbin Liu<sup>1</sup>, 
Haonan Guo<sup>2,3 †</sup>,
En Wang<sup>1 †</sup>,
Bo Du<sup>2,3 †</sup>.

<sup>1</sup> Jilin University,  <sup>2</sup> Wuhan University,  <sup>3</sup> Zhongguancun Academy.

<sup>†</sup> Corresponding author

</div>

<p align="center">
  <a href="#-update">Update</a> |
  <a href="#-abstract">Abstract</a> |
  <a href="#-datasets">Datasets</a> |
  <a href="#-model">Models</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-statement">Statement</a>
</p >


## 🔥 Update

**2026.04.08**
- The model is now available.

**2026.03.26**
- The code is now available.

**2026.02.21**
- The paper is accepted by **CVPR 2026**! 🎉

**2025.12.03**
- The paper is post on arXiv! **([arXiv GeoBridge](http://arxiv.org/abs/2512.02697))** 

## 🌞 Abstract

Cross-view geo-localization infers a location by retrieving geo-tagged reference images that visually correspond to a query image. However, the traditional satellite-centric paradigm limits robustness when high-resolution or up-to-date satellite imagery is unavailable. It further underexploits complementary cues across views (e.g., drone, satellite, and street) and modalities (e.g., language and image). To address these challenges, we propose GeoBridge, a foundation model that performs bidirectional matching across views and supports language-to-image retrieval. Going beyond traditional satellite-centric formulations, GeoBridge builds on a novel semantic-anchor mechanism that bridges multi-view features through textual descriptions for robust, flexible localization. In support of this task, we construct GeoLoc, the first large-scale, cross-modal, and multi-view aligned dataset comprising over 50,000 pairs of drone, street-view panorama, and satellite images as well as their textual descriptions, collected from 36 countries, ensuring both geographic and semantic alignment. We performed broad evaluations across multiple tasks. Experiments confirm that GeoLoc pre-training markedly improves geo-location accuracy for GeoBridge while promoting cross-domain generalization and cross-modal knowledge transfer.

<figure>
<div align="center">
<img src=Figs/intro.png width="100%">
</div>

<div align='center'>
 
**Figure 1. Schematic diagram of GeoBridge.**

</div>
<br>

<div align="center">
<img src=Figs/method.png width="100%">
</div>

<div align='center'>

**Figure 2. Overall workflow.**

</div>

## 📖 Datasets

Coming Soon.

## 🚀 Model

The GeoBridge model is now available on Hugging Face: **[Son12s/GeoBridge](https://huggingface.co/Son12s/GeoBridge)**.
Please visit the repository page for download and usage details.


## 🔨 Usage

### Data Preparation

Please organize the dataset as follows:

```text
data/
├── train/
│   ├── drone/
│   ├── satellite/
│   └── street/
├── val/
│   ├── drone/
│   ├── satellite/
│   └── street/
└── test/
    ├── drone/
    ├── satellite/
    └── street/
```
### Checkpoints
Please download the pretrained checkpoints and place them under:

```text
checkpoints/
├── opts.yaml
└── best_net.pth
```
### Evaluation
Supported evaluation settings include:

- drone ↔ satellite retrieval
- street ↔ satellite retrieval
- satellite ↔ street retrieval
- text → image retrieval

### Example Tasks
GeoBridge supports the following tasks:

1. **Cross-view geo-localization**  
   Retrieve geographically matched reference images across different views.

2. **Bidirectional image retrieval**  
   Perform retrieval between drone, satellite, and street-view imagery.

3. **Language-to-image retrieval**  
   Use natural language descriptions to retrieve semantically aligned geo-images.

## 🍭 Results

### GeoLoc Benchmark

We compare **GeoBridge** with previous methods on the **GeoLoc** dataset.  
**D** denotes **Drone**, **P** denotes **Street-View Panorama**, and **S** denotes **Satellite**.  
**Bold** indicates the best result, and <u>underline</u> indicates the second-best result.

#### Part I

| Method | D2S R@1 | D2S AP | S2D R@1 | S2D AP | D2P R@1 | D2P AP | P2D R@1 | P2D AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sample4Geo [6] | <u>27.27</u> | <u>39.69</u> | <u>28.70</u> | <u>40.32</u> | <u>29.51</u> | <u>31.17</u> | 15.56 | <u>29.68</u> |
| MEAN [2] | 21.52 | 27.08 | 21.38 | 26.97 | 13.08 | 17.76 | 1.87 | 7.74 |
| DAC [36] | 6.19 | 8.41 | 13.91 | 15.16 | 13.74 | 15.16 | <u>19.34</u> | 23.01 |
| CAMP [34] | 19.60 | 24.39 | 14.88 | 18.75 | 11.31 | 12.34 | 11.46 | 19.16 |
| MCCG [26] | 12.23 | 13.22 | 14.73 | 17.70 | 15.51 | 19.11 | 12.90 | 15.75 |
| CCR [7] | 12.91 | 15.56 | 13.89 | 16.04 | 10.91 | 12.34 | 10.81 | 14.77 |
| **GeoBridge (ours)** | **45.05** | **49.05** | **44.81** | **48.76** | **41.22** | **43.54** | **41.15** | **43.41** |

#### Part II

| Method | P2S R@1 | P2S AP | S2P R@1 | S2P AP | D2P R@1 | D2P AP | P2D R@1 | P2D AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| panorama-BEV [40] | 10.21 | 12.70 | 18.70 | 20.56 | 16.32 | 17.11 | 16.34 | 18.36 |
| Sample4Geo [6] | 16.82 | 13.64 | <u>18.87</u> | 17.78 | <u>29.51</u> | <u>31.17</u> | 15.56 | <u>29.68</u> |
| AuxGeo [37] | 13.74 | 17.09 | 15.61 | <u>22.96</u> | 9.34 | 12.69 | 14.70 | 17.40 |
| FRGeo [43] | <u>17.13</u> | 18.70 | 14.35 | 19.69 | 13.69 | 19.34 | 14.38 | 15.00 |
| HC-Net [32] | 14.17 | 15.05 | 11.79 | 12.03 | 12.83 | 13.31 | <u>18.79</u> | 18.93 |
| TransGeo [48] | 11.21 | <u>23.77</u> | 13.74 | 13.43 | 18.70 | 23.41 | 17.48 | 23.26 |
| **GeoBridge (ours)** | **38.87** | **42.10** | **39.20** | **41.96** | **41.22** | **43.54** | **41.15** | **43.41** |


<div align="center">
<img src=Figs/over-result.png width="100%">
</div>


## ⭐ Citation

If you find GeoBridge helpful, please give a ⭐ and cite it as follows:

```
@misc{song2025geobridgesemanticanchoredmultiviewfoundation,
      title={GeoBridge: A Semantic-Anchored Multi-View Foundation Model Bridging Images and Text for Geo-Localization}, 
      author={Zixuan Song and Jing Zhang and Di Wang and Zidie Zhou and Wenbin Liu and Haonan Guo and En Wang and Bo Du},
      year={2025},
      eprint={2512.02697},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.02697}, 
}
```

## 🎺 Statement

For any other questions please contact [Zixuan Song](https://github.com/ssssx) at [jlu.edu.cn](songzx24@mails.jlu.edu.cn) or [gmail.com](estrellaluminous@gmail.com).


