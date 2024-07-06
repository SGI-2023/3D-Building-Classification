# Multi-Modal Diffusion Models For 3D Building Classification 🏢🌐

Team: Biruk Abere, Berfin Inal, Gabriele Dominici, Meher Nigam, Alex Li,  Nursena Koprucu, Sharvaree Vadgama, Le Xue, Shicheng Xu, Alberto Tono 

This repository contains the implementation of our approach that integrates the methodologies of the LION paper and the Diffusion Classifier for advanced 3D building classification. 

![3D Building Classification](https://github.com/SGI-2023/3D-Building-Classification/blob/main/3D%20Buildings.png)

# Table of Contents 📚

  * Introduction
  * Features
  * Data Sources
  * Approach
  * Installation
  * Results
  * License
  * Contact

# **Introduction** 🌟

3D building classification is the computational process of categorizing three-dimensional representations of architectural structures. This involves analyzing volumetric data to percieve distinct architectural features,spatial configurations,and other instrinisc characterstics of buildings. 

We've developed a methodology that intergrates a generative capabilities with classification strengths. At the core of our approach is a hierarchical latent space that captures features of 3D building structures. This representation is then subjected to a diffusion process ,which perturbs the data iteratively. Post-diffusion, a deep neural network is employed to denoise the data, extracting essential features that are crucial for classification.  

Following the denoising, a classification model is introduced which predicts the building category based on the denoised latent representation. Our system is designed to handle multi-modal data, allowing it to process and integrate information from various data types, enhancing its classification accuracy. 

# Features 


# Approach 🔍

Our project intergrates the capabilities of LION, a generative model adept at producing detailed 3D shapes,with the strengths of a diffusion classifier for zero-shot classification. By harnessing LION's ability to generate diverse 3D shapes and feeding them to the diffusion classifier, we achieve nuanced categorization of 3D builings. 

  * **Hierarchical Latent Space** :- At the core, we utilize LION's hieratchical latent space. 3D point cloud is encode in to a dual-layered latent representation and a point-structured latent space. This ensures a comprehensive capture of both macto and micro features of 3D building structures. 

  * **Diffusion Process** :- After the encoding phase, the data is subjected to a diffusion process. This action iteratively introduces disturbances to the data,setting the stage for the upcoming denosing step. 

  * **Denoising & Classification** :- A deep neural network is trained to denoise the disrupted data. After this step, a classification component, conditioned on class labels determines the building's category by assessing which diffusion class most accurately corresponds to the introduced nose. 

  * **Multi-modal Intergration** :- Our model harnesses the power of multi modal machine learning, seamlessly intergrating diverse data modalities - 3D point cloud data, textual descriptions, 2D images to offer a comprehensive approach to 3D building classification. 
  
# Data Source 📊

For this project, we used the dataset provided by the BuildingNet Challenge, available on [here](https://docs.google.com/forms/d/e/1FAIpQLSevg7fWWMYYMd1vaOdDloUX_55VOQK7PqS1DlniFV7_vuoI0w/viewform). To access the dataset, you are required to fill out a specific form. This dataset is a collection of 3D building structures, which serves as the foundation for our diffusion 3D building classification model. 


# Installation 🔧

  * git clone https://github.com/SGI-2023/3D-Building-Classification.git
  * cd multi-modal-diffusion-models
  * pip install -r requirements.txt

# Results  📈


# Licence 📜




