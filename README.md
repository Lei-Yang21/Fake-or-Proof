<a id="readme-top"></a>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="">
    <img src="" alt="Logo" width="80" height="80">
  </a> -->

  <h3 align="center">FAKE-OR-PROOF</h3>

</div>



<!-- TABLE OF CONTENTS -->
<!-- <details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>

  </ol>
</details>-->



<!-- ABOUT THE PROJECT -->
## About The Project
Real or fake? Our application will have the answer.
With the rise of AI-generated images flooding the internet, it's getting harder to tell what's real from what is not. This problem inspired our team, as a part of the Machine Learning on Design System course, to implement a solution.
Our project aims to build an application that can analyze images and determine whether AI created by AI or captured in the real world.

## Dataset Aquisition & EDA

For this milestone, we first to gathered data and tried to understand its characteristics. We downloaded a training dataset of 12,000 images consisting of:
* 6,000 real images
* 6,000 AI-generated images

Our EDA focused on identifying key patterns and biases that could influence model performance. We examined:
-   **Image Dimensions:** Visualized the distribution of image heights and widths.
-   **Aspect Ratios:** Investigated whether AI-generated images tend to have unique proportions.
-   **Color Channels:** Analyzed RGB distributions to detect potential biases between real and fake images.
-   **Standardized Image Sizes:** All images were resized to a uniform shape to ensure consistent input for the model.

## Model Architecture

We implemented **MobileNet_V2**, a lightweight and efficient model architecture with pretrained weights available in PyTorch, making it well-suited for transfer learning.

Since this is a binary classification problem, we used:

-   **Loss Function:** CrossEntropy Loss to quantify the prediction error.
-   **Optimizer:** Adam Optimizer for adaptive learning rate adjustments during training.


## Model Evaluation & Prediction

To assess model performance, we used the following metrics:

-   **Specificity**
-   **Sensitivity**
-   **F1-score**
-   **Precision**
    

Each of these metrics achieved scores above **0.88**, indicating strong predictive performance.

Once the model was trained, we evaluated it on a separate validation set by comparing its predictions with true labels. We also developed a `predict_function()` to classify individual images manually and provide:

-   The **predicted label**
-   The **confidence score**
-   A **Grad-CAM heatmap** to highlight the most influential regions of the image

## API & Interface

The model is deployed via a public API accessible at: [**Fake-or-Proof API**](Waiting)

We also built a **Streamlit interface** for a user-friendly, interactive experience. The app allows users to upload images, view predictions, and examine Grad-CAM visual explanations—all in real time.