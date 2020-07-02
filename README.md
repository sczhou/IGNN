# IGNN

Code repo for "Cross-Scale Internal Graph Neural Network for Image Super-Resolution" &nbsp; [[arXiv]](https://arxiv.org/pdf/2006.16673.pdf)

<p align="center">
  <img width=95% src="https://user-images.githubusercontent.com/14334509/86379250-34450200-bcbd-11ea-9a85-aab4bc73cd2d.png">
</p>

**Note that** the paper is under review. Our code and models will be released once the paper is accepted.

## Abstract

Non-local self-similarity in natural images has been well studied as an effective prior in image restoration. However, for single image super-resolution (SISR), most existing deep non-local methods (e.g., non-local neural networks) only exploit similar patches within the same scale of the low-resolution (LR) input image. Consequently, the restoration is limited to using the same-scale information while neglecting potential high-resolution (HR) cues from other scales. In this paper, we explore the cross-scale patch recurrence property of a natural image, i.e., similar patches tend to recur many times across different scales. This is achieved using a novel cross-scale internal graph neural network (**IGNN**). Specifically, we dynamically construct a cross-scale graph by searching k-nearest neighboring patches in the downsampled LR image for each query patch in the LR image. We then obtain the corresponding k HR neighboring patches in the LR image and aggregate them adaptively in accordance to the edge label of the constructed graph. In this way, the HR information can be passed from k HR neighboring patches to the LR query patch to help it recover more detailed textures. Besides, these internal image-specific LR/HR exemplars are also significant complements to the external information learned from the training dataset. Extensive experiments demonstrate the effectiveness of **IGNN** against the state-of-the-art SISR methods including existing non-local networks on standard benchmarks.

## Some Visual Results (x4)

![image](https://user-images.githubusercontent.com/14334509/86381317-c817cd80-bcbf-11ea-9b29-1f60ebfaa2e5.png)

![image](https://user-images.githubusercontent.com/14334509/86384957-129a4980-bcc2-11ea-9405-c81c3af6d01f.png)



## License

This project is open sourced under MIT license.

