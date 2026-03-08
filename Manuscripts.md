> []{#_Hlk181473546 .anchor}A DMMA based YOLO Approach for Small Ship
> Detection of Optical Remote Sensing Images [^1]
>
> Shuai Yuan^a,c,d\*^ , Huize Dou^a,c,d^ , Jinyu Geng^b^, Fangjun
> Luan^a,c,d^, Xiaowen Zhang^e,f\ ^
>
> []{#_bookmark0 .anchor}
>
> *^a^School of Computer Science and Engineering, Shenyang Jianzhu
> University, Shenyang 110168, China*
>
> *^b^School of Electrical and Control Engineering,Shenyang Jianzhu
> University, Shenyang 110168, China*
>
> *^c^Liaoning Province Big Data Management and Analysis Laboratory of
> Urban Construction, Shenyang 110168, China*
>
> *^d^Shenyang Branch of National Special Computer Engineering
> Technology Research Center, Shenyang 110168, China*
>
> *^e^Changchun Institute of Optics, Fine Mechanics and Physics, Chinese
> Academy of Sciences, Changchun 130033, China*
>
> *^f^University of Chinese Academy of Sciences, Beijing 100049, China*

+----------------------+----------------------+----------------------+
| **A R T I C L E I N  |                      | **A B S T R A C T.** |
| F O.**               |                      |                      |
+======================+======================+======================+
| Keywords:            |                      | Due to complex       |
|                      |                      | backgrounds,         |
| Small Target         |                      | detection of small   |
| Detection            |                      | targets in remote    |
|                      |                      | sensing images is an |
| Deep Learning        |                      | important and        |
|                      |                      | challenging task,    |
| Self-attentive       |                      | especially the ship  |
| Mechanism            |                      | targets are easily   |
|                      |                      | affected by reefs,   |
| Effective Channel    |                      | waves, etc. The YOLO |
| Attention            |                      | (You Only Look Once) |
|                      |                      | series algorithms    |
| You Only Look Once   |                      | are widely used in   |
|                      |                      | optical remote       |
|                      |                      | sensing image object |
|                      |                      | detection due to its |
|                      |                      | advantage of fast    |
|                      |                      | detection speed.     |
|                      |                      | However, the lack of |
|                      |                      | discriminative       |
|                      |                      | features of small    |
|                      |                      | ship targets will    |
|                      |                      | lead to the problem  |
|                      |                      | of false detection   |
|                      |                      | and omission.        |
|                      |                      | Therefore, we        |
|                      |                      | propose a new        |
|                      |                      | algorithm, called    |
|                      |                      | DMMA (Difference     |
|                      |                      | Mask Mixed           |
|                      |                      | Attention) based     |
|                      |                      | YOLO approach, which |
|                      |                      | can improve the      |
|                      |                      | model detection      |
|                      |                      | accuracy without     |
|                      |                      | significantly        |
|                      |                      | increasing the model |
|                      |                      | size. First of all,  |
|                      |                      | we introduce a       |
|                      |                      | self-attention       |
|                      |                      | mechanism to replace |
|                      |                      | some convolutional   |
|                      |                      | layers for capturing |
|                      |                      | global information   |
|                      |                      | of the feature map   |
|                      |                      | while utilizing the  |
|                      |                      | sliding window       |
|                      |                      | strategy to avoid    |
|                      |                      | extensive            |
|                      |                      | computational costs. |
|                      |                      | Moreover, we propose |
|                      |                      | a DMMA model that    |
|                      |                      | leverages the        |
|                      |                      | differences between  |
|                      |                      | target features and  |
|                      |                      | the background to    |
|                      |                      | extract the          |
|                      |                      | effective object     |
|                      |                      | edge information in  |
|                      |                      | the feature map,     |
|                      |                      | thus reducing the    |
|                      |                      | influence of         |
|                      |                      | background           |
|                      |                      | information.         |
|                      |                      | Furthermore, we      |
|                      |                      | integrate the ECA    |
|                      |                      | (Effective Channel   |
|                      |                      | Attention) mechanism |
|                      |                      | into the DMMA        |
|                      |                      | mechanism to achieve |
|                      |                      | information          |
|                      |                      | interaction between  |
|                      |                      | channels. Finally,   |
|                      |                      | the validating       |
|                      |                      | experiment has been  |
|                      |                      | performed to         |
|                      |                      | illustrate that DMMA |
|                      |                      | based YOLOv5         |
|                      |                      | achieved better      |
|                      |                      | performance on all   |
|                      |                      | three datasets,      |
|                      |                      | compared with the    |
|                      |                      | original YOLOv5      |
|                      |                      | model. Additionally, |
|                      |                      | compared with        |
|                      |                      | current algorithms,  |
|                      |                      | the experimental     |
|                      |                      | results represent    |
|                      |                      | that the DMMA based  |
|                      |                      | YOLOv5 has a more    |
|                      |                      | promising and        |
|                      |                      | competitive          |
|                      |                      | performance than     |
|                      |                      | other algorithms.    |
|                      |                      | These results        |
|                      |                      | suggest that the     |
|                      |                      | DMMA based YOLOv5    |
|                      |                      | algorithm enhances   |
|                      |                      | detection accuracy   |
|                      |                      | while maintaining    |
|                      |                      | detection speed.     |
+----------------------+----------------------+----------------------+

> []{#1_Introduction .anchor}

**Introduction **
=================

> Image object detection is an important technology in the field of
> artificial intelligence. With the development of technologies such as
> Deep Learning, Object detection including face detection and vehicle
> detection are becoming more widely used, and also its requirement is
> higher. In recent years, ship detection in remote sensing images has
> received increasing attention due to its crucial role in these
> applications such as military reconnaissance, maritime security and
> port management. Compared with other imaging methods, optical remote
> sensing images have advantages of being similar to natural images and
> being able to visually represent object and background information,
> but their characteristics of wide perception range, small target and
> complex background make ship detection a very challenging task.
>
> Object detection methods mainly include traditional methods and deep
> learning methods. The classical traditional object detection methods
> mainly include Viola-Jones, HOG Detector and DPM, which generally
> suffer from low efficiency, poor accuracy and real-time performance.
> And these problems have been greatly improved with the advent of deep
> learning method. The two most interesting methods of deep
> learning-based object detection are the two-stage detection method
> represented by R-CNN and the one-stage detection method represented by
> YOLO. R-CNN (Jadhav et al., 2014) was the first algorithm to
> successfully apply deep learning to object detection. In order to
> improve the detection speed, Fast R-CNN (Girshick 2015) and Faster
> R-CNN (Shaoqing Ren et al., 2016) were proposed one after another, but
> the two-stage detection method is still inferior to the one-stage
> method in terms of speed.[[]{#_bookmark5 .anchor}]{#_bookmark4
> .anchor}
>
> The YOLOv1 algorithm proposed by Redmon (Redmon 2016) and others
> designed the object detection as a regression problem with spatially
> separated bounding boxes and associated class probabilities, which
> greatly improved the detection speed. Subsequently, (Van Etten 2018)
> proposed YOLT (You Only Look Twice), an extension of YOLOv2, which was
> the first application of the YOLO series of algorithms to remote
> sensing images, to investigate the problem of detecting small targets
> in large-area images. Zhongzhen Sun et al. (2021) proposed the
> Bifa-Yolo algorithm, which can detect ships in any direction in remote
> sensing images by adding an angle detection structure to the detection
> head structure of the YOLO algorithm. The detection speed of the YOLO
> series algorithm is faster, but the detection accuracy has not been
> furtherly enhanced. Later a lot of researchers furtherly make
> improvements on the YOLO performance.
>
> Recently the attention mechanism comes into deep learning. The
> attention mechanism originates from the study of human vision for
> target perception, where human attention is focused on the interest
> target region for enhancing detection. The attention mechanism has
> been used in many fields, for example in image detection, where it
> enables the neural network to focus more on the region of attention in
> the image. Therefore, the combination of the attention mechanism with
> the YOLO series of algorithms has been extensively investigated to
> improve detection accuracy while ensuring detection speed. Liqiong
> Chen et al. (2021) combined the spatial attention and channel
> attention mechanisms with YOLOv3, enabling the model to suppress
> irrelevant regions while highlighting salient features useful for ship
> object detection tasks, achieving a balance between detection accuracy
> and detection speed. Jianming Hu et al. (2021) added spatial attention
> and channel attention mechanisms to YOLOv4 and used a new loss
> function constraint detection step to enable the detector to learn the
> shape of ship targets more efficiently, and furtherly improve training
> efficiency and detection accuracy. Although the existing methods
> perform well in most detection tasks, channel attention and spatial
> attention are based on convolutional layers, which can only extract
> partial information from the feature map in each operation and cannot
> validly handle global information, and face great difficulties when
> detecting small objects in complex backgrounds.
>
> On the foundation of the abovementioned analysis, we propose a new
> small ship object detection method: DMMA based YOLOv5, which can
> improve ship object detection without significantly increasing the
> complexity of the neural network structure. On the foundation of the
> network structure of the YOLOv5, we propose an attention mechanism to
> extract richer feature information, reduce irrelevant information in
> the feature maps and highlight the differences between targets and
> backgrounds. We have presented a module of a Difference Masked Mix
> Attention (DMMA) mechanism. First of all, a window multi-headed
> self-attention structure is adopted as the basis for global
> information processing, and the relevant elements in the feature map
> are extracted using the Difference Mask structure. Once more, the
> Effective Channel Attention (ECA) is integrated to reallocate the
> channel weights of the feature map and learn difference features in
> different channels. Then, the DMMA mechanism is adopted to perform the
> feature extraction block of YOLOv5 for improving the detection
> accuracy without significantly increasing the computational effort.
> Finally, the effectiveness of the algorithm is illustrated through
> implementing ablation experiments, and contrasting tests on two
> datasets.
>
> This paper includes the following sections. Section 2 provides an
> overview of the development and current research status of the YOLO
> and attentional algorithms. Section 3 introduces the proposed DMMA
> based YOLOv5 algorithm and represents its network structure and
> discusses the improvements adopted by the DMMA based YOLOv5 algorithm.
> In Section 4, the proposed method is compared with traditional object
> detection algorithms by using experimental results. Finally, Section 5
> makes a summary of the research work.

1.  **Related work**

> YOLO is a representative algorithm among one-stage object detection
> algorithms, and YOLOv1 was the first proposed one-stage object
> detection algorithm with a significant improvement of detection speed
> compared with two-stage object detection algorithms. Then the YOLOv2
> (Redmon, Farhadi, 2017) was proposed, which improved the learning
> speed of the model by adding a Batch Normalization (BN layer) to all
> the convolutional layers of YOLOv1, and promoted the detection
> accuracy through selecting higher resolution images as input. In
> addition, utilization of anchor frames to predict bounding boxes
> simplifies the problem of predicting target bounding boxes and makes
> the network easier to converge. Subsequently, YOLOv3 (Redmon 2018) was
> developed, whose backbone network adopted Darknet-53 with introducing
> a residual module on the backbone network of YOLOv2 and furtherly
> deepened the network. At same time, a Feature Pyramid Network (FPN)
> was employed to make compensation of the poor predictive ability of
> YOLOv1 and YOLOv2 algorithms for multi-scale targets and improve the
> detection accuracy, especially for small objects. YOLOv4 (Bochkovskiy
> et al., 2020) made another improvement to the backbone network by
> using the CSPDarknet-53 network and adding a PAN structure to enhance
> the feature fusion of the network. In addition, Mosaic Date
> Augmentation was applied to the input images for making the model more
> robust. YOLOv5 retained much of the structure of YOLOv4, and added a
> Focus layer to CSPDarknet-53 for faster processing and an SPP layer
> for better fusing multi-scale feature maps. The backbone network of
> YOLOv6 adopted Efficient Rep, and the neck structure was a Rep-PAN
> built on Rep and PAN. YOLOv7 (Chienyao Wang et al., 2023) was
> different in many ideas compared with YOLOv6, and implemented
> improvements such as extended Efficient Layer Aggregation Network and
> reparametrized convolution improvements, which were faster and more
> accurate in more advantageous. To contrast the performance of the
> algorithms, we experimentally validated YOLOv5 and YOLOv7 with the
> MASATI dataset, and the results showed that YOLOv5 has higher
> detection accuracy, so we finally chose to build on the YOLOv5
> algorithm.
>
> The stronger the model generalization ability, the more parameters it
> requires, which consequently increases the amount of information
> stored. To address the information overload caused by this,
> researchers have introduced attention mechanisms. In 2014, attention
> mechanism (Mnih Heess, Graves 2014) was first applied to the field of
> deep learning by using them on RNN models for image classification. In
> the past few years, the attention mechanism has been used in various
> applications such as Natural Language Processing (Tao Shen et
> al.,2018), Image Classification (Fei Wang et al., 2017) and Image
> Captioning (Jiasen Lu et al., 2017) to validate its effectiveness. In
> 2017, Vaswan (2017) proposed the Transformer, which allows the encoder
> and decoder models in machine translation to break free from the
> traditional framework relying on CNNs or RNNs, and instead, is
> entirely based on the attention mechanism. The Vision Transformer
> (ViT) ( Dosovitskiy 2020) leverages the efficiency and scalability of
> the Transformer architecture by applying the self-attention mechanism
> from natural language processing to computer vision. This approach has
> yielded impressive results across various visual tasks. Subsequently,
> Ze Liu et al. (2021) introduced the Swin Transformer, which employs a
> shift window attention mechanism, extending the application of
> Transformers to different segments of the vision domain. Qibin Hou et
> al. (2021) proposed the Coordinate Attention mechanism, which
> integrates positional information into channel features. This enables
> the capture of long-range dependencies in one spatial direction while
> preserving precise positional information in another. Qilong Wang et
> al. (2020) proposed Efficient Channel Attention, which utilizes
> one-dimensional convolution across the channel dimensions of the
> feature map to capture interconnections between different channels,
> achieving significant performance improvements without increasing
> model complexity. Mohammed et al. (2024) provided a comprehensive
> summary of attention mechanisms in computer vision, classifying them
> into spatial attention, channel attention, and self-attention.
> Additionally, Zixiao Zhang et al. (2021) integrated YOLO with
> self-attention to create ViT-YOLO, which processes global information
> in the final layer of the backbone network rather than relying solely
> on partial convolution. HIC-YOLOv5 (Shiyi Tang, Shu Zhang, Yini Fang
> 2024) was proposed, a model that uses higher resolution input images
> to improve the detection accuracy of small objects. And new loss
> functions such as Focal Loss and IoU Loss were introduced to enhance
> the model focus on small targets and the accuracy of object
> localization. Chunlin Ji et al. (2024) proposed YOLO-TLA, a model that
> added a detection layer specifically for small objects within the Neck
> network pyramid architecture, whose addition generates a larger-scale
> feature map to better distinguish the subtle features of small
> objects. Furtherly, the authors integrated the C3CrossCown module into
> the Backbone network. This module used sliding window feature
> extraction, effectively reducing computational demands and the number
> of parameters, making the model more compact. Xingkui Zhu et al.
> (2021) introduced TPH-YOLOv5 for small object detection by embedding a
> self-attention mechanism in the front end of the YOLOv5 detection head
> and incorporating a Convolutional Block Attention Mechanism (Woo et
> al.,2018). This approach helps the network identify regions of
> interest in images with a broad field of view and optimizes small
> object detection in drone scenarios. This work similarly adopted a
> self-attention mechanism at the front end of the detection head to
> process smaller input features, thereby reducing computational effort
> and memory usage.
>
> It can be obviously concluded that there are various ways to embed
> attention mechanisms in YOLO networks, and many studies and
> improvements have been made over the years aiming to promote the
> effectiveness of object detection, but there are still difficulties in
> detecting small targets in complex backgrounds. Our analysis of
> remotely sensed images reveals that, despite their small size, ship
> targets exhibit significant color differences from the background.
> Additionally, the self-attention mechanism only involves image
> information in the spatial dimension without the channel dimension.
> According to the abovementioned analyzation, we propose a Difference
> Mask structure to retain useful information, mask irrelevant regions,
> and incorporate an Efficient Channel Attention (ECA) mechanism for
> facilitating interaction among different channels.

1.  **Method**
    ==========

    1.  *Framework overview*

> In this section, we will provide a detailed description of the DMMA
> based YOLOv5 model. The network framework of improved YOLOv5 is
> divided into backbone, neck and head according to their different
> functions. The backbone is responsible for extracting the feature
> representation of the input image. It consists of convolutional layers
> and pooling layers, which gradually reduce the size of the feature
> maps and increase the level of feature abstraction. The neck is of an
> enhanced Feature Pyramid Network structure, which further integrates
> and enhances the feature representation from the backbone, facilitates
> multi-scale feature fusion to speed detection of objects of various
> sizes. The head is responsible for taking the fused features and
> inputting them into the prediction layer for object detection.
> Furtherly, convolutional layers and fully connect layers are adopted
> to generate bounding boxes, class predictions, and confidence scores
> for the detected objects. Ultimately, the head converts the feature
> maps into position and class information of the detected objects. The
> Different Mask Mixed Attention is embedded into the neck structure, as
> shown in Fig.1. DMMA-C3 and DMMA structures are represented in Fig.2.

1.  []{#_Hlk179727524 .anchor}*Model selection and module improvement*

> YOLOv5 can be classified into four models according to its different
> widths and depths, namely YOLOv5s, YOLOv5m, YOLOv5l, and YOLOv5x.
> Among them, YOLOv5s has the smallest model size and fastest detection
> speed, which makes it more flexible for model fine-tuning.
> Additionally, the streamlined structure of YOLOv5s makes it easier to
> adjust and optimize across different environments and datasets.
> Therefore, we choose YOLOv5s to pursue higher accuracy. YOLOv5
> incorporates a CSPDarknet-53 backbone network with an added SPP layer
> to extract input features. The neck structure adapts a combination of
> FPN and PAN to fuse the extracted features. Due to the significant
> impact of the final feature map on the detection results, DMMA-C3 is
> adopted to replace the last three C3 modules in the neck structure.
> The fused feature map is sent to the detection head, and the output
> tensor size after detection is *N\*M\**\[3*\**(4*+*1*+*1)\], where *N*
> and *M* respectively represent the width and height of the tensor, 3
> represents the 3 predicted boxes corresponding to each grid in the
> feature map at the same scale, and the other numbers represent the
> four values of the predicted box offset, the object confidence, and
> the class confidence in sequence. Then, non-maximum suppression is
> used to filter the predicted boxes and obtain the final prediction
> results. The original YOLOv5 network outputs three feature maps of
> different sizes through the C3 modules consisting of multiple
> convolutional layers. These feature maps are directly fed into the
> detection head, where the different channels and regions in the
> feature maps determine the detection results. Therefore, before the
> detection head detecting the feature maps, we replace the original
> last three C3 modules in the neck structure with DMMA-C3, as shown in
> Fig.1. The DMMA-C3 structure is presented in Fig.2(a). Inspired by
> (Srinivas et al., 2021), the original Bottleneck in the C3 module is
> replaced with DMMA to enable global self-attention processing. The
> specific structure of DMMA is illustrated in Fig. 2(b) and consists of
> three main components: window-based multi-head self-attention (W-MSA),
> a difference mask, and efficient channel attention. These components
> work together to acquire global information, extract salient features,
> and enhance the target area, all while minimizing excessive
> computations.
>
> **Fig.1.** DMMA based YOLOv5 Model Structure
>
> （a）DMMA-C3 structure （b）DMMA structure
>
> **Fig. 2.** Architecture of DMMA-C3 (a) An DMMA-C3 module, which
> replaces the original C3 module in the network with the DMMA
> structure. (b) DMMA structural diagram, including W-MSA, Difference
> Mask, and ECA module

1.  *Window-based multi-head self-attention *

> When extracting information from the input image via convolutional
> layers, each convolutional operation can only extract partial
> information from the feature map. On the other hand, the Transformer
> can obtain global information and extract cross-correlation
> information in each operation, making the model more expressive.
> However, this also results in a significant increase of computational
> cost. To reduce computation quantity, we adopt a window-based approach
> to process the image. First, to avoid excessive computational cost,
> the three-dimensional input feature map is divided into multiple
> sub-windows, and then self-attention operations are performed on each
> sub-window. Next, the sub-windows serving as input tensors are sliced
> by expanding the channel dimension with linear transformation, thereby
> obtaining three matrices *Q*, *K*, and *V* for computing
> self-attention (Vaswani, A. et al., 2017.), as well as the *M* matrix
> for constructing the difference mask matrix, as shown in Figure 2(b).
> Among them, the *Q* matrix represents the query vector and is used to
> determine the focus of attention. The *K* matrix provides key
> information and helps determine the correlation with the query. The
> *V* matrix is the value matrix, which contains the actual feature
> information. The matrix *M* is used to calculate the difference
> between the target edge and the background, and its specific
> calculation process is represented in the next subsection. Each matrix
> is labeled as follows: where *H* and *W* represent the height and
> width of the input matrix, and *C* represents the number of channels.
> The computation of the output matrix is as follows:

(1)

> where $d_{k}$ is the number of the matrix *K* columns, and the product
> of the matrix *Q* and the transpose of matrix *K* is divided by to
> prevent excessively large self-attention amplitude. In addition, we
> also set up self-attention mechanisms in the front of the detection
> head to handle smaller input feature maps with reducing the
> computational cost and memory usage.

1.  *Difference Mask structure*

> The self-attention mechanism can effectively capture long-range
> dependencies, which is particularly important for handling small
> objects in complex scenes. In real images, small objects may be
> obscured by complex backgrounds or partially overlapped with other
> objects. The self-attention mechanism helps the network focus on the
> global information surrounding the target, thereby improving the
> extraction and understanding of the target features. This global
> information can complement local features and enhance detection
> performance. However, it can also introduce complex background
> information, potentially distracting from the main focus. Through
> analysis of ship targets in remote sensing images, we found that
> although ship targets are small, their shapes, contours, and colors
> exhibit noticeable differences from the background. Therefore, we have
> proposed a difference mask matrix structure to assist the
> self-attention model in better locating the position of the target by
> computing the differences between the edges of the target and the
> background. This different information can be highlighted for allowing
> the model to better focus on the detected object. The computation
> process of the Difference Mask structure is shown in Fig.3. The light
> blue matrix represents the generated mask matrix after computation.

![](media/image6.jpeg){width="4.183875765529309in"
height="3.83200021872266in"}

**Fig. 3.** Difference Mask structure

> The matrix *M* is used to construct the mask matrix for computing the
> difference values related to the object edge. Each column of the
> matrix *M* is represented as follows: where,*C* represents the number
> of columns in the matrix *M* (also represents the number of channels).
> Then, each column of the matrix *M* is duplicated (*H*×*W*) times to
> construct a matrix *M~i~*. The matrix *M~i~* is subtracted from its
> element-wise transpose matrix to obtain the difference matrix . The
> result of []{#_Hlk177933548 .anchor} represents the difference of
> pixel value between each position and other positions. Particularly at
> the edges of the target, the pixel value differs significantly from
> the one of the backgrounds, effectively emphasizing the difference
> degree at the object edges.
>
> []{#_Hlk177933562 .anchor}In order to fully consider the feature
> information of each channel, we perform a sum operation on the pix
> value of corresponding positions in the matrix *M~i~* to obtain matrix
> . In this way, the features of different channels can be fused, so
> that more abundant feature information can be obtained.
>
> Next, by taking the absolute value of each position in the difference
> matrix and summing them, a normalization matrix[]{#_Hlk177933900
> .anchor} is formed, and expressed as follows:

(2)

The normalization matrix integrates element values of corresponding
positions in *M~i~*, which is adopted to normalize the difference
information in by element-wise division for obtaining the difference
matrix :

(3)

> where $M_{\text{mask}} \in R^{(H \times W) \times (H \times W)}$, the
> key function of normalization is that the difference value at
> positions of the target edge can significantly retain high, while the
> influence of background information can be reduced. Subsequently, we
> apply thresholding to []{#_Hlk177933954 .anchor} for identifying the
> edge. Specifically, an element of []{#_Hlk178248588 .anchor} is
> remarked as 0 related to the object edge only when[]{#_Hlk177933982
> .anchor}; otherwise, remarked as 1, whose definition is as follows:

(4)

> where, []{#_Hlk177934892 .anchor}, 𝛼 is the threshold ratio, and *C*
> is the number of channels. Additionally, we have adopted an
> incremental search strategy to illustrate the effectiveness of this
> method. At the same time, considering that the target sizes and
> distribution locations in different datasets are different, the
> corresponding value of α will also change. Therefore, we have used 1×1
> convolution kernel to adaptively learn the value of *α* and had better
> results. The relevant experiment result is represented in Section 4.
> This adaptive thresholding method can validly extract object from the
> background through indicating the edges of the object.
>
> In the previous steps, the difference mask matrix has undergone a
> series of processes to highlight the object edges. Finally, we perform
> dot multiplication between and *QK^T^* for further enhancing the model
> focus on the target. It can more clearly guide the attention of the
> model to the target region and improve accuracy of the model
> recognition of the target.

*3.5. Effective Channel Attention*

The self-attention mechanism can process images in the spatial dimension
and find the target area, but it does not exert an influence in the
channel dimension. In contrast, the channel attention mechanism can
reassign channel weights within the feature map. The ECA mechanism is a
type of channel attention mechanism, it can make the feature information
more sensitive than the background information, thereby contributing to
extract the range of the target object. Firstly, average pooling is
performed on the input feature map with the dimension of *CHW*, and the
spatial information of the feature map is compressed to obtain an
extracted vector of *C*11, which represents the global feature
information of each channel. A 3x1convolution kernel is applied to this
vector, and the Sigmoid function is used to activate the result of the
convolution operation. Finally, the computed 1-dimensional channel with
the reassigned weights is multiplied by the slices of the input feature
map, so as to realize the information interaction between channels.

The overall calculation for the Difference Mask Mixed Attention (DMMA)
module can be represented as follows:

> (5)
>
> The *M* matrix is first processed for obtaining the difference mask
> matrix, multiplied with *QK^T^*, divided by, activated by using the
> Softmax function, and then multiplied with the matrix *V*. The
> calculated result is multiplied with the ECA result of the input
> feature map. This approach effectively leverages the distinct features
> between small objects and the background, avoids unrelated background
> information and implements information interaction between different
> channels, thus improves the detection validity while avoiding
> excessive computation.
>
> ![](media/image22.png){width="1.3805555555555555in"
> height="1.39375in"}
> ![](media/image23.png){width="1.3818897637795275in"
> height="1.3937007874015748in"}
> ![](media/image24.png){width="1.3818897637795275in"
> height="1.3937007874015748in"}

![](media/image25.jpeg){width="1.3818897637795275in"
height="1.3937007874015748in"}
![](media/image26.png){width="1.3777777777777778in"
height="1.3923611111111112in"}
![](media/image26.png){width="1.3777777777777778in"
height="1.3923611111111112in"}

> **Fig. 4.** Representative pictures in MASATI dataset

1.  **Experiments**
    ===============

    1.  *Datasets and Evaluation Metrics*

> To validate the effectiveness of the algorithm, we performed
> experiments on the MASATI dataset (Gallego, Pertusa, Gil, 2018), which
> includes different marine scenes such as sea, reefs, and ports, as
> well as single or multiple ships under different conditions. The
> dataset contains 7389 images with small targets, and we selected 2087
> images containing ships, randomly choosing 90% as the training set
> (1895 images) and 10% as the test set (192 images). We defined targets
> with a ratio of predicted box to image width and height less than 0.2
> as small targets. Fig.4. lists the representative images in the
> dataset.
>
> The detection performance of the model is evaluated by Average
> Precision (AP) (Goodfellow et al., 2016) and Average Recall (AR),
> where Precision and Recall are defined as follows:

(6)

(7)

> where *TP, FP,* and *FN* respectively represent the number of
> correctly detected ship targets (true positive), false detections
> (false positive), and ship targets that were not detected (false
> negative). *AP* is the average precision based on different recall
> rates, which is the area under the Precision-Recall curve obtained
> from the recall rate and precision.

(8)

> where *P*(*R*) is a function of precision (P) and recall (R). *AR* is
> the average IoU over all recall levels and double times of the area
> enclosed by the Recall-IoU curve.

(9)

> where *R*(O) is a function composed of recall (R) and overlapping
> degree (O). In addition, we use frames per second (FPS) to evaluate
> real-time detection efficiency.

1.  *Implementation Details*

> We used a batch size of 32 images per batch and trained the model in
> 300 epochs. The first 20 epochs of the model training were used as a
> warm-up training. In addition, we employed mosaic (Bochkovskiy et
> al.,2020) data augmentation to increase the batch size and data
> diversity by randomly patching four images into one image and
> enriching the background information of the images. Main of
> experiments and models were implemented in the MindSpore platform (Lei
> Chen et al., 2021) of the Ascend910 NPU. The experimental environment
> is configured in Table 1. In the experiment, the initial learning rate
> of the model was set to 1*e*^-2^, the weight decay parameter was
> 5*e*^-4^, and the momentum was 0.9.

Table 1. Experiment environment

+---------------------+-------------+
| > Project           | > Parameter |
+=====================+=============+
| > NPU               | > Ascend910 |
+---------------------+-------------+
| > RAM               | > 192G      |
+---------------------+-------------+
| > Program           | > pyhon3.7  |
+---------------------+-------------+
| > mindspore version | > 1.7.0     |
+---------------------+-------------+

> *4.3.* *Adaptive validation of difference mask matrix structure*
>
> At the beginning of study, the *α* threshold ratio for the mask matrix
> is manually set, and it varies for different target distributions in
> different datasets. To find the optimal threshold, we conducted tests
> on the MASATI dataset and adopted a stable incremental search
> strategy, trying different values of *α* to find the best efficiency.
> The *α* values started from 0.5 and increased with intervals of 0.5.
> When the threshold was less than 2.5, precision increased with
> increasing threshold. At *α* = 2, the recall and precision reached
> their highest values of 80.8% and 82.9%, respectively. Table 2
> displayed the performance of YOLOv5-DMMA on the MASATI dataset under
> different thresholds, while Fig.5 demonstrated the corresponding
> changes of recall and precision with the variation of *α*. Fig.5
> illustrates the increase in recall and precision with the increase of
> *α* threshold ratio, reaching its peak at *α*=2. The experimental
> results indicate that once an appropriate threshold is found, our
> proposed model performs better. However, manual parameter tuning is
> time consuming and resource consuming. To address this issue, we
> propose an adaptive difference mask matrix method. The specific
> approach involves using a 1x1 convolutional kernel to adaptively learn
> the *α* value and to extract valuable elements from the generated
> difference mask matrix. This adaptive learning of α value achieved a
> precision of 82.6% and a recall of 86.3%. Compared with the peak
> values at *α*=2, precision increased by 1.8%, and recall increased by
> 3.2%. The specific results are shown in Table 2. Additionally, we
> validated the effectiveness of this method in subsequent experiments.
>
> **Fig.5.** The change in the recall and precision of the YOLOv5-DMMA
> algorithm as the α threshold ratio increasing.

Table 2 Performance of YOLOv5-DMMA on the MASATI dataset at different α
threshold ratios.

+-----------------------------------------------+------------+------------+
| > Method                                      | > AP(%)    | > AR(%)    |
+===============================================+============+============+
| > DMMA based YOLOv5 (α = 0.5 )                | > 76       | > 78       |
+-----------------------------------------------+------------+------------+
| > DMMA based YOLOv5 (α = 1.0 )                | > 76.3     | > 78.5     |
+-----------------------------------------------+------------+------------+
| > DMMA based YOLOv5 (α = 1.5)                 | > 76.4     | > 79.5     |
+-----------------------------------------------+------------+------------+
| > DMMA based YOLOv5 (α = 2.0)                 | > **80.8** | > **82.9** |
+-----------------------------------------------+------------+------------+
| > DMMA based YOLOv5 (α = 2.5)                 | > 76.5     | > 78.0     |
+-----------------------------------------------+------------+------------+
| > DMMA based YOLOv5 (1×1convolutional kernel) | > **82.6** | > **86.3** |
+-----------------------------------------------+------------+------------+

*4.4. Ablation study *

> To verify the impact of the proposed modules on small target ship
> detection, we performed ablation experiments as presented in Table 3,
> where the \'√\' marker indicates the corresponding module is added. We
> chose the original YOLOv5s as the baseline, and YOLOv5s-1, YOLOv5s-2,
> and YOLOv5s-3 in the table represent adding W-MSA, ECA, and the
> Difference Mask mechanism, respectively. Experimental results
> illustrated that when using YOLOv5s directly for detection, the AP
> value was 77.5%. By adding W-MSA, the result was improved by 2%, and
> by adding ECA, the result was enhanced by 3.3%. Further adding the
> Difference Mask promoted the AP by 6%. The overall AR increased by
> 6.3%. Fig.6 represents the contrasted results of ground truth, raw
> YOLOv5 model and improved YOLOv5 model. Fig.7 presents the PR curves
> of DMMA based YOLOv5 and YOLOv5s algorithms, respectively. The
> convergence curves of training loss
>
> Table 3. Ablation experiments of different modules on the MASATI
> dataset.

+-------------+---------+---------+-------------------+------------+------------+----------+
| > Model     | > W-MSA | > ECA   | > Difference mask | > AP (%)   | > AR (%)   | > FPS    |
+=============+=========+=========+===================+============+============+==========+
| > YOLOv5s   |         |         |                   | > 77.5     | > 80.0     | > 20     |
+-------------+---------+---------+-------------------+------------+------------+----------+
| > YOLOv5s-1 | > **√** |         |                   | > 79.5     | > 82.4     | > 20     |
+-------------+---------+---------+-------------------+------------+------------+----------+
| > YOLOv5s-2 | > **√** | > **√** |                   | > 80.8     | > 83.9     | > 19     |
+-------------+---------+---------+-------------------+------------+------------+----------+
| > YOLOv5s-3 | > **√** | > **√** | > **√**           | > **82.6** | > **86.3** | > **21** |
+-------------+---------+---------+-------------------+------------+------------+----------+

![](media/image32.png){width="1.1535433070866141in"
height="1.1535433070866141in"}![](media/image33.png){width="1.1535433070866141in"
height="1.1535433070866141in"}![](media/image34.png){width="1.1535433070866141in"
height="1.1535433070866141in"}

![](media/image35.png){width="1.1535433070866141in"
height="1.1535433070866141in"}![](media/image36.png){width="1.1535433070866141in"
height="1.1535433070866141in"}![](media/image37.png){width="1.1535433070866141in"
height="1.1535433070866141in"}![](media/image38.png){width="1.1535433070866141in"
height="1.1535433070866141in"}![](media/image39.png){width="1.1535433070866141in"
height="1.1535433070866141in"}![](media/image40.jpeg){width="1.1535433070866141in"
height="1.1535433070866141in"}

![](media/image41.png){width="1.1535433070866141in"
height="1.1535433070866141in"}![](media/image42.png){width="1.1535433070866141in"
height="1.1535433070866141in"}![](media/image43.png){width="1.1535433070866141in"
height="1.1535433070866141in"}

**Fig.6.** There are 3 sets of images in the figure, and in each set,
the left-to-right sequence is ground truth, raw YOLOv5 model prediction,
and prediction from the model incorporating multiple improvement
techniques

> **Fig.7.** The orange curve represents the PR curve of YOLOv5s, and
> the blue curve represents the PR curve of DMMA based YOLOv5.
>
> **Fig.8.**The orange curve represents the Training loss convergence
> curve of YOLOv5s, and the blue curve represents the Training loss
> convergence curve of DMMA based YOLOv5.
>
> **Fig.9.**The orange curve represents the precision rates of YOLOv5s,
> and the blue curve represents the precision rates curve of DMMA based
> YOLOv5s.

for DMMA based YOLOv5 and YOLOv5 are displayed in Fig.8. The precision
rates of YOLOv5s and DMMA based YOLOv5 algorithms are illustrated in
Fig.9 at different epoch.

*4.5. Comparison of Algorithm Performance *

> To further validate the detection performance of the proposed DMMA
> based YOLOv5 algorithm, we compared it with other algorithms,
> including Faster R-CNN, YOLOv4, and PAG-YOLOv5. PAG-YOLO is a proposed
> model, where attention mechanism and an improved loss function were
> embedded into
>
> Table 4. Comparison of different algorithms on MASATI dataset

+------------------------------------------+------------+------------+----------+
| > Model                                  | > AP(%)    | > AR(%)    | > FPS    |
+==========================================+============+============+==========+
| > []{#_Hlk126665129 .anchor}Faster R-CNN | > 69.5     | > 65.8     | > 11     |
+------------------------------------------+------------+------------+----------+
| > YOLOv4                                 | > 75.9     | > 83.4     | > 19     |
+------------------------------------------+------------+------------+----------+
| > PAG-YOLOv5                             | > 79.2     | > 82.9     | > 20     |
+------------------------------------------+------------+------------+----------+
| > YOLOv5                                 | > 77.5     | > 80.0     | > 20     |
+------------------------------------------+------------+------------+----------+
| > DMMA based YOLOv5                      | > **82.6** | > **86.3** | > **21** |
+------------------------------------------+------------+------------+----------+

> YOLOv5. The comparison results are shown in Table 4.
>
> Additionally, we conducted separate evaluations of our model on the
> VisDrone2019 and HRSC2016 datasets. These two datasets provide diverse
> image samples captured in different scenarios. The VisDrone2019
> dataset is a publicly available dataset for object detection and
> tracking, primarily designed for drone vision tasks in aerial scenes.
> It encompasses high-resolution images from various viewpoints and
> environmental conditions, featuring a variety of object types such as
> pedestrians, vehicles, bicycles, and more. Similarly, the HRSC2016
> dataset (Jadhav et al.,2020) is a publicly available dataset for ship
> detection. Its aim is to facilitate the automatic detection and
> localization of ships in high-resolution remote sensing images. The
> dataset contains high-resolution images from different satellites,
> including a substantial number of ship instances. One characteristic
> of this dataset is that the shape and size of ships vary with
> different shooting angles, adding to the difficulty of object
> detection. The detection results are shown in Tables 5 and 6. Compared
> with the original YOLOv5m model, DMMA based YOLOv5m achieved superior
> performance on both the VisDrone2019 and HRSC2016 datasets, further
> affirming the effectiveness of the proposed Different mask mixed
> attention mechanism.
>
> We conducted tests on YOLOv5 and YOLOv7 algorithms to verify which one
> is more suitable for small object detection, using PyTorch framework,
> Intel Xeon Gold 5320 CPU, NVIDIA RTX A4000 GPU, and Ubuntu 18.04
> system. The results are shown in Table 7. The results indicate that
> the YOLOv5s algorithm has better detection performance for small
> object detection.
>
> Table 5 .Detection Performance of YOLOv5m and DMMA based YOLOv5x on
> VisDrone2019 Dataset

+----------------------+------------+------------+----------+
| > Model              | > AP(%)    | > AR(%)    | > FPS    |
+======================+============+============+==========+
| > YOLOv5m            | > 24.5     | > 12.7     | > 60     |
+----------------------+------------+------------+----------+
| > DMMA based YOLOv5m | > **25.3** | > **13.1** | > **76** |
+----------------------+------------+------------+----------+

> Table 6 .Detection Performance of YOLOv5m and DMMA based YOLOv5x on
> HRSC2016 Dataset

+----------------------+------------+------------+----------+
| > Model              | > AP(%)    | > AR(%)    | > FPS    |
+======================+============+============+==========+
| > YOLOv5m            | > 86.6     | > 91.7     | > 70     |
+----------------------+------------+------------+----------+
| > DMMA based YOLOv5m | > **90.6** | > **95.4** | > **72** |
+----------------------+------------+------------+----------+

> Table 7. Detection performance of YOLOv5 and YOLOv7 based on PyTorch
> framework on MASATI dataset.

+-----------+----------------+-------------+------------+----------+------------+
| > Model   | > Precision(%) | > Recall(%) | > AP (%)   | > AP (%) | > FPS      |
+===========+================+=============+============+==========+============+
| > YOLOv5s | > **78.6**     | > **71.2**  | > **73.3** | > **29** | > **29.4** |
+-----------+----------------+-------------+------------+----------+------------+
| > YOLOv7  | > 73.5         | > 66.8      | > 67.8     | > 26.2   | > 28.9     |
+-----------+----------------+-------------+------------+----------+------------+

**Conclusion**
==============

> This paper proposed the DMMA based YOLO algorithm and performed
> several experiments on YOLOv5, aiming to improve the detection
> accuracy of ship targets in optical remote sensing images. Firstly,
> the windowed multi-head self-attention (W-MSA) was adopted to replace
> the convolutional blocks in the original C3 structure, allowing for
> the processing of global information in the feature map while avoiding
> excessive computations. In the next place, a Difference Mask Mixed
> Attention module is added into W-MSA, and its purpose is to filter out
> irrelevant regions in the feature map. However, due to the
> inconvenience of manually setting the *α* threshold ratio during
> training, a 1x1 convolutional kernel was employed to adaptively learn
> the *α* threshold ratio and to extract feature information from the
> mask matrix. Then an effective channel attention mechanism was
> combined with the aforementioned learning network. Finally, several
> datasets were tested to validate the proposed method. Experimental
> results illustrate that the proposed method has higher accuracy
> compared with other algorithms while maintaining detection speed.

CRediT authorship contribution statement
----------------------------------------

Shuai Yuan: Writing, Analysis of results, Conceptualization, Super-vision.
--------------------------------------------------------------------------

Huize Dou: Writing, Software development and experiments.
---------------------------------------------------------

Jinyu Geng: Writing, Consult relevant literature.
-------------------------------------------------

Fangjun Luan: Writing, Theoretical and technical guidance.
----------------------------------------------------------

Xiaowen Zhang: Writing, Responsible for the experimental framework of the thesis.
---------------------------------------------------------------------------------

> **Declaration of competing interest**
>
> The authors declare that they have no known competing financial
> interests or personal relationships that could have appeared to
> influence the work reported in this paper.
>
> **Data availability**
>
> Data will be made available on request.[[]{#_bookmark25
> .anchor}]{#_bookmark24 .anchor}

References
----------

[[]{#_Hlk180154341 .anchor}]{#_Hlk180154234 .anchor}Girshick, R.,
Donahue, J., Darrell, T., & Malik, J. (2014). Rich feature hierarchies
for accurate object detection and semantic segmentation. In Proceedings
of the IEEE conference on computer vision and pattern recognition (pp.
580-587).

Girshick, R. (2015). Fast r-cnn. arXiv preprint arXiv:1504.08083.

Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. (2016). Faster R-CNN:
Towards real-time object detection with region proposal networks. IEEE
transactions on pattern analysis and machine intelligence, 39(6),
1137-1149.

Redmon, J. (2016). You only look once: Unified, real-time object
detection. In Proceedings of the IEEE conference on computer vision and
pattern recognition. (pp. 779-788).

Van Etten, A. (2018). You only look twice: Rapid multi-scale object
detection in satellite imagery. arXiv preprint arXiv:1805.09512.

Zhongzhen Sun, Xiangguang Leng, Lei,Boli Xiong ,Kefeng Ji , Gangyao
Kuang. (2021). BiFA-YOLO: A novel YOLO-based method for
arbitrary-oriented ship detection in high-resolution SAR images. Remote
Sensing, 13(21), 4209.

Liqiong Chen, Wenxuan Shi, Dexiang Deng. (2021). Improved YOLOv3 based
on attention mechanism for fast and accurate ship detection in optical
remote sensing images. Remote Sensing, 13(4), 660.

Jianming Hu, Xiyang Zhi ,Tianjun Shi, Wei Zhang, Yang Cui , Shenggang
Zhao (2021). PAG-YOLO: A portable attention-guided YOLO network for
small ship detection. Remote Sensing, 13(16), 3059.

Redmon, J., & Farhadi, A. (2017). YOLO9000: better, faster, stronger. In
Proceedings of the IEEE conference on computer vision and pattern
recognition (pp. 7263-7271).

Redmon, J. (2018). Yolov3: An incremental improvement. arXiv preprint
arXiv:1804.02767.

Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). Yolov4: Optimal
speed and accuracy of object detection. arXiv preprint arXiv:2004.10934.

Chien Yao Wang, Alexey Bochkovskiy, HongYuan , Mark Liao. (2023).
YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for
real-time object detectors. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition (pp. 7464-7475).

Mnih, V., Heess, N., & Graves, A. (2014). Recurrent models of visual
attention. Advances in neural information processing systems, 27.

Tao Shen, Tianyi Zhou, Guodong Long, Jing Jiang, Shirui Pan, Chengqi
Zhang. (2018, April). Disan: Directional self-attention network for
rnn/cnn-free language understanding. In Proceedings of the AAAI
conference on artificial intelligence (Vol. 32, No. 1).

Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang
Zhang, Xiaogang Wang, Xiaoou Tang. (2017). Residual attention network
for image classification. In Proceedings of the IEEE conference on
computer vision and pattern recognition (pp. 3156-3164).

Jiasen Lu, Caiming Xiong, Devi Parikh, Richard Socher. (2017). Knowing
when to look: Adaptive attention via a visual sentinel for image
captioning. In Proceedings of the IEEE conference on computer vision and
pattern recognition (pp. 375-383).

Vaswani, A., Shazeer N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
A.N., Kaiser, L. (2017). Attention is all you need. Advances in Neural
Information Processing Systems. (pp. 5998--6008).

Dosovitskiy, A. (2020). An image is worth 16x16 words: Transformers for
image recognition at scale. arXiv preprint arXiv:2010.11929.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen
Lin, Baining Guo. (2021). Swin transformer: Hierarchical vision
transformer using shifted windows. In Proceedings of the IEEE/CVF
international conference on computer vision (pp. 10012-10022).

Qibin Hou, Daquan Zhou, Jiashi Feng. (2021) "Coordinate attention for
efficient mobile network design," in Proc. IEEE/CVF Conf. Computer
Vision and Pattern Recognition, 2021, pp. 13713-13722.

Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, Qinghua
Hu. (2020). ECA-Net: Efficient channel attention for deep convolutional
neural networks. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition (pp. 11534-11542).

Hassanin, M., Anwar, S., Radwan, I., Khan, F. S., & Mian, A. (2024).
Visual attention methods in deep learning: An in-depth survey.
Information Fusion, 108, 102417.

Zixiao Zhang, Xiaoqiang Lu, Guojin Cao, Yuting Yang, Licheng Jiao, Fang
Liu. (2021). ViT-YOLO: Transformer-based YOLO for object detection. In
Proceedings of the IEEE/CVF international conference on computer vision
(pp. 2799-2808).

Shiyi Tang, Shu Zhang, Yini Fang.(2024, May). HIC-YOLOv5: Improved
YOLOv5 for small object detection. In 2024 IEEE International Conference
on Robotics and Automation (ICRA) (pp. 6614-6619). IEEE.

Chun-Lin Ji, Tao Yu, Peng Gao, Fei Wang , Ru-Yue Yuan. (2024). Yolo-tla:
An Efficient and Lightweight Small Object Detection Model based on
YOLOv5. Journal of Real-Time Image Processing, 21(4), 141.

Xingkui Zhu, Shuchang Lyu, Xu Wang, Qi Zhao. (2021). TPH-YOLOv5:
Improved YOLOv5 based on transformer prediction head for object
detection on drone-captured scenarios. In Proceedings of the IEEE/CVF
international conference on computer vision (pp. 2778-2788).

Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). Cbam:
Convolutional block attention module. In Proceedings of the European
conference on computer vision (ECCV) (pp. 3-19).

Srinivas, A., Lin, T. Y., Parmar, N., Shlens, J., Abbeel, P., & Vaswani,
A. (2021). Bottleneck transformers for visual recognition. In
Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition (pp. 16519-16529).

Gallego, A. J., Pertusa, A., & Gil, P. (2018). Automatic ship
classification from optical aerial images with convolutional neural
networks. Remote Sensing, 10(4), 511.

Goodfellow, I., Bengio, Y., Courville, A., & Bengio, Y. (2016). Deep
learning (Vol. 1). MIT press Cambridge.

Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). Yolov4: Optimal
speed and accuracy of object detection. arXiv preprint arXiv:2004.10934.

Lei Chen. (2021). Deep learning and practice with mindspore. Springer
Nature.

Zhihao Tong, Ning Du, Xiaobo Song, Xiaoli Wang. (2021, November). Study
on mindspore deep learning framework. In 2021 17th International
Conference on Computational Intelligence and Security (CIS) (pp.
183-186). IEEE.

Jadhav, A., Mukherjee, P., Kaushik, V., & Lall, B. (2020, February).
Aerial multi-object tracking by detection using deep association
networks. In 2020 National Conference on Communications (NCC) (pp. 1-6).
IEEE

[^1]: ^ ^This work was supported by National Natural Science Foundation
    > of China (62073227) and Liaoning Provincial Science and Technology
    > Department Foundation (2023JH2/101300212).
    >
    > \*Corresponding author.
    >
    > *E-mail address:* reidyuan@163.com (Shuai Yuan).
