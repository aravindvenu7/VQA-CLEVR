# VQA-CLEVR

The dataset provided is a modification of the CLEVR dataset. It consists of a collection of synthesized images and a set of questions associated with each image. Your task is to predict the answers to these questions. It has about 15000 images and 135000 questions and answers for training. Most images have about 10 questions associated with them, however, some images don’t have any questions associated with them.The images contain simple 3D objects with each object having one of the 96 property profiles obtained by picking one choice each from the following four types:
Shape: Sphere, Cube, Cylinder
Size: Large, Small
Material: (Matte) Rubber, (Shiny) Metal
Color: Gray, Cyan, Blue, Purple, Brown… (8 colors)

The questions test the understanding regarding the various type of relationships between these objects. The relationships include the understanding of the position relative to the other objects i.e which of two objects lies to the left/right of the other or which of the two objects lies in front of/behind the other. 


For addressing the task at hand,we have trained the dataset on three different models and chosen the model with the best relative accuracy score. Model architectures used are:
1.  CNN-LSTM Model
2.  Stacked Attention Model


![image1](https://user-images.githubusercontent.com/28951885/52520377-4456de80-2c8f-11e9-9851-e71919dd75fe.jpg)

-Pretrained 50-d GloVe embeddings have been used for the embedding layer of LSTM
-CNN head outputs 16 feature maps of size 8x8
-The question vector and flatten image vector are concatenated and fed into a series of dense layers





CNN –LSTM Model with Stacked Attention


![unnamed](https://user-images.githubusercontent.com/28951885/52520378-4751cf00-2c8f-11e9-9f21-ae4e3b5c3181.png)

This model is a Keras implementation of the model proposed in the paper : 
Stacked Attention Networks for Image Question Answering.
Link to the paper :
https://www.cvfoundation.org/openaccess/content_cvpr_2016/papers/Yang_Stacked_Attention_Networks_CVPR_2016_paper.pdf
As in the paper,we have used 2 attention layers to capture region-wise relations between the objects in the image and objects referred to in the question


![attn24](https://user-images.githubusercontent.com/28951885/52520375-3acd7680-2c8f-11e9-9adb-10de07378aef.jpg)

To find the optimum parameters We tried to implement the Grid Search on the various the various hyper-parameters like number of dense units ,epochs,learning rate,etc . however later we came to know that one can’t apply gridsearch on multi-input models like the ones being used to solve the given problem statement. Thus various parameters like epochs,learning rate were manually tested by running various models and analysing the results. It was found that the optimizer Adadelta() at the default parameters preformed the best among the various other parameters like Adam, SGD , etc. 


