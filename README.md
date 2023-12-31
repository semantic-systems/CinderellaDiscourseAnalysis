# CinderellaDiscourseAnalysis
a prototype for discourse analysis on 500 versions of cinderella, and 497 more to go...
![and 497 to go...](cinderella.png)
_download the .html file to interact with Cinderellas🧚🏻‍♀️_

Each node represents a sentence from different versions of Cinderella, temporally ordered. 
Different versions are indicated by the color. Edges represent similarities between each sentences across versions.

This representation aims to discover how stories of Cinderella are told differently.  

We assume that there is something as a "core cinderella", which is represented as 
- a sequence of events that is perceived as the essence of a piece of narrative 
- or, the invariance in the variances of the same story (seeing that as finding class properties with observations from instances)

# Resources
1. Gutenberg Project
2. Narrative representation models
   1. [Pretrained framenet transformer](https://github.com/chanind/frame-semantic-transformer)
   2. [UTC](https://github.com/PaddlePaddle/PaddleNLP/blob/466872bd72a11c6d548d530c50dbda1ac6354dfe/paddlenlp/transformers/ernie/modeling.py#L1286)

# Installation
### Requirements
- frame-semantic-transformer
- gutenberg 
- clean-text 
- paddlenlp 
- fastcoref

Note: In order to install gutenberg, you will need a local version of `berkeley-db` installed in the system. 
For more information regarding platform-specific installation, please visit the gutenberg project [homepage](https://pypi.org/project/Gutenberg/).

For MacOS, here are the steps:
```
brew install berkeley-db4
BERKELEYDB_DIR=$(brew --prefix berkeley-db@4) pip install bsddb3
pip install -r requirements.txt
```


# Idea
As a fast proof-of-concept, the goal here is to see how different versions of Cinderella differ from each other on a discourse level (how a story is told), where stories (or events) are represented as frames. Generative models will be used to generate frames. 

1. Discourse analysis: Temporality of events and Cinderella ...  
2. Summarization through core event mining of various narratives
3. Media biases study


[//]: # (If meaningful findings are discovered, then it makes sense to go beyond FrameNet -> proper event extraction)

[//]: # (Another potential of this is to use a CLIP-like architecture with contrastive pre-training to map both raw text and KGs onto the same space to allow unsupervised text-KG conversion or another way around. )
