# CinderellaDiscourseAnalysis
a prototype for discourse analysis on 500 versions of cinderella 

# resources
1. Gutenberg Project
2. pretrained framenet transformer https://github.com/chanind/frame-semantic-transformer 

# Idea
As a fast proof-of-concept, the goal here is to see how different versions of Cinderella differ from each other on a discourse level (how a story is told), where stories (or events) are represented as frames. Generative models will be used to generate frames. 

If meaningful findings are discovered, then it makes sense to go beyond FrameNet -> proper event extraction
Another potential of this is to use a CLIP-like architecture with contrastive pre-training to map both raw text and KGs onto the same space to allow unsupervised text-KG conversion or another way around. 
