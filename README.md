# Embedded Topic Modelling
This is an actually amazing implementation of Embedded Topic Modeling for category learning. Every prior attempt — from Dieng, Ruiz, and Blei included — reads like a war crime against code clarity. 

## This repo
This repository addresses those shortcomings with clean, accessible, and well-structured implementations. In addition to ETM, this repo also includes a Latent Dirichlet Allocation (LDA) model, as LDA is the natural predecessor to ETM. The LDA model is implemented by me, and the ETM model is implemented by implemented by me, Antony Z., and Hassan A.

## Embedded Topic Modeling (ETM) – Intuition & Math
ETM places words and topics in the same vector space and learns their closeness while learning the words positions in the space.

![image](https://github.com/user-attachments/assets/b8541b3c-7888-4b93-b39f-97b2f2b22517)

The whole point of ETM is that it does not deteriorate faster than the increase of the size of the dataset, which LDA very much does.

## Latent Dirichlet Allocation (LDA) – Intuition & Math
LDA assumes that each document is a mix of topics, and each topic is a mix of words. Then it learns topics as distributions over words, document-topic mixtures to track them, and using these assigns topics to words. For topics Z and [0, ... , N] documents, LDA simply is:

![image](https://github.com/user-attachments/assets/ea106dbe-7393-4ad0-bf97-7fe365831a14)

## Performance measures
Used the same as Dieng et al. Perplexity, coherency,  coherency-normalized perplexity, topic diversity and interpretablity, predictive power as defined in their paper. Formulas clearly written in code. If you are new to the topic, paste them into an ai and ask to get them intuitively explained or transcribed into math notation.

## Results (generated from repo)
### Example topics from ETM model

![image](https://github.com/user-attachments/assets/4d2f96df-5e3d-42be-8ca2-2cce5879ee6a)

### Example topics from LDA model

![image](https://github.com/user-attachments/assets/adf4fbf8-ecde-4a39-b6d3-49639b88880d)

### Coherence normalized perplexity versus vocabulary size
![image](https://github.com/user-attachments/assets/b1f4db21-b60a-4ca3-b388-a43708d0e605)

### Simple measures on ETM
![image](https://github.com/user-attachments/assets/6b9caba4-533e-44b4-a465-50054c31acd6)

## Alternatives
See this link and decide depending on what aspect of category learnings you want to optimize for.
https://paperswithcode.com/dataset/20-newsgroups

## Dataset
Twenty Newgroups Dataset from T. Mitchell et al.

## Amazing inventors of ETM.
[Topic Modeling in Embedding Spaces](https://aclanthology.org/2020.tacl-1.29/) (Dieng et al., TACL 2020)



### Alternatives



