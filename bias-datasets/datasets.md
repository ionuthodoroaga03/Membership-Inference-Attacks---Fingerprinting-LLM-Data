1\. GlobalVoices (OPUS) - **sentiment, translation bias, political bias, stereotype classification (news framing of ethnic groups)**



* real news text, multi-domain, parallel across many language pairs



[https://opus.nlpl.eu/legacy/GlobalVoices.php](https://opus.nlpl.eu/legacy/GlobalVoices.php)

[https://huggingface.co/datasets/sentence-transformers/parallel-sentences-global-voices](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-global-voices)





2\. OpenSubtitles (OPUS) - **sentiment, stereotype classification, translation bias**



* dialog-heavy, strong presence of informal ethnic stereotypes and slurs.



[https://opus.nlpl.eu/OpenSubtitles/corpus/version/OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles/corpus/version/OpenSubtitles)

[https://huggingface.co/datasets/Helsinki-NLP/open\_subtitles](https://huggingface.co/datasets/Helsinki-NLP/open_subtitles)





3\. TED Talks (TED2013 / WIT3 / TED parallel corpora) - **translation bias, sentiment (especially in identity / discrimination talks), moderate for political bias**



[https://opus.nlpl.eu/TED2013/corpus/version/TED2013](https://opus.nlpl.eu/TED2013/corpus/version/TED2013)

[https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus](https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus)





4\. Wikipedia (multilingual dumps) -  **political bias (framing of ethnic groups, conflicts, migration), some sentiment / stereotype analysis**



[**https://www.clarin.eu/resource-families/parallel-corpora**](https://www.clarin.eu/resource-families/parallel-corpora)





5\. OSCAR (web-scale multilingual corpus) **- large-scale sentiment and political bias analysis across many languages; background corpora for training / probing embeddings**



[https://huggingface.co/datasets/oscar-corpus/OSCAR-2201](https://huggingface.co/datasets/oscar-corpus/OSCAR-2201)

[https://oscar-project.github.io/documentation/versions/oscar-2019/](https://oscar-project.github.io/documentation/versions/oscar-2019/)



6\. Multilingual Bias Evaluation - **stereotype classification, controlled sentiment bias evaluation, cross-lingual comparison**



[https://github.com/microsoft/MultilingualBiasEvaluation](https://github.com/microsoft/MultilingualBiasEvaluation)



7\. CrowS-Pairs - **tereotype classification, sentiment skew around ethnic groups, multi-language comparison**



[https://github.com/nyu-mll/crows-pairs](https://github.com/nyu-mll/crows-pairs)

[https://gitlab.inria.fr/corpus4ethics/multilingualcrowspairs](https://gitlab.inria.fr/corpus4ethics/multilingualcrowspairs)

[https://github.com/jerryspan/Dutch-CrowS-Pairs](https://github.com/jerryspan/Dutch-CrowS-Pairs)



8\. StereoSet + translated variants (e.g., K-StereoSet) - **stereotype classification, quantitative bias scores, some sentiment analysis**



[https://github.com/moinnadeem/StereoSet](https://github.com/moinnadeem/StereoSet)

[https://huggingface.co/datasets/McGill-NLP/stereoset](https://huggingface.co/datasets/McGill-NLP/stereoset)

[https://github.com/JongyoonSong/K-StereoSet](https://github.com/JongyoonSong/K-StereoSet)



9\. MUSE - **For intrinsic bias metrics (WEAT/SEAT-style), cross-lingual comparisons of ethnic associations**



[https://github.com/facebookresearch/MUSE](https://github.com/facebookresearch/MUSE)

[https://ai.meta.com/tools/muse/](https://ai.meta.com/tools/muse/)



10\. fastText multilingual vectors - **similar to MUSE, can be aligned across languages, broad coverage (157 languages)**



[https://fasttext.cc/docs/en/crawl-vectors.html](https://fasttext.cc/docs/en/crawl-vectors.html)

[https://github.com/babylonhealth/fastText\_multilingual](https://github.com/babylonhealth/fastText_multilingual)





11\. Tatoeba Challenge - **strong for translation-bias evaluation across many languages; less useful for sentiment per se unless you curate an ethnic subset**





https://github.com/Helsinki-NLP/Tatoeba-Challenge

https://huggingface.co/datasets/Helsinki-NLP/tatoeba\_mt

https://www.kaggle.com/datasets/alvations/tatoeba



12\. FLORES-200 - **Gold standard for translation bias across 200+ languages (parallel, human-checked)**



https://huggingface.co/datasets/Muennighoff/flores200

https://github.com/facebookresearch/flores/blob/main/flores200/README.md

https://www.emergentmind.com/topics/flores-200-benchmark-dataset



13. BBQ
[https://github.com/nyu-mll/BBQ](https://github.com/nyu-mll/BBQ)

14. EUROPARL
[https://www.statmt.org/europarl/](https://www.statmt.org/europarl/)

15. FACTOID
[https://github.com/caisa-lab/FACTOID-dataset](https://github.com/caisa-lab/FACTOID-dataset)

16. HATECHECK
[https://github.com/paul-rottger/hatecheck-data](https://github.com/paul-rottger/hatecheck-data)

17. HONEST
[https://github.com/MilaNLProc/honest](https://github.com/MilaNLProc/honest)

18. Jigsaw Multilingual
[https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/data](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/data)

19. NEWB
[https://github.com/JerryWeiAI/NewB](https://github.com/JerryWeiAI/NewB)

20. REDDIT BIAS
[https://github.com/umanlp/RedditBias](https://github.com/umanlp/RedditBias)


NLP datasets (Romanian included)
21. SART is a sentiment analysis dataset built from Romanian-language tweets annotated into three classes: positive, negative, and neutral. It contains 3,900 instances, evenly distributed with 1,300 tweets per class, and is commonly used as a benchmark for sentiment analysis models in Romanian.
Link for 21. and 22. : https://github.com/Alegzandra/KES-2023/tree/main/datasets . Reference: Alexandra Ciobotaru & Liviu Dinu (2023):  SART & COVIDSentiRo: Datasets for Sentiment Analysis Applied to Analyzing COVID-19 Vaccination Perception in Romanian Tweets
22. COVIDSentiRo is a larger corpus of Romanian tweets related to COVID-19 vaccination, collected between January 2021 and February 2022. The dataset includes 19,320 tweets labeled with the same three sentiment classes and is used to analyze public perception of vaccination on social media.
23. Ro-Offense: A dataset of comments annotated for offensive language, specifically targeting groups like the Roma community, welfare recipients, and Hungarians. 
https://huggingface.co/datasets/readerbench/ro-offense-sequences
It contains 4800 labeled comments (used for offensive language detection)
{'train': (4000, 4), 'validation': (400, 4), 'test': (400, 4)}
{'id': 32362, 'text': 'SUNTETI NISTE ISTERICI DE FIECARE DATA CAND CASTICA STEAUA NUMAI BLAT STITI SI FACETI SPAM\n\nPANARAMELORR !', 'offensive_substrings': "['ISTERICI', 'PANARAMELORR']", 'offensive_sequences': '[(14, 22), (92, 104)]'}
24. News-Ro-Offense: Andreea Cojocaru, Andrei Paraschiv, Mihai Dascalu (2022): News-RO-Offense - A Romanian Offensive Language Dataset and Baseline Models Centered on News Article Comments
Corpus statistics
Class	Label	# examples	Percentage
Non-offensive	0	2682	66.19%
Targeted insult	1	777	19.18%
Racist	2	252	6.22%
Homophobic	3	186	4.59%
Sexist	4	155	3.82%
TOTAL		4052
https://github.com/readerbench/news-ro-offense
25. Diana Constantina Hoefels, Çağrı Çöltekin, Irina Diana Mădroane (2022): CoRoSeOf - An Annotated Corpus of Romanian Sexist and Offensive Tweets
This paper introduces CoRoSeOf, a large corpus of Romanian social media manually annotated for sexist and offensive language. This dataset is used for sexism detection. The resulting corpus contains 39,245 tweets, annotated by multiple annotators (with an agreement rate of Fleiss’κ= 0.45), following the sexist label set of a recent study. The automatic sexism detection yields scores similar to some of the earlier studies (macro averaged F1 score of 83.07% on binary classification task). The corpus is released with a permissive license.
https://github.com/DianaHoefels/CoRoSeOf
26. LaRoSeDa -> A Large Romanian Sentiment Dataset, reference: Andreea Cojocaru, Andrei Paraschiv, Mihai Dascalu (2021) : Clustering Word Embeddings with Self-Organizing Maps. Application on LaRoSeDa – A Large Romanian Sentiment Data Set
LaRoSeDa - A Large and Romanian Sentiment Data Set. LaRoSeDa contains 15,000 reviews written in Romanian, of which 7,500 are positive and 7,500 negative. The samples have one of four star ratings: 1 or 2 - for reviews that can be considered of negative polarity, and 4 or 5 for the positive ones. The 15,000 samples featured in the corpus and labelled with the star rating, are splitted in a train and test subsets, with 12,000 and 3,000 samples in each subset.
https://huggingface.co/datasets/universityofbucharest/laroseda
{
    "index": "9675",
    "title": "Nu recomand",
    "content": "probleme cu localizarea, mari...",
    "starRating": 1,
}
27. XED: 
https://github.com/Helsinki-NLP/XED
Number of annotations:	24164 + 9384 neutral
Number of unique data points:	17530 + 6420 neutral
Number of emotions:	8 (+pos, neg, neu)
Number of annotators:	108 (63 active)
This is the XED dataset. The dataset consists of emotion annotated movie subtitles from OPUS. It uses Plutchik's 8 core emotions to annotate. The data is multilabel. The original annotations have been sourced for mainly English and Finnish, with the rest created using annotation projection to aligned subtitles in 41 additional languages, with 31 languages included in the final dataset (more than 950 lines of annotated subtitle lines). The dataset is an ongoing project with forthcoming additions such as machine translated datasets. 
Paper: Öhman, E., Pàmies, M., Kajava, K. and Tiedemann, J., 2020. XED: A Multilingual Dataset for Sentiment Analysis and Emotion Detection. In Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020)



Vision Datasets
28. MAXM: https://github.com/google-research-datasets/maxm

MaXM, a test-only multi-lingual visual question answering benchmark in 7 diverse languages: English (en), French (fr), Hindi (hi), Hebrew (iw), Romanian (ro), Thai (th), and Chinese (zh). 


By Changpinyo et al. (2023) MaXM: Towards Multilingual Visual Question Answering
dataset                 str: dataset name
version                 str: dataset version
split                   str: language ID
annotations             List of image-question-answers triplets, each of which is
-- image_id             str: image ID
-- image_url            str: image URL
-- qa_pairs             List of question-answer pairs, each of which is
---- question_id        str: question ID
---- question           str: raw question
---- answers            List of str: ground-truth answers
---- processed_answers  List of str: processed ground-truth answers. 16 tokenized answers.
---- is_collection      bool: "true" if the question is of the "Collection" type; "false" otherwise..
The datasets are based on the images and the captions from the Crossmodal-3600 dataset (XM3600).
CrossModal dataset: A Massively Multilingual Multimodal Evaluation Dataset
https://google.github.io/crossmodal-3600/
29. (Data not out yet, only paper, but interesting idea): CultureVLM: Characterizing and Improving Cultural Understanding of Vision-Language Models for over 100 Countries; by Liu et al. (2025)
Abstract: "Vision-language models (VLMs) have advanced human-AI interaction but struggle with cultural understanding, often misinterpreting symbols, gestures, and artifacts due to biases in predominantly Western-centric training data.
In this paper, we construct CultureVerse, a large-scale multimodal benchmark covering 19, 682 cultural concepts, 188countries/regions, 15 cultural concepts, and 3 question types, with the aim of characterizing and improving VLMs’ multicultural understanding capabilities. Then, we propose CultureVLM, a series of VLMs fine-tuned on our dataset to achieve significant performance improvement in cultural understanding. Our evaluation of 16 models reveals significant disparities, with a stronger performance in Western concepts and weaker results in African and Asian contexts. Fine-tuning on our CultureVerse enhances cultural perception, demonstrating cross-cultural, cross-continent, and cross-dataset generalization without sacrificing performance on models’ general VLM benchmarks. We further present insights on cultural generalization and forgetting. We hope that this work could lay the foundation for more equitable and culturally aware multimodal AI systems."
--> See CultureVLM.png photo.