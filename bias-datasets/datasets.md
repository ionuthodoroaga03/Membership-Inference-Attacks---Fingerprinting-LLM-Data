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

21. SART is a sentiment analysis dataset built from Romanian-language tweets annotated into three classes: positive, negative, and neutral. It contains 3,900 instances, evenly distributed with 1,300 tweets per class, and is commonly used as a benchmark for sentiment analysis models in Romanian.
Link for 21. and 22. : https://github.com/Alegzandra/KES-2023/tree/main/datasets . Reference: Alexandra Ciobotaru & Liviu Dinu (2023):  SART & COVIDSentiRo: Datasets for Sentiment Analysis Applied to Analyzing COVID-19 Vaccination Perception in Romanian Tweets
22. COVIDSentiRo is a larger corpus of Romanian tweets related to COVID-19 vaccination, collected between January 2021 and February 2022. The dataset includes 19,320 tweets labeled with the same three sentiment classes and is used to analyze public perception of vaccination on social media.