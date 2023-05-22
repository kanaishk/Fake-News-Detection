Context
As we all know, Fake-News has become the centre of attraction worldwide because of its hazardous impact on our society. One of the recent example is spread of Fake-news related to Covid-19 cure, precautions, and symptoms and you must be understood by now, how dangerous this bogus information could be. Distorted piece of information propagated at the times of election for achieving political agenda is not hidden from anyone.

Fake news is quickly becoming an epidemic, and it alarms and angers me how often and how rapidly totally fabricated stories circulate.
Why? In the first place, the deceptive effect: the fact that if a lie is repeated enough times, you’ll begin to believe it’s true.

You understand by now that fake news and other types of false information can take on various appearances. They can likewise have significant effects, because information shapes our world view: we make important decisions based on information. We form an idea about people or a situation by obtaining information. So if the information we saw on the Web is invented, false, exaggerated or distorted, we won’t make good decisions.

Hence, Its in dire need to do something about it and It's a Big Data problem, where data scientist can contribute from their end to fight against Fake-News.

Content
Although, fighting against fake-News is a big data problem but I have created this small dataset having approx. 10,000 piece of news article and meta-data scraped through approx. 600 web-pages of Politifact website to analyse it using data science skills and get some insights of how can we stop spread of misinformation at broader aspect and what approach will give us better accuracy to achieve the same.

This dataset is having 6 attributes among which News_Headline is the most important to us in order to classify news as FALSE or TRUE. As you notice the Label attribute clearly, there are 6 classes specified in it. So, it's totally up-to you whether you want to use my dataset for multi-class classification or convert these class labels into FALSE or TRUE and then, perform binary classification. Although, for your convenience, I will write a notebook on how to convert this dataset from multi-class to binary-class. To deal with the text data, you need to have good hands on practice on NLP & Data-Mining concepts.

News_Headline - contains piece of information that has to be analysed.

Link_Of_News - contains url of News Headlines specified in very first column.

Source - this column contains author names who has posted the information on facebook, instagram, twitter or any other social-media platform.

Stated_On - This column contains date when the information is posted by the authors on different social-media platforms.

Date - This column contains date when this piece of information is analysed by politifact team of fact-checkers in order to labelize as FAKE or REAL.

Label - This column contains 5 class labels : True, Mostly-True, Half-True, Barely-True, False, Pants on Fire.

So, you can either perform multi-class classification on it or convert Mostly-True, Half-True, Barely-True as True and drop Pants on Fire and perform Binary-class classification.

Acknowledgements
A very Big thanks to fact-checking team of Politifact.com website as they provide with correct labels by working hard manually. So that we data science people can take advantage to train our models on such labels and make better models.
These are some research papers that will help you to get start with the project and clear your fundamentals.

Big Data and quality data for fake news and misinformation detection by Fatemeh Torabi Asr, Maite Taboada

Automatic deception detection: Methods for finding fake news by Nadia K. Conroy Victoria L. Rubin Yimin Chen

Inspiration
I want to see which approach can solve this problem of combating Fake-News with greater accuracy.