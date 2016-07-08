---
layout: post
title: Hindi-Urdu Transliteration 
permalink: /house-united/
category: gsoc
date: 2016-07-05
tags:
- libindic
- gsoc 2016
- machine transliteration
- machine learning
- Indian languages
- structured perceptron
- beam search
- viterbi search
---

Hindi and Urdu transliteration has always been treated special among Indic script transliterations. It has received a lot of attention from the NLP research community of South Asia. It has been seen to break the barrier that makes the two look different, although they are facets of the same language. 

### Hindi-Urdu: Same or Different 

Hindi and Urdu are spoken primarily in northern India and Pakistan and together constitute the third largest language spoken in the world. They are two standardized registers of what has been called the Hindustani language, which belong to the Indo-Aryan language family. While Hindi and Urdu are different languages officially, they are not even different dialects or sub-dialects in a linguistic sense; rather, they are different literary styles based on the same linguistically defined sub-dialect. At the colloquial level, Hindi and Urdu are nearly identical, both in terms of core vocabulary and grammar. However, at formal and literary levels, vocabulary differences begin to loom much larger (Hindi drawing its higher lexicon from Sanskrit and Urdu from Persian and Arabic) to the point where the two styles/languages become mutually unintelligible. In written form, not only the vocabulary but the way Urdu and Hindi are written makes one believe that they are two separate languages. They are written in separate orthographies, Hindi being written in Devanagari, and Urdu in a modified Perso-Arabic script. Given these differences in script and vocabulary, Hindi and Urdu are socially and even officially considered two separate languages. Transliteration can remove these differences between Hindi and Urdu texts.

### Challenges in Transliteration

 * **Urdu to Hindi:** The conversion of Urdu text into Hindi poses two problems (a) inference of missing short vowels and, (b) disambiguation of some Urdu letters like 'ا'(_maps to a or A_), 'و'(_maps to va, o, u, O or U_) and 'ی'(_maps to ya, i or I_). In Urdu writing, short vowels are hardly represented, even though the script has provision for their representation. They are dropped due to the fact that readers can infer them easily in the context. Inferring these short vowels is the major bottleneck in Urdu to Hindi transliteration. For example, ambiguities in few Urdu words due to missing short vowels are given in table below: 

<p>
<center>
<table class="table-fill">
<th><b>Urdu Word</b></th> <th><b>Ambiguous Meaning</b></th>
<tr>
<td>ہوا</td> <td>ہَوا (air) or ہُوا (happen)</td>
</tr>
<tr>
<td>کیا</td> <td>کیا (what) or کِیا (done)</td>
</tr>
<tr>
<td>اچھا</td> <td>اچھا (good) or اِچھا (wish)</td>
</tr>
<tr>
<td>دیا</td> <td>دَیا (sympathy) دِیا (give)</td>
</tr>
<tr>
<td>سر</td> <td>سِر (head) or سُر (tone)</td>
</tr>
</table>
</center>
</p>

 * **Hindi to Urdu:** The major challenge that one faces in transliterating Hindi text to Urdu is the ambiguity in a number of Hindi letters. In Hindi script around 5 letters have ambiguous mappings in Urdu. The ambiguity can be attributed to the Perso-Arabic borrowings in Urdu. The letters that these ambiguous Hindi letters are mapped to have their origin in Persian and Arabic phonology and are not native to Hindi-Urdu. These ambiguous letters with their mappings are listed in Table below.

<p>
<center>
<table class="table-fill">
<th><b>Letter</b></th> <th><b>Mappings</b></th>
<tr>
<td>अ</td>         <td>ع, ا</td>
</tr>
<tr>
<td>त</td>           <td>ت, ط</td>
</tr>
<tr>
<td>स</td>          <td>س, ص, ث</td>
</tr>
<tr>
<td>ह</td>           <td>ح, ہ</td>
</tr>
<tr>
<td>ज़</td>           <td>ذ, ز, ض, ژ, ظ</td>
</tr>
</table>
</center>
</p>

### Training and Testing

For data extraction and learning the model papameters, I followed the same pipeline as described in the [Roman-Indic Transliteration](/rom-ind/) post. I extracted a total of _54,035_ translation pairs from the ILCI parallel corpus and _66,668_ pairs from Indo-wordnet synset mappings. The transliteration accuracy Hindi to Urdu system is _98.03%_ for a test-set of 10,000 words, where as the transliteration accuracy of Urdu to Hindi system is _88.03%_ for the same test-set.

```python
from indictrans import transliterator
tr = transliterator(source='hin', target='urd')

hindi = u"""भोपाल के कई जिलों में लगातार हो रही बारिश के कारण बाढ़ के हालात बने 
हुए हैं। सतना, रीवा, नरसिंहपुर, दमोह, सागर सहित कई जिले बुरी तरह से 
प्रभावित हुए हैं। सतना से 4 किलो मीटर दूर माधवगढ़ के एक शासकीय 
स्कूल में दो दिन से चार टीचर और कुछ लोग बाढ़ के कारण फंसे हुए थे। 
पानी धीरे-धीरे स्कूल के अंदर तक आ चुका था, बचने की उम्मीद कम होती 
जा रही थी ऐसे में शुक्रवार सुबह आर्मी के जवानों ने उन्हें नया जीवन दिया। 
और कई घंटों के रेस्क्यू ऑपरेशन के बाद ऐसे सुरक्षित निकाल लिया गया."""

print(tr.transform(hindi))
بھوپال کے کئی ضلعوں میں لگاتار ہو رہی بارش کے کارن باڑھ کے حالات بنے 
ہوئے ہیں۔ ستنا، ریوا، نرسنہپر، دموہ، ساگر سہت کئی ضلعے بری طرح سے 
پربھاوت ہوئے ہیں۔ ستنا سے ۴ کیلو میٹر دور مادھوگڑھ کے ایک شاسکیہ 
سکول میں دو دن سے چار ٹیچر اور کچھ لوگ باڑھ کے کارن پھنسے ہوئے تھے۔ 
پانی دھیرے-دھیرے سکول کے اندر تک آ چکا تھا، بچنے کی امید کم ہوتی 
جا رہی تھی ایسے میں شکروار صبح آرمی کے جوانوں نے انہیں نیا جیون دیا۔ 
اور کئی گھنٹوں کے ریسکیو آپریشن کے بعد ایسے سرکشت نکال لیا گیا۔

```

### Possible improvements in Urdu to Hindi Transliteration

```python
from indictrans import transliterator
tr = transliterator(source='urd', target='hin', decode='beamsearch')

print('\t'.join(tr.transform(u'ہوا', k_best=3)))
हुआ  ह्व्वा हौआ

print('\t'.join(tr.transform(u'کیا', k_best=3)))
किया  क्या  कया

print('\t'.join(tr.transform(u'اچھا', k_best=3)))
अच्छा इच्छा अछा

print('\t'.join(tr.transform(u'دیا', k_best=3)))
दिया  दया  दीया

print('\t'.join(tr.transform(u'سر', k_best=3)))
सर  सुर  सिर
```

As discussed above, Urdu to Hindi transliteration accuracy is quite low compared to Hindi to Urdu, which is mostly due to missing short vowels in Urdu text, thus generating ambiguous words. If we consider top 2 to 3 transliterations of Urdu, the system accuracy is approximately same as Hindi to Urdu. Thus, to resolve word ambiguity, we can perform sentence-level decoding on the k-best transliterations from the perceptron model. We can use a noisy channel model and exact Viterbi search to find the most likely Hindi sentences. The noisy-channel model can be formally defined as follows:

<center>
<pre lang="latex">
\textbf{h}^{*} &= \arg\max \enspace p(h) \times p(h|u) %\nonumber
</pre>
</center>

<span lang="latex">p(h)</span> is the language model score which gives a prior distribution over the most likely sentences in Hindi and <span lang="latex">p(h|u)</span> is the perceptron score which indicates how likely the Hindi sentence <span lang="latex">\textit{h}</span> is a word by word transliteration of the Urdu sentence  <span lang="latex">\textit{u}</span>. We can assign uniform probabilities to all the transliteration options and thus redefining the model without  <span lang="latex">p(h|u)</span> as:

<center>
<pre lang="latex">
\textbf{h}^{*} &= \arg\max \enspace p(h) %\nonumber
</pre>
</center>

### Bridging the Script and Lexical Barrier

In Computational Linguistics, Hindi and Urdu are not viewed as a monolithic entity and have received separate attention with respect to their text processing. From part-of-speech tagging to machine translation, models are separately trained for both Hindi and Urdu despite the fact that they represent the same language. The reasons mainly are their divergent literary vocabularies and separate orthographies, and probably also their political status and the social perception that they are two separate languages. Hindi-Urdu transliteration proposes a simple but efficient approach to bridge the lexical and orthographic differences between Hindi and Urdu texts. With respect to text processing, addressing the differences between the Hindi and Urdu texts would be beneficial in the following ways: (a) instead of training separate models, their individual resources can be augmented to train single, unified models for better generalization, and (b) their individual text processing applications can be used interchangeably under varied resource conditions.
