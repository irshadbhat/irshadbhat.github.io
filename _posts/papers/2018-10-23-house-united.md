---
layout: page
title: Abstract
permalink: /house-united/
category: subpost
date: 2018-10-23

---

#### A House United: Bridging the Script and Lexical Barrier between Hindi and Urdu

In Computational Linguistics, Hindi and Urdu are not viewed as a monolithic entity and have received separate attention with respect to their text processing. From part-of-speech tagging to machine translation, models are separately trained for both Hindi and Urdu despite the fact that they represent the same language. The reasons mainly are their divergent literary vocabularies and separate orthographies, and probably also their political status and the social perception that they are two separate languages. In this paper, we propose a simple but efficient approach to bridge the lexical and orthographic differences between Hindi and Urdu texts. With respect to text processing, addressing the differences between their texts would be beneficial in the following ways: (a) instead of training separate models, their individual resources can be augmented to train single, unified models for better generalization, and (b) their individual text processing applications can be used interchangeably under varied resource conditions.

To remove the script barrier, we learn accurate statistical transliteration models which use sentencelevel decoding to resolve word ambiguity. Similarly, we learn cross-register word embeddings from the harmonized Hindi and Urdu corpora to nullify their lexical divergences. As a proof of the concept, we evaluate our approach on the Hindi and Urdu dependency parsing under two scenarios: (a) resource sharing, and (b) resource augmentation. We demonstrate that a neural network-based dependency parser trained on augmented, harmonized Hindi and Urdu resources performs significantly better than the parsing models trained separately on the individual resources. We also show that we can achieve near state-of-the-art results when the parsers are used interchangeably. 
