---
layout: page
title: Abstract
permalink: /leveraging/
category: subpost
date: 2018-10-23

---

#### Leveraging Newswire Treebanks for Parsing Conversational Data with Argument Scrambling

We investigate the problem of parsing conversational data of morphologically-rich languages such as Hindi where argument scrambling occurs frequently. We evaluate a state-of-the-art non-linear transitionbased parsing system on a new dataset containing 506 dependency trees for sentences from Bollywood (Hindi) movie scripts and Twitter posts of Hindi monolingual speakers. We show that a dependency parser trained on a newswire treebank is strongly biased towards the canonical structures and degrades when applied to conversational data. Inspired by Transformational Generative Grammar (Chomsky, 1965), we mitigate the sampling bias by generating all theoretically possible alternative word orders of a clause from the existing (kernel) structures in the treebank. Training our parser on canonical and transformed structures improves performance on conversational data by around 9% LAS over the baseline newswire parser. 
