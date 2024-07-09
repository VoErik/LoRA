## Whats this?
---
Basic (unoptimized) LoRA implementation done for a seminar presentation.
In `train.py` a pretrained `DistilBert`-model is trained on a reduced version of the `IMDB sentiment analysis` dataset.

We train three times:

1. Fully fine-tuned (FFT)
2. Classification-head fine-tuning (CHFT)
3. Fine-tuning with LoRA (LoRA)

These are the results:

| Model              | Trainable Params | Val Acc. | Test Acc. | F1 (Test) | Train Time (min) |
|--------------------|------------------|----------|-----------|-----------|------------------|
| DistilBert (FFT)*  | 66,955,010       | 0.9280   | 0.9231    | -         | -                |
| DistilBert (CHFT)* | 592,130          | 0.8726   | 0.8622    | -         | -                |
| DistilBert (LoRA)* | 516,096          | 0.9296   | 0.9239    | -         | -                |                  
| DistilBert (FFT)   | 66,955,010       | 0.8900   | 0.9010    | 0.9010    | 39.88            |
| DistilBert (CHFT)  | 592,130          | 0.7580   | 0.8070    | 0.8070    | 12.99            |
| DistilBert (LoRA)  | 516,096          | 0.8980   | 0.8950    | 0.8719    | 29.44            |

The training was done on an Apple M2, with MPS. The higher training time for the LoRA fine-tuning likely stems from the code being far from optimized (currently).

---

Run the training script with

```console
python train.py --task TASKNAME
```
* fft - full fine-tuning
* cls - classification head fine-tuning
* lora - LoRA fine-tuning

### Dataset
```
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
title     = {Learning Word Vectors for Sentiment Analysis},
booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
month     = {June},
year      = {2011},
address   = {Portland, Oregon, USA},
publisher = {Association for Computational Linguistics},
pages     = {142--150},
url       = {http://www.aclweb.org/anthology/P11-1015}
}
```
