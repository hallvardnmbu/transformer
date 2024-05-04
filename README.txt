"Modern applied deep learning with Transformer model methodology"

Special syllabus Spring 2024
Norwegian University of Life Sciences (NMBU)

---

Various models were created and trained. The code for these, along with the results are presented
in their respective directories. The different models are seen below, marked with "* <path>". The
actual model checkpoints are not included in this repository, but can be provided upon request.

  Encoder-decoder (full transformer)
  ..................................

    Translation
    * nb-nn/
      Norwegian Bokmål to Norwegian Nynorsk
      Small and poor dataset.
      Poor results (due to dataset).
    * en-nb/
      English to Norwegian Bokmål
      Bigger and better dataset.
      OK results.

    Miguel de Cervantes (Don Quixote)
    * cervantes/
      Next sentence(s) prediction based on some input.
      OK results.

    Franz Kafka
    * kafka/
      Next sentence(s) prediction based on some input.
      Good results.

    Bible paragraph generation
    * bible/
      Next sentence(s) prediction based on some input.
      Good results.

    Song lyric generation
    * lyrics/
      Lyrics generation based on song title and artist.
      OK results.

    Finetuning T5
    ¨¨¨¨¨¨¨¨¨¨¨¨¨
      Song lyric generation
      * finetune/t5/lyrics/
        Lyrics generation based on song title and artist.
        Great results.

  Decoder-only
  ............

    Miguel de Cervantes (Don Quixote)
    * cervantes/decoder-only/big/
      N.b.: does NOT contain stop-tokens, i.e., generates until `generate` tokens created.
      Big model.
      Slow training (too big model), and therefore not much learned when forced to stop training.
    * cervantes/decoder-only/small/
      N.b.: does NOT contain stop-tokens, i.e., generates until `generate` tokens created.
      Small model.
      Good results.

    Finetuning GPT-2
    ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
      Biblical text generation
      * finetune/gpt2/bible/
        Note: does NOT contain stop-tokens, i.e., generates until `generate` tokens created.
        Great results.

      Miguel de Cervantes (Don Quixote)
      * finetune/gpt2/cervantes/
        N.b.: does NOT contain stop-tokens, i.e., generates until `generate` tokens created.
        Great results.

      Franz Kafka
      * finetune/gpt2/kafka/
        N.b.: does NOT contain stop-tokens, i.e., generates until `generate` tokens created.
        Great results.

---

The theory is presented in `report.pdf`, along with results and simplified implementation examples.

The implementation, examples and results are presented in their corresponding directories. During
training, Orion HPC (https://orion.nmbu.no) at the Norwegian University of
Life Sciences (NMBU) provided computational resources.

---

Relevant literature:

- Geometry of deep learning
  * Chapter 9.3 ("Attention")
  * Chapter 9.4.5 ("Transformer")
  * Chapter 9.4.7 ("Generative Pre-trained Transformer (GPT)")
  ISBN 978-981-16-6046-7

Relevant papers:

- "Attention is All You Need"
           arXiv:1706.03762v7
- "Language Models are Unsupervised Multitask Learners"
                                            OpenAI 2019
- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
                                                                  arXiv:1810.04805v2

---

Learning goals:

- Understand and know how to build, use and deploy Transformer architectures
  * Experiment with architectures and applications (for instance a language translator)

Learning outcomes:

- Be competent in modern deep learning situations
  * Understand (and to some extent be able to reproduce) cutting-edge "artificial intelligence"
    models
