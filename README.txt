"Modern applied deep learning with Transformer model methodology"

Special syllabus Spring 2024
Norwegian University of Life Sciences (NMBU)

---

Various models were created and trained. The code for these, along with the results are presented
in their respective directories. The different models are seen below, marked with "* `<path>`". The
actual model checkpoints are not included in this repository, but can be provided upon request.

Translation
  * `nb_nn/`
    Norwegian Bokmål to Norwegian Nynorsk
    Small and poor dataset.
    Therefore, also quite poor results.

  * `nb_nn/`
    English to Norwegian Bokmål
    Bigger and better dataset.
    Therefore, also better results.

Generative
  Miguel de Cervantes (Don Quixote)

    Next word prediction:
    * `cervantes/variants/big/`
      Big model.
      Slow training (too big model), and therefore not much learned.
    * `cervantes/variants/small/`
      Small model.
      Very good model, but predictions continue until forced to stop.

    Next sentence(s) prediction:
    * `cervantes/`
      Small model.
      OK results.

  Franz Kafka
  * `kafka/`
    Small model.
    Good results.

The theory is presented in `report.pdf`, along with results and simplified implementation examples.

The implementation, examples and results are presented in their corresponding directories. During
training of the latter four games, Orion HPC (https://orion.nmbu.no) at the Norwegian University of
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
- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
                                                                  arXiv:1810.04805v2
- "An image is worth 16x16 words: Transformers for image recognition at scale"
                                                            arXiv:2010.11929v2

---

Learning goals:

- Understand and know how to build, use and deploy Transformer architectures
  * Experiment with architectures and applications (for instance a language translator)

Learning outcomes:

- Be competent in modern deep learning situations
  * Understand (and to some extent be able to reproduce) cutting-edge “artificial intelligence”
    models
