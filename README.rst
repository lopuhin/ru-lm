Language models (for Russian language)
======================================

Data
----

In order to train the model, pass ``--data_path`` - a path to folder with
3 files: ``train.txt``, ``valid.txt``, ``test.txt``. All files have the same
format: tokens should be separated by whitespace, with one sentence on a line,
without special end-of-sentence symbols.


License
-------

Apache 2.0

Initial version of rnn module is based on
https://www.tensorflow.org/versions/r0.11/tutorials/recurrent/index.html#recurrent-neural-networks,
which is licensed under Apache 2.0 license.
