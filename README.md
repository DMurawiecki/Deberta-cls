# Deberta-cls

This task solves the NLP problem from this Kaggle competition
(https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data),
in which it is required to build a model that detects personally identifiable
information (PII) in students' writing. The task of this competition falls under
Token Classification, sometimes known as Named Entity Recognition (NER). Given a
corpus of 22,000 texts, essays written by students, one prompt at a time. The
competition asks competitors to assign labels to the following seven types of
PII:

- NAME_STUDENT - The full or partial name of a student that is not necessarily
  the author of the essay. This excludes instructors, authors, and other person
  names.
- EMAIL - A student’s email address.
- USERNAME - A student's username on any platform.
- ID_NUM - A number or sequence of characters that could be used to identify a
  student, such as a student ID or a social security number.
- PHONE_NUM - A phone number associated with a student.
- URL_PERSONAL - A URL that might be used to identify a student.
- STREET_ADDRESS - A full or partial street address that is associated with the
  student, such as their home address.

Data is presented in JSON format, which includes a document identifier, the full
text of the essay, a list of tokens, information about whitespace, and token
annotations.Token labels are presented in BIO (Beginning, Inner, Outer) format.

The project used classic libraries: hugging face, transformers, json, pathlib,
pandas and numpy. A blending of 3 pretrained DeBERTa models was used as a model.
Word level tokenization is used, and the received tokens and token_map are
loaded into the training dataset. We employ parallel processing to tokenize our
dataset, ensuring speedy execution. We slightly fine-tune the model’s on full
train data and present the final predictions on the test in .csv file.

As a plan for a model's deploy, we have several points:

- creating an API for accepting test data, loading it into the model and
  tracking results of experiments
- creating Docker Image for model’s deploy
- launching and installing libraries in Docker container using our Image on the
  server, giving it access to our API
- scaling the system onto several servers that can withstand the load

In the model's inference, it would be possible to implement the output of a text
document in which all PII would be highlighted in color.
