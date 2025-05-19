# Deberta-cls

This repository solves the problem of ranking LLM responses with system based on
a finetuned BERT model. User sends request to Telegram bot which is a RAG with
Mstral tiny model with scientific base. Mistral generates 5 answers and then
they go to our LLM checker and it ranks them. User gets the best answer.

![Схема архитектуры](images/architecture.png) \*\* \*\*
