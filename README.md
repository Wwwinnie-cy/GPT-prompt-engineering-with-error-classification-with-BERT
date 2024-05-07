# bert_error_classification
Inspired by a new prompt method--generate low principles and high principles from GPT initial wrong answers, and use {question, answer, principle} as a new prompt,
we train an automatically error classification BERT model, and then use {question, answer, principle, principle} as a new prompt. Since with error type,
the learning process of GPT is more likely to human-learning process, people always identify what error types they made when learning.
The results show that our new prompt method overcomes the original {question, answer, principle} method and CoT.
