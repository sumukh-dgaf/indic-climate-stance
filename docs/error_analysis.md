# Error Analysis – IndicBERT v2 C_full

The error analysis shows that indirect, sarcastic and Romanised comments are difficult for the model. On the synthetic Hindi+Bengali test, Roman comments had the lowest phenomenon accuracy (0.796), followed by sarcasm (0.816) and negation (0.840). On the pilot set, sarcasm accuracy fell to 0.500 and implicit comments reached only 0.571; Roman comments scored 0.727. These results suggest that the model is more reliable on direct wording than on implied meaning, informal spelling and tone.

The confusion matrices show two recurring problems. First, Favour and Against can be confused when comments mention warming, cold weather or climate action but express the opposite stance through sarcasm or rhetorical wording. Second, some Favour comments in the translated Hindi and Bengali sets are predicted as None, while some None weather observations are predicted as Favour. In the pilot matrix, many Against comments are classified as None.

For example, a Romanised Hindi comment says that the cold makes the claim of a warming Earth seem like a joke; the model predicts Favour, apparently reacting to the warming keyword rather than the dismissive meaning. A Bengali comment about a village pond drying unusually early expresses concern indirectly, but the model predicts None. Another comment sarcastically asks God for water because it is scarce, yet the model predicts None rather than recognising its sceptical tone.

These examples show why keyword matching is insufficient: stance depends on context, negation, sarcasm and implied meaning. The results should also be interpreted cautiously because the synthetic test shares its generator with synthetic training data, and the pilot is synthetic as well.


## Analysis of Manual Demo Testing

A total of 20 comments were tested using the fine-tuned IndicBERT model across Bengali, code-mixed Hindi and English inputs.

The model demonstrated good performance on clearly expressed stance comments and off-topic questions.

### Observed Errors

1. Indirect stance: The model predicted None for "I don't think climate change is a serious problem" instead of Against.
2. Contextual understanding: The model predicted None for a comment supporting solar energy as an alternative to fossil fuels instead of Favour.

### Key Observations

- The model performs well on clear Favour and Against statements.
- Off-topic questions are generally classified as None.
- Indirect opinions and implicit climate-related support remain challenging.
- Code-mixed language testing helps evaluate multilingual robustness.

### Recommendations

- Add more examples of indirect opinions to the training dataset.
- Include climate-related renewable energy statements.
- Improve representation of code-mixed and multilingual comments.
- Evaluate performance using a larger and more balanced test dataset.

