# Demo Testing Log – Kushika Mishra

## Model Information

**Model tested:** IndicBERT v2 fine-tuned on configuration C_full  
**Model path:** `models/indicbert_C_full/`  
**Demo command:** `python src/demo.py`

## Testing Results

| # | Comment | Type | My label | Model label | Probabilities (Favour, Against, None) | Correct? |
|---|---|---|---|---|---|---|
| 5 | Climate change ki wajah se garmi har saal badh rahi hai. | Code-mixed Hindi | Favour | Favour | 0.98, 0.01, 0.00 | Yes |
| 6 | Climate change is just fake news, is par believe mat karo. | Code-mixed Hindi | Against | Against | 0.07, 0.92, 0.01 | Yes |
| 7 | Yaar aaj weather bahut amazing hai! | Code-mixed Hindi | None | None | 0.02, 0.01, 0.97 | Yes |
| 8 | Government ko renewable energy par zyada investment karna chahiye. | Code-mixed Hindi | Favour | Favour | 0.61, 0.00, 0.39 | Yes |
| 9 | The government must take strict action against pollution. | English | Favour | Favour | 0.58, 0.00, 0.42 | Yes |
| 10 | I don't think climate change is a serious problem. | English | Against | None | 0.11, 0.34, 0.56 | No |
| 11 | What is your favourite movie? | English | None | None | 0.01, 0.00, 0.99 | Yes |
| 12 | Solar energy can help reduce our dependence on fossil fuels. | English | Favour | None | 0.35, 0.00, 0.65 | No |
| 13 | Climate change is affecting millions of people around the world. | English | Favour | Favour | 0.99, 0.00, 0.01 | Yes |
| 14 | People are exaggerating the effects of global warming. | English | Against | Against | 0.03, 0.96, 0.00 | Yes |
| 15 | Can someone explain how carbon emissions affect the atmosphere? | English | None | None | 0.27, 0.00, 0.73 | Yes |
| 16 | Planting more trees is an effective way to fight climate change. | English | Favour | Favour | 0.99, 0.00, 0.01 | Yes |
| 17 | Rising temperatures are creating serious problems for farmers. | English | Favour | Favour | 0.95, 0.00, 0.05 | Yes |
| 18 | Climate change is a hoax created to scare people. | English | Against | Against | 0.00, 1.00, 0.00 | Yes |
| 19 | Why do seasons change throughout the year? | English | None | None | 0.01, 0.00, 0.99 | Yes |
| 20 | We should adopt sustainable practices to protect our environment. | English | Favour | Favour | 0.94, 0.00, 0.06 | Yes |


## What the Model Gets Wrong

1. The model may classify comments expressing neutral or uncertain opinions as Against, even when they do not communicate a clear stance.
2. Sarcastic statements and indirect expressions can be difficult to interpret correctly.
3. The model may sometimes confuse criticism of climate-related policies with opposition to climate change itself.

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
