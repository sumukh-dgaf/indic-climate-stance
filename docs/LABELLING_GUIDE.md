# Labelling guide (synthetic test review)

Label each comment's stance toward the claim **"Climate change and global warming is a serious concern."**
Ask: *would the person who wrote this agree with the claim?* Yes: **Favour**. No: **Against**. Cannot tell: **None**.

| Label | Use it when the comment... | Typical signals |
|---|---|---|
| Favour | treats warming or its effects as real and worrying, even without the words "climate change" | melting snow or glaciers, water running out, unbearable heat, "we must act", praise for climate action |
| Against | denies, dismisses or mocks the concern | "it's natural", "God controls the weather", "media hype / propaganda", "it was so cold, where is global warming?", calling action useless or a scam |
| None | takes no stance at all | off-topic, a genuine question, a neutral fact, undecided |

Rules that decide the hard cases:

* **Sarcasm: label what the writer means, not the literal words.** Mocking the people who worry ("wah, in this cold they
  still cry global warming") is **Against**. Mocking the people who deny it or pollute ("great, cut more trees, what a
  bright future") is **Favour**.
* **Negation: read the whole sentence.** "How can anyone say less winter and more heat is nothing to worry about?" is
  **Favour**. "Calling this a man-made crisis is a lie" is **Against**.
* **Implicit stance still counts.** A comment about drying rivers or failed crops that sounds worried is **Favour**,
  and one that brushes them off ("there were droughts before too, always new crying") is **Against**. Use **None**
  only when there is really no opinion.
* Do not aim for any particular number of each label.

`comment_is_natural`: `yes` if a real person could plausibly have posted it, `no` if it reads machine-made or broken.
