# LinguaRoute — 5-minute pitch

**Track:** AI + Languages  
**Speaker/team:** Tang Tang (Team Subaru)

## 1. Hook — about 15 seconds

“Imagine you’re a support agent handling messages in several languages. Before you can help, you need to get each message to the right team. LinguaRoute explores whether AI can make that first step faster.”

## 2. Problem — about 30 seconds

“Our user is a customer-support agent working with international customers. If a message is sent to the wrong language queue, it may take longer to reach someone who can help. We focused on a small, specific task: identify whether a message is English, Spanish, Japanese, or Mandarin, then recommend the matching queue. Our prototype makes a recommendation; it does not connect to a real ticketing system.”

## 3. Solution — about 20 seconds

“We built LinguaRoute, a Streamlit demo where an agent enters a message and gets a predicted language, a confidence estimate, and a suggested support queue. The AI is a TF-IDF character n-gram classifier with logistic regression.”

## 4. Live demo — about 2 minutes

1. **Single-message tab (20 sec):** Enter “Hello, my order was due yesterday, but it has not arrived. Could you please check its delivery status?” Show the English queue suggestion; clarify it does not transfer a real ticket.
2. **Show short text (15 sec):** Enter “Thanks!” The model estimates English after the script check, but the app asks for review because six letters are too little to route safely.
3. **Show unknown-language screening (25 sec):** Enter the user's Afrikaans example, “My besending het nog nie aangekom nie.” Show “Possibly outside supported languages” and explain the separate screen.
4. **Inbox tab (35 sec):** Review the six fictional example requests. Four clear messages get queue suggestions; two brief messages are held for review. Clarify that these examples are illustrative, not real customer data or a performance benchmark.
5. **Results tab (25 sec):** Show raw four-language accuracy (3,974/4,000, 99.35%) against the 25% fixed-label baseline, then the separate unseen-language check: it rejected 66.4% of held-out Afrikaans and 77.2% of Dutch sentences. Some unknown messages still pass.

**Demo backup:** Have screenshots or a short recording ready. The first run downloads and processes about 305 MB of Tatoeba data and trains the model, so run it before presenting. If unavailable, explain the input, output, and verified results rather than implying a live prediction occurred.

## 5. How we built it — about 40 seconds

“We used Tatoeba's detailed sentence export and sampled attributed examples. The main classifier uses character n-gram TF-IDF and logistic regression. Separately, a binary supported-versus-other detector learned from French, Portuguese, German, and Italian examples; we tested it on languages it never saw during training—Afrikaans and Dutch. We built with Python, scikit-learn, pandas, and Streamlit.”

## 6. Limitation and next step — about 35 seconds

“The raw model made 26 errors out of 4,000; 17 Japanese examples were predicted as Mandarin. On the separate held-out-language test, our unknown-language detector rejected 66.4% of Afrikaans and 77.2% of Dutch, but missed the rest. On the target-language test, 3,301 messages passed the routing safeguards with zero errors in this sample. These encouraging results are not guarantees for real customer messages.”

## 7. Credits and wrap-up — about 20 seconds

“We used Tatoeba's detailed sentence export. Tatoeba's downloads page states sentence files are CC BY 2.0 FR. We generated a contributor and sentence-ID attribution manifest, and we do not redistribute the corpus or quote corpus examples. We used Claude Code and GitHub Copilot for planning and coding assistance; the team reviewed and tested the project.”

## Likely judge questions

### How is this different from a general translation app?

“We focus on the first step of support triage: identifying a message's language and suggesting a queue. We do not translate messages or claim to replace a support agent.”

### Is this really an AI/ML project, or mostly rules?

“The main prediction comes from a model trained on 20,000 labeled Tatoeba sentences: character n-gram TF-IDF features with logistic regression. The writing-system checks are a small safety layer around that model; they narrow possible languages or send unsupported scripts to review. They do not replace the learned classifier.”

### What if the message is in a language you do not support?

“We added a separate ML detector trained on French, Portuguese, German, and Italian examples, and tested it on languages excluded from its training. It rejected 66.4% of Afrikaans and 77.2% of Dutch examples at our chosen threshold. It rejected the demo Afrikaans sentence, but it does not catch every unsupported message; some can still be confused with a supported language.”

### Why is the benchmark accuracy so high?

“The test sample is balanced across four languages, and their writing systems provide strong character-level clues. We selected features on a separate validation split, then evaluated on held-out Tatoeba sentences. The result describes this sample, not performance on support conversations or unknown languages.”

### What was the hardest part?

“Choosing the right data and evaluating fairly. A dataset can have clear labels but still not match the real messages we want to help with, so we kept the benchmark claim separate from real-world performance.”

### Where does it fail?

“The raw character model made 26 errors out of 4,000, mostly Japanese predicted as Mandarin. The app then narrows candidate languages by writing system and holds short or low-confidence messages. On this test sample, 3,301 passed the automatic-routing checks with zero errors; 699 were held for review. We need new support-message tests before drawing real-world conclusions.”

### What would you do with another week?

“Add diverse, short, real-world-style test messages from more languages, then improve the rejection model while measuring both missed unknowns and supported messages incorrectly sent to review.”

### Who would actually use it?

“A support agent or triage coordinator who receives incoming messages in several languages and needs a quick suggestion for which team should review each one.”

## Before presenting

- Run the app and rehearse the demo examples; keep a screenshot/video backup.
- Do not describe the benchmark score as real-world accuracy.
- Keep both Tatoeba attribution CSVs in `docs/`; do not redistribute the corpus archive or quote its examples. Add a recorded demo link only after one exists.
