import pandas as pd
import streamlit as st

from train_language_model import (
    MIN_MESSAGE_LETTERS,
    build_model,
    predict_language,
    review_reason,
    script_candidates,
)


st.set_page_config(page_title="LinguaRoute", page_icon="🌐", layout="wide")

QUEUE_NAMES = {
    "english": "English queue",
    "spanish": "Spanish queue",
    "japanese": "Japanese queue",
    "mandarin": "Mandarin queue",
}

# These made-up examples are for an interactive presentation, not training data.
DEMO_INBOX = [
    {
        "Ticket": "LR-101",
        "Message": "Hello, my order was due yesterday, but it has not arrived. Could you please check its delivery status?",
        "Expected": "english",
    },
    {
        "Ticket": "LR-102",
        "Message": "Hola, mi pedido debía llegar ayer, pero todavía no ha llegado. ¿Podrían revisar el estado de la entrega?",
        "Expected": "spanish",
    },
    {
        "Ticket": "LR-103",
        "Message": "こんにちは。注文した商品がまだ届いていません。配送状況を確認していただけますか？",
        "Expected": "japanese",
    },
    {
        "Ticket": "LR-104",
        "Message": "您好，我的订单还没有送达。请问可以帮我查询一下物流状态吗？",
        "Expected": "mandarin",
    },
    {
        "Ticket": "LR-105",
        "Message": "Thanks!",
        "Expected": "english",
    },
    {
        "Ticket": "LR-106",
        "Message": "返品希望",
        "Expected": "japanese",
    },
]


@st.cache_resource
def get_model():
    return build_model()


def get_ready_model():
    with st.spinner(
        "Preparing Tatoeba data and training the model. The first run may take a few minutes..."
    ):
        result = get_model()
        if not isinstance(result, tuple) or len(result) != 4:
            # A previous app version cached only the language model. Drop that
            # stale entry and rebuild the language and unknown-language models.
            get_model.clear()
            result = build_model()
        if not isinstance(result, tuple) or len(result) != 4:
            raise RuntimeError("Model setup did not return both classifiers and vectorizers.")
        return result


def classify_messages(messages, vectorizer, model, unknown_vectorizer, unknown_model):
    rows = []
    for item in messages:
        prediction, confidence = predict_language(
            item["Message"], vectorizer, model, unknown_vectorizer, unknown_model
        )
        candidates, _ = script_candidates(item["Message"])
        possibly_unsupported = prediction is None and candidates == {"english", "spanish"}
        reason = review_reason(
            item["Message"], confidence, possibly_unsupported=possibly_unsupported
        )
        is_review = reason is not None
        rows.append(
            {
                "Ticket": item["Ticket"],
                "Message": item["Message"],
                "Reference language": item["Expected"].title(),
                "Model estimate": prediction.title() if prediction else "No safe estimate",
                "Confidence": confidence,
                "Matches reference": prediction == item["Expected"],
                "Action": "Human review" if is_review else QUEUE_NAMES[prediction],
                "Review reason": reason or "",
            }
        )
    return pd.DataFrame(rows)


st.title("🌐 LinguaRoute")
st.caption("A language-routing prototype for international customer-support teams")
st.info(
    "It estimates English, Spanish, Japanese, or Mandarin. Uncertain or very short "
    "messages are held for human review; this prototype does not connect to a real ticket system."
)
st.warning(
    "Supported languages only: English, Spanish, Japanese, and Mandarin. Other languages—"
    "including Afrikaans and Dutch—are screened, but the detector can still miss some. "
    "Verify the language before using a queue suggestion."
)

single_tab, inbox_tab, results_tab = st.tabs(
    ["✉️ Try a message", "📥 Demo inbox", "📊 Results & how it works"]
)

with single_tab:
    st.subheader("Check one message")
    st.write("Enter a message and see both the model estimate and the routing safeguard.")
    text = st.text_area(
        "Customer message",
        height=120,
        placeholder="Example: I need help changing the delivery address for my order.",
        key="single_message",
    )
    if st.button("Analyze message", type="primary", key="analyze_single"):
        if not text.strip():
            st.warning("Enter a message first.")
        else:
            vectorizer, model, unknown_vectorizer, unknown_model = get_ready_model()
            pred_label, confidence = predict_language(
                text, vectorizer, model, unknown_vectorizer, unknown_model
            )
            candidates, _ = script_candidates(text)
            possibly_unsupported = pred_label is None and candidates == {"english", "spanish"}
            reason = review_reason(
                text, confidence, possibly_unsupported=possibly_unsupported
            )
            letter_count = sum(character.isalpha() for character in text)
            if pred_label is None:
                if possibly_unsupported:
                    st.subheader("Language estimate: Possibly outside supported languages")
                    st.metric("Supported-language detector score", f"{confidence:.1%}")
                    st.caption("Screening score only—not a calibrated probability.")
                else:
                    st.subheader("Language estimate: Needs human review")
                st.warning(reason or "The writing system could not be matched safely.")
                st.caption("No queue is recommended for this message.")
            elif letter_count < MIN_MESSAGE_LETTERS:
                st.subheader("Language estimate: Not enough text")
                st.warning(f"Human review recommended: {reason}")
                with st.expander("Why did the classifier guess a language?"):
                    st.write(
                        f"After limiting candidates based on its writing system, the model "
                        f"leans **{pred_label.title()}** ({confidence:.1%} confidence "
                        f"estimate among those candidates). But this message has only "
                        f"{letter_count} letters. That is too little text to show a reliable "
                        "language result or route it."
                    )
                st.caption("No queue is recommended for this message.")
            elif reason:
                st.subheader("Language estimate: Needs human review")
                st.metric("Script-aware model estimate", pred_label.title())
                st.metric("Confidence estimate", f"{confidence:.1%}")
                st.warning(f"Human review recommended: {reason}")
                st.caption("No queue is recommended for this message.")
            else:
                left, right = st.columns(2)
                left.metric("Language estimate", pred_label.title())
                right.metric("Confidence estimate", f"{confidence:.1%}")
                st.success(f"Suggested destination: {QUEUE_NAMES[pred_label]}")
                st.caption(
                    "Prototype recommendation only—not an actual transfer. The model only "
                    "supports four languages; verify the message is one of them before using this suggestion."
                )

with inbox_tab:
    st.subheader("Support inbox simulation")
    st.write(
        "Review a batch of incoming requests and see which language queues the model suggests. "
        "Two brief messages demonstrate when a human should verify the language."
    )
    st.caption(
        "Illustrative fictional examples only—not real customer messages or personal data. "
        "The reference-language column is included for this demonstration."
    )
    preview = pd.DataFrame(
        [{"Ticket": row["Ticket"], "Message": row["Message"]} for row in DEMO_INBOX]
    )
    st.dataframe(preview, use_container_width=True, hide_index=True)

    if st.button("Review incoming messages", type="primary", key="process_inbox"):
        vectorizer, model, unknown_vectorizer, unknown_model = get_ready_model()
        st.session_state["inbox_results"] = classify_messages(
            DEMO_INBOX, vectorizer, model, unknown_vectorizer, unknown_model
        )

    if "inbox_results" in st.session_state:
        results = st.session_state["inbox_results"]
        auto_rows = results[results["Action"] != "Human review"]
        review_rows = results[results["Action"] == "Human review"]
        wrong_auto = auto_rows[~auto_rows["Matches reference"]]
        metric_columns = st.columns(3)
        metric_columns[0].metric("Messages", len(results))
        metric_columns[1].metric("Suggested queue", len(auto_rows))
        metric_columns[2].metric("Held for review", len(review_rows))

        st.dataframe(
            results[
                [
                    "Ticket",
                    "Reference language",
                    "Model estimate",
                    "Confidence",
                    "Matches reference",
                    "Action",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={"Confidence": st.column_config.ProgressColumn("Confidence", format="%.1%%", min_value=0, max_value=1)},
        )
        if len(review_rows):
            st.warning(
                f"{len(review_rows)} message(s) were held for a person to review. "
                f"Among messages eligible for a queue suggestion, {len(wrong_auto)} "
                "did not match the reference language in this illustrative example set."
            )
            with st.expander("See why messages were held"):
                st.dataframe(
                    review_rows[["Ticket", "Message", "Model estimate", "Review reason"]],
                    use_container_width=True,
                    hide_index=True,
                )
        st.caption(
            "This is a tiny illustrative scenario check, not a representative accuracy test. "
            "Use the held-out Tatoeba metrics in the Results tab for benchmark performance."
        )

with results_tab:
    st.subheader("What we measured")
    st.write("Held-out test results on a balanced sample of Tatoeba sentences—not real support tickets.")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Model accuracy", "99.35%", help="3,974 correct out of 4,000 held-out examples.")
    metric_columns[1].metric("Fixed-label baseline", "25.00%", help="Always predict Mandarin: 1,000/4,000 on four balanced classes.")
    metric_columns[2].metric("Automatic-route checks", "3,301 / 4,000", help="After script-aware filtering, unknown-language screening, and review checks, zero errors among eligible messages in this held-out sample.")

    st.markdown("#### How the model works")
    st.markdown(
        "1. **TF-IDF** turns short character sequences into numeric features.\n"
        "2. **Logistic regression** uses those features to estimate one of four languages.\n"
        "3. **Writing-system check** removes impossible script candidates: Latin text is compared as English/Spanish; Japanese kana points to Japanese; Han text is compared as Japanese/Mandarin.\n"
        "4. A separate supported-vs-other model screens Latin text using examples from French, Portuguese, German, and Italian.\n"
        "5. The app asks for human review if there are fewer than 8 letters, confidence is low, the message may be unsupported, or its writing system is mixed/unsupported."
    )
    st.markdown("#### Unknown-language screening check")
    st.write(
        "The supported-vs-other detector was trained with French, Portuguese, German, and "
        "Italian as examples outside the four target languages. Afrikaans and Dutch were "
        "kept out of detector training and used as a language-level generalization check."
    )
    unknown_columns = st.columns(3)
    unknown_columns[0].metric("Validation: supported accepted", "96.6%")
    unknown_columns[1].metric("Held-out Afrikaans rejected", "66.4%")
    unknown_columns[2].metric("Held-out Dutch rejected", "77.2%")
    st.caption(
        "At a 70% supported-score threshold; not perfect. Some Afrikaans and Dutch messages "
        "still pass as supported. These sentence-corpus results do not guarantee detection "
        "of every unknown language or of real support messages."
    )
    with st.expander("Show per-language benchmark scores"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Language": "English", "Precision": 0.9960, "Recall": 0.9970, "F1": 0.9965, "Test examples": 1000},
                    {"Language": "Japanese", "Precision": 1.0000, "Recall": 0.9830, "F1": 0.9914, "Test examples": 1000},
                    {"Language": "Mandarin Chinese", "Precision": 0.9794, "Recall": 0.9970, "F1": 0.9881, "Test examples": 1000},
                    {"Language": "Spanish", "Precision": 0.9990, "Recall": 0.9970, "F1": 0.9980, "Test examples": 1000},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.warning(
        "Limitations: the benchmark contains only four languages and balanced Tatoeba "
        "sentences. The out-of-scope detector rejected 66.4% of held-out Afrikaans and "
        "77.2% of held-out Dutch sentences; it missed the rest. It screens but does not "
        "reliably reject every unknown language. Mixed text and shared writing systems can "
        "also cause errors. Detector scores are not calibrated probabilities."
    )
    st.caption("Data: Tatoeba detailed sentence export, CC BY 2.0 FR. See the project README for attribution and full methodology.")
