import torch

labels = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION",
          "B-ANSWER", "I-ANSWER"]
id2label = {i: label for i, label in enumerate(labels)}

def classify_text(text: str, processor, model):
    """
    Classify entities in in plain text using a BERT-based NER pipeline.
    """

    # Encode image
    encoding = processor(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        predictions = logits.argmax(-1).squeeze().tolist()

    tokens = encoding.input_ids[0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(tokens)

    entities = []

    # Group subwords
    current_entity = None
    current_word = []
    current_score = []

    for token, pred in zip(tokens, predictions):
        # Pick label from config if available, else manual fallback
        label_map = getattr(model.config, "id2label", None)
        if label_map and isinstance(label_map, dict) and pred in label_map:
            label = label_map[pred]
        else:
            label = id2label.get(pred, "O")

        if label == "O":
            # End of an entity → flush buffer if exists
            if current_entity and current_word:
                entities.append({
                    "entity": current_entity,
                    "word": "".join(current_word).replace("##", ""),
                    "score": sum(current_score) / len(current_score),
                })
                current_entity, current_word, current_score = None, [], []
            continue

        # Same entity type → keep appending
        if current_entity == label:
            current_word.append(token)
            current_score.append(
                float(torch.max(torch.softmax(logits, dim=-1)))
            )
        else:
            # Different label → flush old entity
            if current_entity and current_word:
                entities.append({
                    "entity": current_entity,
                    "word": "".join(current_word).replace("##", ""),
                    "score": sum(current_score) / len(current_score),
                })
            # Start new entity
            current_entity = label
            current_word = [token]
            current_score = [float(torch.max(torch.softmax(logits, dim=-1)))]

    # Flush last entity
    if current_entity and current_word:
        entities.append({
            "entity": current_entity,
            "word": "".join(current_word).replace("##", ""),
            "score": sum(current_score) / len(current_score),
        })

    return {"entities": entities}
