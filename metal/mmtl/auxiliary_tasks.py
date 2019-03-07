import numpy as np

from metal.mmtl.utils.dataloaders import add_labels_to_payload


# Function to add BLEU labels
def add_bleu_labels(payload):
    """
    Adds 1-gram bleu score labelset for sentence similarity tasks
    """

    def get_bleu_label(it):
        toks, segs = it[0]
        toks = payload.data_loader.dataset.tokenizer.convert_ids_to_tokens(toks)
        toks, segs = np.array(toks), np.array(segs)
        sent1 = list(toks[segs == 0])
        sent2 = list(toks[segs == 1])
        bleu_score = sentence_bleu(sent1, sent2, weights=(1, 0, 0, 0))
        return bleu_score

    add_labels_to_payload(payload, "BLEU", get_bleu_label)
