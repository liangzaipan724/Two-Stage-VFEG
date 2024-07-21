import torch
from einops import rearrange

import torch
from transformers import BertTokenizer, BertModel

def exists(val):
    return val is not None



BERT_MODEL_DIM = 768


def tokenize(texts, add_special_tokens=True):
    if not isinstance(texts, (list, tuple)):
        texts = [texts]
    tokenizer = BertTokenizer(vocab_file='/data/bert-base-cased/vocab.txt')
    encoding = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=add_special_tokens,
        padding=True,
        return_tensors='pt'
    )

    token_ids = encoding.input_ids
    return token_ids


# embedding function

@torch.no_grad()
def bert_embed(
        token_ids,
        return_cls_repr=False,
        eps=1e-8,
        pad_id=0.
):
    # model = get_bert()
    model = BertModel.from_pretrained(
        pretrained_model_name_or_path='/data/bert-base-cased',
        config='/data/bert-base-cased'
    ).cuda()
    mask = token_ids != pad_id

    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
        mask = mask.cuda()

    outputs = model(
        input_ids=token_ids,
        attention_mask=mask,
        output_hidden_states=True
    )

    hidden_state = outputs.hidden_states[-1]

    if return_cls_repr:
        return hidden_state[:, 0]  # return [cls] as representation

    if not exists(mask):
        return hidden_state.mean(dim=1)

    mask = mask[:, 1:]  # mean all tokens excluding [cls], accounting for length
    mask = rearrange(mask, 'b n -> b n 1')

    numer = (hidden_state[:, 1:] * mask).sum(dim=1)
    denom = mask.sum(dim=1)
    masked_mean = numer / (denom + eps)
    return masked_mean
