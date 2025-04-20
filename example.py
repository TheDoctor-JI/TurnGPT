from argparse import ArgumentParser
from turngpt.model import TurnGPT

parser = ArgumentParser()
parser = TurnGPT.add_model_specific_args(parser)
args = parser.parse_args()

# print out args
for k, v in vars(args).items():
    print(f"{k}: {v}")

# Fresh Initialization

# model = TurnGPT(
#     pretrained_model_name_or_path=args.pretrained_model_name_or_path,
#     # pretrained_model_name_or_path=path,
#     trp_projection_steps=args.trp_projection_steps,
#     trp_projection_type=args.trp_projection_type,
#     weight_loss=args.weight_loss,
#     weight_eos_token=args.weight_eos_token,
#     weight_regular_token=args.weight_regular_token,
#     learning_rate=args.learning_rate,
#     dropout=args.dropout,
#     pretrained=args.pretrained,
#     no_train_first_n=args.no_train_first_n,
#     omit_dialog_states=args.omit_dialog_states,
# )



# These must be called on a freash initialization (later done when loading the model)
# on checkpoint-save the `tokenizer` is saved with the model.
# # on checkpoint-load the `tokenizer` is loaded and the weights extended automatically
# model.init_tokenizer()  # required for fresh model (saved on checkpoint)
# model.initialize_special_embeddings()  # required for fresh model (also performed on load_checkpoint)
# model.print_parameters()


path = '/home/eeyifanshen/e2e_audio_LLM/TurnGPT/runs/TurnGPT/TurnGPT_8ae66gi9/epoch=6_val_loss=1.7988.ckpt'

model = TurnGPT.load_from_checkpoint(path)


print(model.tokenizer)
# PreTrainedTokenizerFast(name_or_path='gpt2', vocab_size=50257,
# model_max_len=1024, is_fast=True, padding_side='right',
# special_tokens={'bos_token': '<|endoftext|>', 'eos_token': '<ts>',
# 'unk_token': '<|endoftext|>', 'pad_token': '<|endoftext|>',
# 'additional_special_tokens': ['<speaker1>', '<speaker2>']})


# Example use
turn_list = [
    "Hello there I basically had the worst day of my life",
    "Oh no, what happened?",
    "Do you want the long or the short story?",
]
turn_list2 = [
    "Hello there I basically had the worst day of my life",
    "Oh no, what happened?",
]
multiple = [turn_list, turn_list2]

# Get trp from a text string
out = model.string_list_to_trp(turn_list[0], add_post_eos_token=True)

# Get trp from a text list
out = model.string_list_to_trp(turn_list)

# Get trp from a list of text lists
out = model.string_list_to_trp(multiple)

# out: dict_keys(['logits', 'past_key_values', 'probs', 'trp_probs', 'tokens'])

# Simple Plot
import matplotlib.pyplot as plt
import torch

def plot_trp(P, text):
    fig, ax = plt.subplots(1, 1)
    x = torch.arange(len(P))
    ax.bar(x, P)
    ax.set_xticks(x)
    ax.set_xticklabels(text, rotation=60)
    ax.set_ylim([0, 1])
    plt.pause(0.01)
    return fig, ax

fig, ax = plot_trp(out["trp_probs"][0], out["tokens"][0])
print(f"Tokens: {out['tokens'][0]}, TRP: {out['trp_probs'][0]}")