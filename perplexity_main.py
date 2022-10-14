"""A small application that explores the use of seq2seq output 
probability vector of gpt2 to highlight interesting parts of text.
"""
import transformers
from torch import nn
import torch
import numpy
import gradio

_CHECKPOINT = "gpt2"

def transform(seq): 
    """
    Transform sequence into html where top 80% percentile of 
    interesting words by log-likelihood are highlighted.
    """
    def run_model(tokenizer, sequence): 
        """
        Runs the GPT2 model with given tokenizer over a sequence.
        """
        model = transformers.AutoModelForCausalLM.from_pretrained(_CHECKPOINT)
        inputs = tokenizer(sequence, return_tensors="pt") # Pytorch tensor outputs.
        inputs_raw = tokenizer(sequence) # Regular python list.
        full_tokens = tokenizer.convert_ids_to_tokens(inputs_raw['input_ids'])
        tokens = full_tokens[1:]
        model_outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"], return_dict=True)

        # The probability distribution for each index predicts the next value.
        # Compute the unreduced loss.
        per_token_logits = model_outputs['logits'][0, :-1, :]
        per_token_labels = inputs["input_ids"][:, 1:]
        loss_funct = nn.CrossEntropyLoss(reduce=False)
        loss = loss_funct(per_token_logits, per_token_labels.view(-1))
        
        # Find the outliers in loss. 
        # TODO(akshay): Improve this silly rule.
        logs = loss.detach().numpy()
        idxs = numpy.where(logs > numpy.percentile(logs, 80))[0] + 1
        return full_tokens, idxs
    
    # Construct the sequence. 
    def construct_output_seq(tokenizer, full_tokens, idxs): 
        """
        Returns an html string where the any token in 
        idxs is highlighted.
        Args: 
            tokenizer: The gpt2 tokenizer. 
            full_tokens: Each token in the input sequence. 
            idxs: Indices that must be highlighted. 
        """
        out = ""
        convert_fn = tokenizer.convert_tokens_to_string
        for i, tok in enumerate(full_tokens): 
            if i == 0: 
                out += convert_fn(tok)
                continue

            if i in idxs: 
                # Outlier.
                out += "<mark>"+convert_fn(tok)+"</mark>"
            else: 
                out += convert_fn(tok)

        return out
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(_CHECKPOINT)
    tokens, idxs = run_model(tokenizer, seq)
    return construct_output_seq(tokenizer, tokens, idxs)

SNIPPET_2 = """
Agency is the capacity to act. More subtly: An individual's life can continue, 
with a certain inertia, that will lead them on to the next year or decade. 
Most people today more-or-less know what they are going to be doing for the 
first twenty-or-more years of their life—being in some kind of school (the 
"doing" is almost more "being told what to do"). Beyond that age there is of
course the proverbial worker, in modern stories usually an office worker,
who is often so inert that he becomes blindsided by a sudden yank of reality 
(that forces him out of his inertia, and in doing so the story begins).
""".strip()
if __name__ == '__main__': 
    demo = gradio.Interface(
        transform,
        gradio.inputs.Textbox(placeholder="Enter sentence here..."),
        outputs=["html"],
        title="Interesting parts of text are automatically highlighted.",
        live=True, 
        examples=[
            ["The more things change, the more they stay the same."], 
            [SNIPPET_2], 
        ]
    )

    demo.launch()