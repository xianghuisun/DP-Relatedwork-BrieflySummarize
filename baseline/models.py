import torch
import torch.nn as nn
from transformers import AutoConfig,AutoModel
from utils import match_kwargs

class NERNetwork(nn.Module):
    """A Generic Network for NERDA models.
    The network has an analogous architecture to the models in
    [Hvingelby et al. 2020](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.565.pdf).
    Can be replaced with a custom user-defined network with 
    the restriction, that it must take the same arguments.
    """

    def __init__(self, model_name_or_path: str, n_tags: int, dropout: float = 0.1) -> None:
        """Initialize a NERDA Network
        Args:
            bert_model (nn.Module): huggingface `torch` transformers.
            device (str): Computational device.
            n_tags (int): Number of unique entity tags (incl. outside tag)
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(NERNetwork, self).__init__()
        
        # extract AutoConfig, from which relevant parameters can be extracted.
        bert_model_config = AutoConfig.from_pretrained(model_name_or_path)
        self.bert_model = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(dropout)
        self.tags = nn.Linear(bert_model_config.hidden_size, n_tags)#BERT+Linear

    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                token_type_ids: torch.Tensor,
                ) -> torch.Tensor:
        """Model Forward Iteration
        Args:
            input_ids (torch.Tensor): Input IDs.
            attention_mask (torch.Tensor): Attention attention_mask.
            token_type_ids (torch.Tensor): Token Type IDs.
        Returns:
            torch.Tensor: predicted values.
        """

        # TODO: can be improved with ** and move everything to device in a
        # single step.
        bert_model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
            }
        
        # match args with bert_model
        # bert_model_inputs = match_kwargs(self.bert_model.forward, **bert_model_inputs)
           
        outputs = self.bert_model(**bert_model_inputs)
        # apply drop-out
        last_hidden_state=outputs.last_hidden_state
        last_hidden_state = self.dropout(last_hidden_state)

        # last_hidden_state for all labels/tags
        last_hidden_state = self.tags(last_hidden_state)

        return last_hidden_state