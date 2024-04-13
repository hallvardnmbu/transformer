"""The GPT Language Model. From https://github.com/karpathy/nanoGPT/blob/master/model.py"""

import logging
import inspect
import torch

from block import Block, LayerNorm
from config.gpt import GPTConfig


LOGGER = logging.getLogger(__name__)


class GPT(torch.nn.Module):
    """The GPT Language Model."""
    def __init__(self, config):
        """
        Initialize the GPT model.

        Parameters
        ----------
        config : dataclasses.dataclass
            Configuration object (hyperparameters) for the model.
        """
        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        self.transformer = torch.nn.ModuleDict({
            "wte": torch.nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": torch.nn.Embedding(config.block_size, config.n_embd),
            "drop": torch.nn.Dropout(config.dropout),
            "h": torch.nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": LayerNorm(config.n_embd, bias=config.bias),
        })
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / torch.sqrt(2 * config.n_layer))

        LOGGER.info("Number of parameters: %s", self.get_num_params() / 1e6)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.

        Parameters
        ----------
        non_embedding: bool, optional
            If True, subtract the number of parameters in the position embeddings.

        Returns
        -------
        n_params : int
            The number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    @staticmethod
    def _init_weights(module):
        """
        Initialize the weights of a predefined model.

        Parameters
        ----------
        module : torch.nn.Module
        """
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the GPT model.

        Parameters
        ----------
        idx : torch.Tensor
            Input tensor of shape `(batch_size, sequence_length)`.
        targets : torch.Tensor, optional
            Target tensor of shape `(batch_size, sequence_length)`.

        Returns
        -------
        logits : torch.Tensor
            The output logits of the model.
        loss : torch.Tensor, optional
            The loss of the model, if targets are provided.
        """
        device = idx.device
        _, t = idx.size()

        assert t <= self.config.block_size, (f"Cannot forward sequence of length {t}, "
                                             f"block size is only {self.config.block_size}")

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        tok_emb = self.transformer.wte(idx)  # Token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # Position embeddings of shape (t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Calculate the loss if targets are provided.
        if targets is not None:
            logits = self.lm_head(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1
            )
            return logits, loss

        # Inference-time mini-optimization: only forward the lm_head on the very last position.
        logits = self.lm_head(x[:, [-1], :])  # Note: using list [-1] to preserve the time dim.

        return logits, None

    def crop_block_size(self, block_size):
        """
        Model surgery to decrease the block size if necessary.

        Parameters
        ----------
        block_size : int
            The new block size.

        Notes
        -----
        E.g., we may load the GPT2 pretrained model checkpoint (block size 1024) but want to use
        a smaller block size for some smaller, simpler model.
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

        self.transformer.wpe.weight = torch.nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Load a pretrained GPT model from the Huggingface Transformers library.

        Parameters
        ----------
        model_type : str
            The model type to load. One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'.
        override_args : dict, optional
            Override arguments for the model.

        Returns
        -------
        model : GPT
            The pretrained GPT model.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}

        # Only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)

        from transformers import GPT2LMHeadModel  # pylint: disable=import-outside-toplevel
        LOGGER.info("Loading weights from pretrained GPT: %s", model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': {"n_layer": 12, "n_head": 12, "n_embd": 768},   # 124M params
            'gpt2-medium': {"n_layer": 24, "n_head": 16, "n_embd": 1024},  # 350M params
            'gpt2-large': {"n_layer": 36, "n_head": 20, "n_embd": 1280},  # 774M params
            'gpt2-xl': {"n_layer": 48, "n_head": 25, "n_embd": 1600},  # 1558M params
        }[model_type]

        LOGGER.info("Forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257   # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024    # always 1024 for GPT model checkpoints
        config_args['bias'] = True          # always True for GPT model checkpoints

        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        # Create a from-scratch initialized karpathy/nanoGPT model.
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # Discard this mask / buffer

        # Fetch the huggingface/transformers model.
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy while ensuring all the parameters are aligned and match in names and shapes.
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight',
                      'attn.c_proj.weight',
                      'mlp.c_fc.weight',
                      'mlp.c_proj.weight']

        # OpenAI's checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them.
        assert len(sd_keys_hf) == len(sd_keys), (f"mismatched keys: "
                                                 f"{len(sd_keys_hf)} != {len(sd_keys)}")
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for the Conv1D weights we need to transpose.
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy over the other parameters.
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure the optimizer for the model.

        Parameters
        ----------
        weight_decay : float
        learning_rate : float
        betas : tuple[float, float]
        device_type : str

        Returns
        -------
        optimizer : torch.optim.AdamW
            The AdamW optimizer.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # I.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        LOGGER.info("Num decayed parameter tensors: %s, with %s parameters",
                    len(decay_params), num_decay_params)
        LOGGER.info("Num non-decayed parameter tensors: %s, with %s parameters",
                    len(nodecay_params), num_nodecay_params)

        # Create AdamW optimizer and use the fused version if it is available.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        LOGGER.info("Using fused AdamW: %s", use_fused)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.

        Parameters
        ----------
        fwdbwd_per_iter : int
            Number of forward-backward passes per iteration.
        dt : float
            Time in seconds for one forward-backward pass.

        Notes
        -----
        First estimate the number of flops we do per iteration.
        - See PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        Express our flops throughput as ratio of A100 bfloat16 peak flops.
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        Parameters
        ----------
        idx : torch.Tensor
            Input tensor of shape `(batch_size, sequence_length)`.
        max_new_tokens : int
            The number of tokens to generate.
        temperature : float, optional
            The temperature for sampling.
        top_k : int, optional
            The top-k value for sampling.

        Returns
        -------
        idx : torch.Tensor
            The generated indices.

        Notes
        -----
        The temperature is a hyperparameter that controls the randomness of the sampling.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx \
                if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
