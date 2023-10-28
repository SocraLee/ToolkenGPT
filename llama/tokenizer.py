# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List
import os


logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: str, new_tokens):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.replace_dict={}
        self.reverse_dict = {}
        for index in range(len(new_tokens)):
            tokens = self.sp_model.encode(new_tokens[index])[:]
            self.replace_dict[tokens] = index+self.sp_model.vocab_size()
            self.reverse_dict[index+self.sp_model.vocab_size()]= tokens
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def replace(self, tokenizer_output):
        new_output = []
        i = 0
        while i < len(tokenizer_output):
            indicator = False
            token = tokenizer_output[i]
            for multi_token in self.replace_dict:
                if multi_token[0] == token:
                    # 计算多token字符串的长度
                    length = len(multi_token)
                    # 提取出对应的token序列
                    check = True
                    for k in range(length):
                        if multi_token[k]!=tokenizer_output[i+k]:
                            check=False
                            break
                    if check:
                        new_id = self.replace_dict[multi_token]
                        new_output.append(new_id)
                        i+=length
                        indicator = True
                        break
            if not indicator:
                # 如果没有找到匹配的多token字符串，就添加原始id到输出中
                new_output.append(tokenizer_output[i])
                i += 1
        return new_output
    def reverse(self,t):



    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        self.replace(t)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        new_output= []
        for token in t:
            if token in self.reverse_dict:
                new_output.extend(self.reverse_dict[token])
            else:
                new_output.append(token)
        return self.sp_model.decode(new_output)
