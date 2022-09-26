from transformers.convert_slow_tokenizer import SpmConverter
from transformers.models.deberta_v2.tokenization_deberta_v2 import (
        DebertaV2Tokenizer,
    )

from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast

from tokenizers import Regex, normalizers, processors
class DebertaV2Converter(SpmConverter):
    def normalizer(self, proto):
        list_normalizers = []
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        # precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        # if precompiled_charsmap:
        #     list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))
        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))

        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


def convert_deberta_v2_tokenizer(
    tokenizer: DebertaV2Tokenizer
) -> DebertaV2TokenizerFast:
    tokenizer.vocab_file = tokenizer._tokenizer.vocab_file
    return DebertaV2TokenizerFast(
        tokenizer._tokenizer.vocab_file,
        **tokenizer.init_kwargs,
        tokenizer_object=DebertaV2Converter(tokenizer).converted()
    )
