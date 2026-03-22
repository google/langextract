import unittest
from langextract import resolver as resolver_lib
from langextract.core import data
from langextract.core import tokenizer

class ResolverParamsTest(unittest.TestCase):

    def test_resolver_accepts_alignment_params(self):
        # This should currently raise TypeError
        resolver = resolver_lib.Resolver(
            fuzzy_alignment_threshold=0.6,
            enable_fuzzy_alignment=True
        )
        self.assertEqual(resolver.fuzzy_alignment_threshold, 0.6)
        self.assertTrue(resolver.enable_fuzzy_alignment)

    def test_align_uses_instance_threshold(self):
        # Setup a case where default threshold (0.75) fails but 0.6 succeeds
        # Extraction: "headache and fever" (3 tokens)
        # Source: "Patient reports back pain and a fever."
        # Intersection: "and", "fever" (2 tokens) -> 2/3 = 0.66

        resolver = resolver_lib.Resolver(
            fuzzy_alignment_threshold=0.6,
            enable_fuzzy_alignment=True,
            accept_match_lesser=False
        )

        extractions = [
            data.Extraction(
                extraction_class="symptom",
                extraction_text="headache and fever"
            )
        ]
        source_text = "Patient reports back pain and a fever."

        aligned = list(resolver.align(
            extractions,
            source_text,
            token_offset=0
        ))

        self.assertEqual(len(aligned), 1)
        # Should match because 0.66 >= 0.6
        self.assertEqual(aligned[0].alignment_status, data.AlignmentStatus.MATCH_FUZZY)

    def test_align_uses_instance_threshold_fail(self):
        # Same case, but threshold 0.8 -> should fail
        resolver = resolver_lib.Resolver(
            fuzzy_alignment_threshold=0.8,
            enable_fuzzy_alignment=True,
            accept_match_lesser=False
        )

        extractions = [
            data.Extraction(
                extraction_class="symptom",
                extraction_text="headache and fever"
            )
        ]
        source_text = "Patient reports back pain and a fever."

        aligned = list(resolver.align(
            extractions,
            source_text,
            token_offset=0
        ))

        # Should not match
        self.assertIsNone(aligned[0].alignment_status)

if __name__ == '__main__':
    unittest.main()
