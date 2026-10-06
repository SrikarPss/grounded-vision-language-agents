"""Synthetic dataset curation.

See ``grounded_vla.synthetic.builder.SyntheticBuilder`` for the end-to-end
pipeline and ``grounded_vla.synthetic.review`` for the two-person review
workflow described in Section 3.3 of the proposal.
"""
from .builder import SyntheticBuilder
from .review import ReviewQueue, ReviewStatus

__all__ = ["SyntheticBuilder", "ReviewQueue", "ReviewStatus"]
