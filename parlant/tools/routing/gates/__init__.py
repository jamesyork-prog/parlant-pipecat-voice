"""
Routing gates for ticket processing.

This module contains the individual gates that make routing decisions
based on various criteria like merchant of record and product type.
"""

from .mor_gate import MORGate
from .product_type_gate import ProductTypeGate

__all__ = ['MORGate', 'ProductTypeGate']