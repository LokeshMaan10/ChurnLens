"""Deterministic row orderings shared by analysis and feature code.

Same-day transactions tie on date. Ordering by every column means "latest plan" and
"previous plan" resolve identically on every run (clean transactions are unique on
all columns, so these orders are total within a user once is_cancel is fixed).
"""

# Latest transaction first.
TXN_LATEST_FIRST = ("transaction_date DESC, membership_expire_date DESC NULLS LAST, "
                    "actual_amount_paid DESC, plan_list_price DESC, payment_plan_days DESC, "
                    "payment_method_id, is_auto_renew DESC")

# Oldest transaction first (exact reverse of TXN_LATEST_FIRST).
TXN_OLDEST_FIRST = ("transaction_date, membership_expire_date NULLS FIRST, "
                    "actual_amount_paid, plan_list_price, payment_plan_days, "
                    "payment_method_id DESC, is_auto_renew")
