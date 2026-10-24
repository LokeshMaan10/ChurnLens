# Power BI build guide

`make exports` (or `python -m churnlens.scoring.exports`) writes clean tables to
`data/exports/<mode>/` as CSV and Parquet. Snapshot B (March 2017 expirations) is the scored
population. All money figures rest on placeholder costs, save rates and margins
(`configs/costs.yaml`, `configs/value.yaml`, `configs/prioritize.yaml`).

## 1. Load the tables

*Get data → Text/CSV* (or *Parquet*) for each file:

| Table | Grain | Key |
|---|---|---|
| `fact_customer_scores` | one row per customer | `customer_id` |
| `fact_customer_drivers` | one row per customer × driver rank (1–5) | `customer_id`, `rank` |
| `dim_customer` | one row per customer | `customer_id` |
| `kpis` | one row per headline metric | `kpi` |

Set types: `p_churn`, `cltv`, `value_at_risk`, `expected_benefit`, `contribution` → Decimal;
`on_contact_list`, `actual_churn` → True/False or Whole number; ids → Text.

## 2. Model

- `dim_customer[customer_id]` 1 → * `fact_customer_scores[customer_id]` (single direction).
- `dim_customer[customer_id]` 1 → * `fact_customer_drivers[customer_id]`.
- `kpis` stays disconnected (card visuals only).

## 3. Measures (DAX)

```DAX
Customers = COUNTROWS ( fact_customer_scores )
Mean P(churn) = AVERAGE ( fact_customer_scores[p_churn] )
Actual churn rate = AVERAGE ( fact_customer_scores[actual_churn] )
Value at risk = SUM ( fact_customer_scores[value_at_risk] )
On contact list = CALCULATE ( [Customers], fact_customer_scores[on_contact_list] = TRUE () )
Campaign spend = CALCULATE ( SUM ( fact_customer_scores[action_cost] ),
                             fact_customer_scores[on_contact_list] = TRUE () )
Expected net benefit = CALCULATE ( SUM ( fact_customer_scores[expected_benefit] ),
                                   fact_customer_scores[on_contact_list] = TRUE () )
```

## 4. Suggested pages

1. **Executive** — cards from `kpis` and the measures above; note the data mode and sample %.
2. **Risk × value** — scatter (`cltv` × `p_churn`, log Y), legend `quadrant`; matrix of
   quadrant × [Customers], [Value at risk].
3. **Actions** — bar of [Customers] by `action` filtered to the contact list; table of the
   top 50 by `priority_rank` with `top_driver_reason`.
4. **Drivers** — bar of count by `fact_customer_drivers[driver]` filtered to `rank = 1`.
5. **Segments** — `dim_customer[rfm_segment]` × [Mean P(churn)] and [Actual churn rate].

Label driver visuals "what the model relies on — associations, not causes".
