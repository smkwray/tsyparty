# Data dictionary

## Canonical keys

- `date`
- `frequency`
- `source`
- `series_id`
- `sector`
- `instrument_bucket`
- `value`
- `units`
- `transform`
- `vintage`
- `notes`

## Canonical frequencies

- `daily`
- `weekly`
- `monthly`
- `quarterly`

## Canonical instrument buckets

- `all_treasuries`
- `bills`
- `nominal_coupons`
- `frns`
- `tips`

## Canonical sectors

See `configs/sectors.yml`.

## Derived dataset layers

### `raw_public`
Untouched downloads from official sources.

### `interim`
Parsed but not yet crosswalked source tables.

### `derived`
Crosswalked quarterly panels used for estimation.

### `sample`
Toy data and examples used for tests and documentation.
