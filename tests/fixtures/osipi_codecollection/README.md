# OSIPI CodeCollection Reference Data

Reference data for cross-implementation testing of DCE-MRI pharmacokinetic models.

## Source

Parameter combinations are drawn from the OSIPI DCE-DSC-MRI CodeCollection patterns.
Concentration curves are generated using osipy's own forward models with the Parker AIF.

## Structure

```
dce/
  tofts.json           - Standard Tofts model reference cases
  extended_tofts.json  - Extended Tofts model reference cases
  patlak.json          - Patlak model reference cases
  2cxm.json            - Two-Compartment Exchange Model reference cases
  2cum.json            - Two-Compartment Uptake Model reference cases
```

## Regeneration

```bash
python3 scripts/generate_osipi_reference.py
```

## License

Same as osipy.
