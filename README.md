# FORCE — PSI Protocol for Multi-Jurisdictional AML Deconfliction

Implementation accompanying the paper *"Threshold-Based Private Set 
Intersection Protocol for Secure Deconfliction in Multi-Jurisdictional 
Blockchain Investigations"*.

## Repository contents

- `aml_psi_tool.py` — interactive Jupyter dashboard implementing the 
  threshold PSI protocol (Phases 1–4)
- `prepare_dataset.py` — partitions the Elliptic++ dataset into n 
  jurisdictional datasets with controlled overlap structure
- `benchmark_psi.py` — scalability benchmark suite generating Figures 

## Requirements

```bash
pip install -r requirements.txt
```

## Reproducing the primary case study

1. Download the [Elliptic++ Dataset](https://github.com/git-disl/EllipticPlusPlus)
2. Run `prepare_dataset.py` to generate `addr_labels_J0.csv` ... `addr_labels_J4.csv`
3. Open `aml_psi_tool.py` in Jupyter Notebook
4. Set `n=5`, `t=2`
5. Click "Start Investigation"

## Reproducing scalability benchmarks

```bash
python benchmark_psi.py
```

Outputs Figures.

## License

MIT

## Citation
Shevchuk R, Adamyk B, Benson V. Threshold-Based Private Set Intersection Protocol for Secure Deconfliction in Multi-Jurisdictional Blockchain Investigations. Electronics. 2026; 15(12):2709. https://doi.org/10.3390/electronics15122709

## Contact
Ruslan Shevchuk - rshevchuk@ubb.edu.pl
