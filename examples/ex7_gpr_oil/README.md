## The daily economic data example

Requires `pip install -e .` from the repository root.

Data of WTI crude oil prices are downloaded from [https://fred.stlouisfed.org/series/DCOILWTICO](https://fred.stlouisfed.org/series/DCOILWTICO), and data of the geopolitical risk index are downloaded from the [Geopolitical Risk (GPR) Index](https://www.matteoiacoviello.com/gpr.htm) website.

[flow_distill_prune_all.py](flow_distill_prune_all.py): runs the method for the period from 1986/01/02 to 2026/04/20.

[flow_distill_prune_phase_1986_1999.py](flow_distill_prune_phase_1986_1999.py): runs the method for the period from the 1986/01/02 to 1999/12/30.

[flow_distill_prune_phase_2000_2009.py](flow_distill_prune_phase_2000_2009.py): runs the method for the period from the 2000/01/04 to 2009/12/31.

[flow_distill_prune_phase_2010_2026.py](flow_distill_prune_phase_2010_2026.py): runs the method for the period from the 2010/01/04 to 2026/04/20.

[flow_distill_prune_phase_five_years.py](flow_distill_prune_phase_five_years.py): runs the method for one of the eight five-year periods.
