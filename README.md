<p align="center">
    <img align="center" src='docs/img/uu_logo.png' width="150px">
</p>

# Structure and dynamics of growing networks of Reddit threads
**[Diletta Goglia](https://orcid.org/0000-0002-2622-7495), [Davide Vega](https://orcid.org/0000-0001-8795-1957)**</br>[diletta.goglia@it.uu.se](mailto:diletta.goglia@it.uu.se) (D.G.), [davide.vega@it.uu.se](mailto:davide.vega@it.uu.se) (D.V.)


_[InfoLab](https://uuinfolab.github.io/), Department of Information Technology, Uppsala University, Uppsala, Sweden_


<a href="https://doi.org/10.1007/s41109-024-00654-y"><img src="https://zenodo.org/badge/DOI/10.1007/s41109-024-00654-y.svg" alt="DOI"></a></br>
<!--<a href="https://github.com/uuinfolab/Structure_and_dynamics_of_growing_networks_of_Reddit_threads/stargazers"><img src="https://img.shields.io/github/stars/uuinfolab/Structure_and_dynamics_of_growing_networks_of_Reddit_threads" alt="GitHub stars" /></a>
<a href="https://github.com/uuinfolab/Structure_and_dynamics_of_growing_networks_of_Reddit_threads/network/members"><img alt="GitHub forks" src="https://img.shields.io/github/forks/uuinfolab/Structure_and_dynamics_of_growing_networks_of_Reddit_threads" /></a>-->
<a href="https://github.com/uuinfolab/Structure_and_dynamics_of_growing_networks_of_Reddit_threads/blob/main/LICENSE">
  <img alt="License" src="https://img.shields.io/github/license/uuinfolab/Structure_and_dynamics_of_growing_networks_of_Reddit_threads">
</a> <a href="https://github.com/uuinfolab/Structure_and_dynamics_of_growing_networks_of_Reddit_threads/releases">
  <img alt="GitHub release date" src="https://img.shields.io/github/release-date/uuinfolab/Structure_and_dynamics_of_growing_networks_of_Reddit_threads">
</a> <a href="uuinfolab/Structure_and_dynamics_of_growing_networks_of_Reddit_threads"><img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/uuinfolab/Structure_and_dynamics_of_growing_networks_of_Reddit_threads"></a>




## Abstract
Millions of people use online social networks to reinforce their sense of belonging, for example by giving and asking for feedback as a form of social validation and self-recognition. It is common to observe disagreement among people beliefs and points of view when expressing this feedback. Modeling and analyzing such interactions is crucial to understand social phenomena that happen when people face different opinions while expressing and discussing their values. In this work, we study a Reddit community in which people participate to judge or be judged with respect to some behavior, as it represents a valuable source to study how users express judgments online. We model threads of this community as complex networks of user interactions growing in time, and we analyze the evolution of their structural properties. We show that the evolution of Reddit networks differ from other real social networks, despite falling in the same category. This happens because their global clustering coefficient is extremely small and the average shortest path length increases over time. Such properties reveal how users discuss in threads, i.e. with mostly one other user and often by a single message. We strengthen such result by analyzing the role that disagreement and reciprocity play in such conversations. We also show that Reddit thread’s evolution over time is governed by two subgraphs growing at different speeds. We discover that, in the studied community, the difference of such speed is higher than in other communities because of the user guidelines enforcing specific user interactions. Finally, we interpret the obtained results on user behavior drawing back to Social Judgment Theory.

## Directory structure 
```
ROOT
  │── src/
  │    │── utilities.py                     # useful fuctions (divided in groups based on purpose)
  │    │── AITA_data.py                     # data preparation and preprocessing functions
  │    │── network_analysis.py              # data analysis functions
  │    └── main.py                          # file to run to execute the analysis
  │── data-raw/                             #
  │    └── CSV/                             # raw data: one directory per query (metedata + texts)
  │    └── sample.csv                       # small sample of raw data
  │── data-tidy/                            #
  │    │── processed_CSV/                   # clean data + sentiment and language features -- csv files 
  │    │── recipr_in_time/                  # reciprocity metric over time -- csv files 
  │    │── threads_properties.csv           # thread statistics, structural properties and reciprocity
  │    └── ...                              # folders with networks growth at different time resolutions
  │── data-analysis/                        #  
  │    │── figs/                            # figures and files to reproduce them
  │    └── network-data/                    #
  │         │── networks_in_time/           # networks reconstruction (edgelist with structural properties) -- csv files 
  │         └── user_edgelists.csv          # user networks (as edgelist)  
  │──  docs                                 # utilities for this repo    
  │──  requirements.txt                     # packages to install      
  │──  README.md
  └──  LICENSE  
```

# Resources
Download the dataset here:  <a href="https://doi.org/10.5281/zenodo.13620016"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.13620016.svg" alt="DOI"></a>

# Quick start
Install Python:<br>
`sudo apt install python3`

Install pip:<br>
`sudo apt install --upgrade python3-pip`

Install requirements:<br>
`python -m pip install --requirement requirements.txt`

Execute [main](src/main.py):
```
cd src/
python main.py
```


# Fundings
Open access funding provided by Uppsala University. This work has been partly funded by [eSSENCE](https://www.essenceofescience.se/w/es/en), an e-Science collaboration funded as a strategic research area of Sweden. The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.

# Rights
This work is licensed under the [GNU General Public License](/LICENSE).

- If you use this **code**, please cite the following paper:
> Goglia, D., Vega, D. Structure and dynamics of growing networks of Reddit threads. Appl Netw Sci 9, 48 (2024). [10.1007/s41109-024-00654-y](https://doi.org/10.1007/s41109-024-00654-y)

```tex
@article{Goglia2024,
  author = {Diletta Goglia and Davide Vega},
  title = {Structure and dynamics of growing networks of Reddit threads},
  month = {aug},
  year = {2024},
  doi = {10.1007/s41109-024-00654-y},
  url = {https://doi.org/10.1007/s41109-024-00654-y},
  journal = {Applied Network Science},
  volume = {9},
  number = {48}}
```

- If you use the **data** included in this work, please ALSO cite the following source:

> Goglia, D. Structure and dynamics of growing networks of Reddit threads [Dataset], v1.0. Appl Netw Sci 9, 48 (2024). [10.5281/zenodo.13620016](https://doi.org/10.5281/zenodo.13620016)

```tex
@misc{Goglia2024Zenodo,
  author       = {Goglia, Diletta},
  title        = {Structure and dynamics of growing networks of Reddit threads},
  month        = {sep},
  year         = {2024},
  publisher    = {Applied Network Science},
  version      = {v1.0},
  doi          = {10.5281/zenodo.13620016},
  url          = {https://doi.org/10.5281/zenodo.13620016},
  note = {Dataset}
}
```


# Contact 
This repository is actively maintained. For any questions or further information, please feel free to contact the **corresponding author:**

Diletta Goglia <a href="https://orcid.org/0000-0002-2622-7495"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a> <br/>
Ph.D. Candidate at Uppsala University Information Laboratory [(UU-InfoLab)](https://uuinfolab.github.io/) research group. <br/>
Information Technology department, Uppsala University, Sweden. <br/>
[diletta.goglia@it.uu.se](mailto:dilettagoglia@it.uu.se) <br/>
[@dilettagoglia](https://x.com/dilettagoglia?lang=en)
<!-- [dilettagoglia.netlify.app](http://www.dilettagoglia.netlify.app) <br/> -->


---
_Last update: September 2024_