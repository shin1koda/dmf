---
title: 'PyDMF: A Python package for variational double-ended reaction-path and transition-state optimization'
tags:
  - Python
  - chemistry
  - physics
  - materials science
  - reaction path
  - transition state
authors:
  - name: Shin-ichi Koda
    orcid: 0000-0003-0993-3678
    affiliation: "1, 2"
    corresponding: true
  - name: Shinji Saito
    orcid: 0000-0003-4982-4820
    affiliation: "1, 2"
    corresponding: true
affiliations:
  - name: Institute for Molecular Science, National Institutes of Natural Sciences, Okazaki, 444-8585, Japan
    index: 1
  - name: Graduate University for Advanced Studies, SOKENDAI, Okazaki, 444-8585, Japan
    index: 2

date: 3 December 2025
bibliography: paper.bib
---

# Summary

Identifying accurate reaction paths and transition states is essential for understanding structural changes in molecular systems. PyDMF is a Python package that implements two recent methods for reaction-path optimization: the direct MaxFlux method [@koda2024locating], which improves computational efficiency through a variational formalism, and the flat-bottom elastic network model [@koda2024flatbott; @koda2025correlat], which improves the applicability of reaction-path optimization by generating chemically plausible initial paths. Through integration with the Atomic Simulation Environment, PyDMF can use a wide range of atomistic simulation software to evaluate energies along a path. Benchmark studies demonstrate that PyDMF achieves higher overall performance than existing reaction-path optimization methods.


# Statement of need

In processes such as chemical reactions, where a material moves between stable states while changing its structure, it necessarily passes through energetically unstable structures. The actual transition therefore follows a path that minimizes the rise in energy. Understanding such structural transitions requires characterizing the reaction path, particularly the transition state that forms the energy maximum along it. Because transition states are transient, they are difficult to observe experimentally and are usually obtained computationally. As a result, locating transition states with standard atomistic simulation software is a fundamental task for both theoretical and experimental researchers.

Double-ended methods, a major class of approaches for locating reaction paths and transition states, optimize the entire path between two given states. Their main advantage is that only the endpoints must be specified, eliminating the need to guess the unknown transition-state structure. Well-established techniques such as the nudged elastic band (NEB) method [@henkelman2000aclimbin; @henkelman2000improved] and the string method [@e2002stringme] are implemented in many atomistic simulation programs and are widely used. However, these methods face two key limitations: they require energy evaluations for many discrete structures along the path, reducing computational efficiency, and they are sensitive to the choice of the initial path, which limits their applicability. Overcoming these issues is crucial for accelerating computational studies.

We recently proposed two methods that substantially alleviate these limitations: the direct MaxFlux method (DMF) [@koda2024locating] and the flat-bottom elastic network model (FB-ENM) [@koda2024flatbott; @koda2025correlat]. DMF, based on a variational formalism, locates the region near the transition state using only a small number of energy evaluation points, greatly reducing the computational cost. FB-ENM generates chemically plausible initial paths by enforcing constraints that exclude nonchemical structures, enabling reliable construction of energetically favorable pathways.

PyDMF provides Python implementations of both methods [@pydmf]. Because reaction-path optimization is a fundamental component of studies across chemistry, physics, and materials science, PyDMF offers an efficient and robust framework that improves the practical accessibility of transition-state searches.

# Key features

PyDMF is designed as a flexible and interoperable framework for reaction-path optimization. Through its integration with the Atomic Simulation Environment (ASE), a Python package that provides a unified interface to many atomistic simulation programs, PyDMF can access a broad range of electronic-structure and force-field engines for transition-state searches. Well-established packages such as VASP, Quantum ESPRESSO, CP2K, ORCA, Gaussian, GAMESS, LAMMPS, Amber, and GROMACS can be used as backends without any modification to PyDMF. This backend flexibility allows PyDMF to integrate smoothly into diverse atomistic modeling workflows. Usage examples and API documentation are available in the project’s GitHub repository [@pydmf].

# Related Works

As noted above, existing double-ended optimization methods such as NEB are implemented internally in many atomistic simulation programs. ASE also provides its own implementations of NEB and several of its variants. In addition, the image-dependent pair potential (IDPP) method [@smidstrup2014improved] for generating initial paths is available in software such as ORCA and ASE.

The efficiency and robustness of DMF and FB-ENM relative to existing approaches have been demonstrated in the benchmark studies reported in their original publications [@koda2024locating; @koda2024flatbott; @koda2025correlat]. Using a dataset of 121 representative chemical reactions involving typical elements [@asgeirsson2021nudgedel], DMF reduced the computational cost by roughly 70% compared with a conventional NEB calculation. FB-ENM was shown to produce more energetically favorable paths than IDPP, and to generate chemically plausible initial paths even for complex reactions in which IDPP often fails.

PyDMF has also been integrated into other computational workflows. For example, it is employed as the transition-state search engine in ColabReaction, a web-based application for transition-state search [@karasawa2025colabrea].

# Acknowledgements

This work has been supported by JSPS KAKENHI, Grant Number JP22K14652 (S-i.K.), JP21H04676, and JP23K17361 (S.S.).
The software development and benchmarking were performed using Research Center for Computational Science, Okazaki, Japan (Project: 23-IMS-C196, 24-IMS-C193, and 25-IMS-C223).

# References
