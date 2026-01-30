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

# State of the field

As noted above, existing double-ended optimization methods such as NEB are implemented internally in many atomistic simulation programs. The Atomic Simulation Environment (ASE) [@larsen2017theatomi], a Python package that provides a unified interface to many atomistic simulation programs, also provides its own implementations of NEB and several of its variants. In addition, the image-dependent pair potential (IDPP) method [@smidstrup2014improved] for generating initial paths is available in software such as ORCA and ASE.

From a theoretical standpoint, DMF implemented in PyDMF differs fundamentally from the existing approaches. DMF is based on a variational formulation in which reaction-path optimization is expressed as a well-defined minimization problem with an explicit objective function. In contrast to NEB or string methods, which rely on non-variational schemes, DMF can directly leverage general-purpose nonlinear optimization algorithms. In practice, PyDMF employs the state-of-the-art optimizer IPOPT [@wachter2006ontheimp] via its Python interface cyipopt, enabling efficient and robust optimization without introducing method-specific path-update algorithms.

This design choice necessarily introduces a dependency on external optimization libraries. For this reason, PyDMF was developed as a standalone software package rather than as a contribution to an existing framework. Requiring DMF-specific optimizer dependencies, which also require the use of conda, for all users of an integrated framework such as ASE would impose unnecessary constraints on users who do not require DMF functionality. Implementing PyDMF as an independent package therefore provides a practical separation of concerns, allowing advanced variational reaction-path optimization capabilities to be offered exclusively to users who need them.

# Software design

PyDMF is designed around a clear separation between reaction-path optimization algorithms and energy evaluation backends. Its core design principle is to focus on the implementation of reaction-path optimization methods, while delegating energy and force evaluations to existing atomistic simulation software. ASE provides a well-established abstraction layer that enables this separation, and PyDMF inherits this abstraction by directly interfacing with ASE rather than reimplementing backend-specific functionality.

Through its integration with ASE, PyDMF can access a broad range of electronic-structure and force-field engines for transition-state searches. Well-established packages such as VASP, Quantum ESPRESSO, CP2K, ORCA, Gaussian, GAMESS, LAMMPS, Amber, and GROMACS can be used as backends without any modification to PyDMF. This design choice preserves backend flexibility while allowing PyDMF to integrate smoothly into diverse atomistic modeling workflows. Usage examples and API documentation are available in the project’s GitHub repository [@pydmf].

In PyDMF, the implementation focuses on defining the nonlinear optimization problem, while the actual optimization is performed using powerful external libraries. This design choice introduces certain disadvantages, such as a strong reliance on conda-based environments and the resulting difficulty of inclusion within frameworks such as ASE. Nevertheless, this trade-off was made deliberately, prioritizing improved performance and robustness of transition-state searches over minimizing external dependencies.

# Research impact statement

The research impact of PyDMF is supported by both quantitative benchmark results and its integration into user-facing computational workflows. The efficiency and robustness of DMF and FB-ENM relative to existing reaction-path optimization approaches have been demonstrated in benchmark studies reported in their original publications [@koda2024locating; @koda2024flatbott; @koda2025correlat]. Using a dataset of 121 representative chemical reactions involving typical elements [@asgeirsson2021nudgedel], DMF reduced the computational cost by approximately 70% compared with conventional NEB calculations. FB-ENM was shown to generate more energetically favorable reaction paths than IDPP and to produce chemically plausible initial paths even for complex reactions where IDPP often fails.

Beyond these benchmarks, PyDMF has been adopted as a core component in higher-level computational workflows. Notably, it serves as the transition-state search engine in ColabReaction, a web-based application for transition-state searches [@karasawa2025colabrea]. This integration demonstrates that PyDMF is not only a methodological implementation but also a practical and reusable software component that supports accessible and reliable transition-state searches.


# AI usage disclosure

ChatGPT was used during the development of PyDMF and the preparation of this manuscript. For the source code, suggested code snippets were used specifically for implementing parts of the MPI parallelization, as well as for guidance on Python package directory structure and the preparation of pyproject.toml. All such code was reviewed and tested by the authors. ChatGPT was also used interactively to assist in drafting the manuscript, the GitHub repository README, and the API documentation. In all cases, the final content was written, verified, and approved by the authors.

# Acknowledgements

This work has been supported by JSPS KAKENHI, Grant Number JP22K14652 (S-i.K.), JP21H04676, and JP23K17361 (S.S.).
The software development and benchmarking were performed using Research Center for Computational Science, Okazaki, Japan (Project: 23-IMS-C196, 24-IMS-C193, and 25-IMS-C223).

# References
