# H-AMR Analysis

This is a beta version of the analysis pipeline for _H-AMR_, a new GPU-accelerated general relativistic magnetohydrodynamics (GRMHD) code optimized to leverage exascale GPU-computing architectures ([Liska et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJS..263...26L/abstract)).

_H-AMR_'s 3D adaptive mesh refinement (AMR) and adaptive time-stepping capabilities enable the code to perform state-of-the-art computations of accretion onto black holes at some of the highest resolutions achieved to date (e.g. [Liska et al. 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474L..81L/abstract), [Ripperda et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...924L..32R/abstract)). However, the enormous data sets produced by _H-AMR_ require a unique read-in pipeline that can interpret the custom binary formats used for optimal memory storage on modern supercomputers.

This pipeline, written in Python and interfaced with C++ through a Cython-based API, allows the user to read in _H-AMR_ binary data and upscale the data to the highest level of mesh refinement via a nearest-neighbor interpolation scheme. More advanced interpolation routines, specifically tailored to allow post-facto ray-tracing of the data, are currently under development. The pipeline provides support for many of _H-AMR_'s physics modules, including M1 radiation transport ([Sadowski et al. 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.429.3533S/abstract), [McKinney et al. 2014](https://ui.adsabs.harvard.edu/abs/2014MNRAS.441.3177M/abstract), [Liska et al. 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...944L..48L/abstract)), "two-temperature" thermodynamics ([Ressler et al. 2015](https://ui.adsabs.harvard.edu/abs/2015MNRAS.454.1848R/abstract); [Sadowski et al. 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..705S/abstract); [Liska et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...935L...1L/abstract)) which separately evolves ion and electron temperatures according to sub-grid schemes for allocating dissipated energy between the species, a photon-number conserving treatment of Comptonization ([Sadowski & Narayan 2015](https://ui.adsabs.harvard.edu/abs/2015MNRAS.454.2372S/abstract); [Liska et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...935L...1L/abstract)), resistive GRMHD schemes that separately evolve the electric field and magnetic field (e.g. [Ripperda et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJS..244...10R/abstract)), passive-scalar "tracer" fluids, and M1 neutrino transport ([Issa et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv241002852I/abstract)). The pipeline can be easily extended to handle additional physics modules, many of which are currently under active development.  

## Getting Started 

To use this pipeline, clone the repository with 

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc


Title
Description
How to install and run
How to use
Credits
